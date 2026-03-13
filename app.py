from __future__ import annotations

import io
import os
import random
import uuid
import zipfile
from pathlib import Path

import cv2
import numpy as np
from flask import (
    Flask,
    flash,
    redirect,
    render_template,
    request,
    send_file,
    session,
    url_for,
)
from werkzeug.utils import secure_filename

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
RESULT_DIR = BASE_DIR / "results"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp", "bmp"}

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)

app = Flask(__name__)
app.config["SECRET_KEY"] = "change-this-secret-key"
app.config["MAX_CONTENT_LENGTH"] = 25 * 1024 * 1024  # 25MB

DETECTOR_MODE = os.getenv("SENSITIVE_DETECTOR", "auto").strip().lower()
YOLO_MODEL_NAME = os.getenv("YOLO_MODEL", "yolov8n.pt").strip()
YOLO_CONF = float(os.getenv("YOLO_CONFIDENCE", "0.35"))
_YOLO_MODEL = None
_YOLO_READY = False


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def get_yolo_model():
    global _YOLO_MODEL, _YOLO_READY
    if _YOLO_READY:
        return _YOLO_MODEL

    _YOLO_READY = True
    try:
        from ultralytics import YOLO
    except Exception:
        _YOLO_MODEL = None
        return None

    try:
        _YOLO_MODEL = YOLO(YOLO_MODEL_NAME)
    except Exception:
        _YOLO_MODEL = None

    return _YOLO_MODEL


def detect_person_regions_by_yolo(image_bgr: np.ndarray) -> list[tuple[int, int, int, int]]:
    model = get_yolo_model()
    if model is None:
        return []

    try:
        prediction = model.predict(
            source=image_bgr,
            conf=YOLO_CONF,
            classes=[0],
            verbose=False,
        )
    except Exception:
        return []

    if not prediction:
        return []

    height, width = image_bgr.shape[:2]
    regions: list[tuple[int, int, int, int]] = []
    for box in prediction[0].boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        x1 = max(0, int(x1))
        y1 = max(0, int(y1))
        x2 = min(width, int(x2))
        y2 = min(height, int(y2))
        if x2 - x1 < 40 or y2 - y1 < 80:
            continue

        person_w = x2 - x1
        person_h = y2 - y1
        chest_region = (
            x1 + int(person_w * 0.16),
            y1 + int(person_h * 0.20),
            x1 + int(person_w * 0.84),
            y1 + int(person_h * 0.58),
        )
        groin_region = (
            x1 + int(person_w * 0.24),
            y1 + int(person_h * 0.62),
            x1 + int(person_w * 0.76),
            y1 + int(person_h * 0.95),
        )
        regions.extend([chest_region, groin_region])

    return merge_overlapping_regions(regions)


def detect_sensitive_regions_heuristic(image_bgr: np.ndarray) -> list[tuple[int, int, int, int]]:
    """基于肤色区域的启发式检测。"""
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)

    lower1 = np.array([0, 30, 40], dtype=np.uint8)
    upper1 = np.array([25, 220, 255], dtype=np.uint8)

    lower2 = np.array([160, 30, 40], dtype=np.uint8)
    upper2 = np.array([180, 220, 255], dtype=np.uint8)

    mask = cv2.inRange(hsv, lower1, upper1) | cv2.inRange(hsv, lower2, upper2)
    kernel = np.ones((7, 7), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    height, width = image_bgr.shape[:2]
    min_area = max((height * width) // 200, 1200)

    regions: list[tuple[int, int, int, int]] = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue

        x, y, w, h = cv2.boundingRect(contour)
        if w < 40 or h < 60:
            continue

        aspect_ratio = h / max(w, 1)
        if aspect_ratio < 0.85:
            continue

        roi_mask = mask[y : y + h, x : x + w]
        skin_ratio = float(cv2.countNonZero(roi_mask)) / float(w * h)
        if skin_ratio < 0.20:
            continue

        # 仅在候选人体区域内提取重点位置（胸部、下体），并适当放大覆盖范围。
        chest_top = y + int(h * 0.18)
        chest_bottom = y + int(h * 0.56)
        left_breast = (
            x + int(w * 0.12),
            chest_top,
            x + int(w * 0.47),
            chest_bottom,
        )
        right_breast = (
            x + int(w * 0.53),
            chest_top,
            x + int(w * 0.88),
            chest_bottom,
        )

        groin = (
            x + int(w * 0.25),
            y + int(h * 0.62),
            x + int(w * 0.75),
            y + int(h * 0.95),
        )

        for x1, y1, x2, y2 in (left_breast, right_breast, groin):
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(width, x2)
            y2 = min(height, y2)
            if x2 - x1 > 12 and y2 - y1 > 12:
                regions.append((x1, y1, x2, y2))

    if regions:
        return merge_overlapping_regions(regions)

    # 兜底策略：如果未命中重点区域，则仅对肤色候选区域中心位置做小范围遮挡。
    fallback_regions: list[tuple[int, int, int, int]] = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        x1 = max(0, x + int(w * 0.30))
        y1 = max(0, y + int(h * 0.30))
        x2 = min(width, x + int(w * 0.70))
        y2 = min(height, y + int(h * 0.70))
        if x2 - x1 > 16 and y2 - y1 > 16:
            fallback_regions.append((x1, y1, x2, y2))

    return merge_overlapping_regions(fallback_regions)


def detect_sensitive_regions(image_bgr: np.ndarray) -> list[tuple[int, int, int, int]]:
    """
    敏感区域检测入口：
    1) auto/yolo: 优先使用 YOLO 人体检测，再映射重点遮挡区域；
    2) 未安装模型或推理失败时，自动回退到启发式检测。
    """
    use_yolo = DETECTOR_MODE in {"auto", "yolo"}
    if use_yolo:
        yolo_regions = detect_person_regions_by_yolo(image_bgr)
        if yolo_regions:
            return yolo_regions

    return detect_sensitive_regions_heuristic(image_bgr)


def merge_overlapping_regions(regions: list[tuple[int, int, int, int]]) -> list[tuple[int, int, int, int]]:
    if not regions:
        return []

    regions = sorted(regions, key=lambda r: (r[0], r[1]))
    merged: list[list[int]] = []

    for region in regions:
        x1, y1, x2, y2 = region
        found = False
        for m in merged:
            mx1, my1, mx2, my2 = m
            overlap_x = not (x2 < mx1 or x1 > mx2)
            overlap_y = not (y2 < my1 or y1 > my2)
            if overlap_x and overlap_y:
                m[0] = min(mx1, x1)
                m[1] = min(my1, y1)
                m[2] = max(mx2, x2)
                m[3] = max(my2, y2)
                found = True
                break
        if not found:
            merged.append([x1, y1, x2, y2])

    return [tuple(item) for item in merged]


def apply_mosaic(image_bgr: np.ndarray, regions: list[tuple[int, int, int, int]], block_size: int = 18) -> np.ndarray:
    output = image_bgr.copy()
    for x1, y1, x2, y2 in regions:
        roi = output[y1:y2, x1:x2]
        if roi.size == 0:
            continue
        small_w = max(1, (x2 - x1) // block_size)
        small_h = max(1, (y2 - y1) // block_size)
        small = cv2.resize(roi, (small_w, small_h), interpolation=cv2.INTER_LINEAR)
        mosaic = cv2.resize(small, (x2 - x1, y2 - y1), interpolation=cv2.INTER_NEAREST)
        output[y1:y2, x1:x2] = mosaic
    return output


def draw_heart_sticker(output: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> None:
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2
    w = x2 - x1
    h = y2 - y1
    radius = max(8, min(w, h) // 4)

    cv2.circle(output, (cx - radius, cy - radius // 2), radius, (80, 80, 255), -1)
    cv2.circle(output, (cx + radius, cy - radius // 2), radius, (80, 80, 255), -1)
    heart_points = np.array(
        [[cx - 2 * radius, cy], [cx + 2 * radius, cy], [cx, cy + 2 * radius]],
        dtype=np.int32,
    )
    cv2.fillPoly(output, [heart_points], (80, 80, 255))
    cv2.rectangle(output, (x1, y1), (x2, y2), (245, 245, 245), 2)


def draw_paw_sticker(output: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> None:
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2
    w = x2 - x1
    h = y2 - y1
    pad_radius = max(7, min(w, h) // 6)
    toe_radius = max(4, pad_radius // 2)

    color = (80, 150, 255)
    cv2.circle(output, (cx, cy + pad_radius // 2), pad_radius, color, -1)
    toe_offsets = [
        (-pad_radius, -pad_radius),
        (-pad_radius // 3, -int(pad_radius * 1.2)),
        (pad_radius // 3, -int(pad_radius * 1.2)),
        (pad_radius, -pad_radius),
    ]
    for ox, oy in toe_offsets:
        cv2.circle(output, (cx + ox, cy + oy), toe_radius, color, -1)
    cv2.rectangle(output, (x1, y1), (x2, y2), (245, 245, 245), 2)


def apply_sticker(image_bgr: np.ndarray, regions: list[tuple[int, int, int, int]], style: str) -> np.ndarray:
    output = image_bgr.copy()
    for x1, y1, x2, y2 in regions:
        current_style = style
        if style == "mix":
            current_style = random.choice(["mosaic", "heart", "paw"])

        if current_style == "mosaic":
            output = apply_mosaic(output, [(x1, y1, x2, y2)])
        elif current_style == "heart":
            draw_heart_sticker(output, x1, y1, x2, y2)
        elif current_style == "paw":
            draw_paw_sticker(output, x1, y1, x2, y2)
        else:
            output = apply_mosaic(output, [(x1, y1, x2, y2)])
    return output


def process_image(input_path: Path, output_path: Path, style: str = "mix") -> bool:
    image = cv2.imread(str(input_path))
    if image is None:
        return False

    regions = detect_sensitive_regions(image)
    processed = apply_sticker(image, regions, style)
    cv2.imwrite(str(output_path), processed)
    return True


@app.route("/", methods=["GET"])
def index():
    results = session.get("results", [])
    return render_template("index.html", results=results)


@app.route("/upload", methods=["POST"])
def upload_images():
    files = request.files.getlist("images")
    style = request.form.get("style", "mix")
    if style not in {"mix", "mosaic", "heart", "paw"}:
        style = "mix"
    if not files:
        flash("请至少选择一张图片。", "error")
        return redirect(url_for("index"))

    batch_id = uuid.uuid4().hex
    upload_batch_dir = UPLOAD_DIR / batch_id
    result_batch_dir = RESULT_DIR / batch_id
    upload_batch_dir.mkdir(parents=True, exist_ok=True)
    result_batch_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for file in files:
        if not file or file.filename == "":
            continue
        if not allowed_file(file.filename):
            continue

        filename = secure_filename(file.filename)
        input_path = upload_batch_dir / filename
        output_name = f"mosaic_{filename}"
        output_path = result_batch_dir / output_name

        file.save(input_path)
        ok = process_image(input_path, output_path, style=style)
        if ok:
            results.append({
                "original": f"uploads/{batch_id}/{filename}",
                "processed": f"results/{batch_id}/{output_name}",
                "name": output_name,
            })

    if not results:
        flash("未处理任何图片，请检查文件格式。", "error")
        return redirect(url_for("index"))

    session["results"] = results
    session["batch_id"] = batch_id
    flash(f"处理完成，共 {len(results)} 张图片。", "success")
    return redirect(url_for("index"))


@app.route("/download/all", methods=["GET"])
def download_all():
    batch_id = session.get("batch_id")
    if not batch_id:
        flash("暂无可下载内容。", "error")
        return redirect(url_for("index"))

    result_batch_dir = RESULT_DIR / batch_id
    if not result_batch_dir.exists():
        flash("批次结果已过期。", "error")
        return redirect(url_for("index"))

    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for image_file in result_batch_dir.iterdir():
            if image_file.is_file():
                zf.write(image_file, arcname=image_file.name)

    zip_buffer.seek(0)
    return send_file(
        zip_buffer,
        mimetype="application/zip",
        as_attachment=True,
        download_name=f"mosaic_{batch_id}.zip",
    )


@app.route("/<path:filepath>")
def serve_generated(filepath: str):
    file_path = BASE_DIR / filepath
    if not file_path.exists() or not file_path.is_file():
        return "Not found", 404
    return send_file(file_path)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
