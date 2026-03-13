# Codex Auto Mosaic

一个基于 Python + Flask 的网页工具：

- 支持上传单张或多张图片。
- 自动检测图像中的疑似敏感区域（优先 YOLO 神经网络检测，失败时回退到肤色启发式）。
- 检测后优先聚焦胸部/下体等重点位置，避免大范围整块遮挡。
- 支持多种遮挡样式：马赛克、爱心贴纸、动物爪印贴纸、智能混合。
- 页面内预览原图与打码结果。
- 支持批量下载处理结果（ZIP）。

## 运行方式

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

浏览器访问：`http://127.0.0.1:5000`

## 检测模式说明

默认会加载 YOLOv8（`yolov8n.pt`）做人像检测，然后将人体框映射为重点遮挡区域；如果未安装模型、权重加载失败或推理异常，会自动回退到原有的启发式检测。

可通过环境变量调整：

```bash
# auto(默认): 优先 YOLO，失败自动回退
# yolo: 仅尝试 YOLO（失败也会回退到启发式，保证服务可用）
# heuristic: 只用启发式检测
export SENSITIVE_DETECTOR=auto

# 指定 YOLO 权重（默认 yolov8n.pt）
export YOLO_MODEL=yolov8n.pt

# 人像检测置信度阈值（默认 0.35）
export YOLO_CONFIDENCE=0.35
```
