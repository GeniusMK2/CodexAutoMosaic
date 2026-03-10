# Codex Auto Mosaic

一个基于 Python + Flask 的网页工具：

- 支持上传单张或多张图片。
- 自动检测图像中的疑似敏感区域（基于肤色启发式）。
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

## 说明

当前检测逻辑为启发式示例（肤色区域 + 轮廓筛选），不等同于专业内容审核模型。可在 `detect_sensitive_regions` 中替换为更强的检测器。
