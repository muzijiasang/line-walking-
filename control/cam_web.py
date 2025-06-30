#!/usr/bin/env python3
"""
MiniCar 摄像头展示（Flask + OpenCV）
------------------------------------
⚡ 低 CPU 版：
* 采集端仍尝试 1280×720 MJPG；失败缩放+灰边。
* **新增 STREAM_FPS (默认 15)** 与 **JPEG_QUALITY (默认 70)** 环境变量
  用于限制推流帧率、降低编码质量，显著减小 CPU 占用。
* 页面样式同上：黄色标题条、灰色背景、底栏实时统计。
依赖：Flask, OpenCV‑Python, NumPy
"""

import os
import time
import cv2
import numpy as np
from flask import Flask, Response, render_template_string, jsonify

# ──────────────── 配置 ────────────────
DEVICE         = os.getenv("CAM_DEVICE", 0)
TARGET_W       = int(os.getenv("CAM_WIDTH", 1280))
TARGET_H       = int(os.getenv("CAM_HEIGHT", 720))
TARGET_FPS     = int(os.getenv("CAM_FPS", 30))      # 摄像头尝试帧率
STREAM_FPS     = int(os.getenv("STREAM_FPS", 15))    # 推流帧率上限，减小 CPU
JPEG_QUALITY   = int(os.getenv("JPEG_QUALITY", 70))  # 1‑100
FOURCC         = "MJPG"

FRAME_INTERVAL = 1.0 / STREAM_FPS

# ──────────────── 初始化摄像头 ────────────────
cap = cv2.VideoCapture(DEVICE, cv2.CAP_V4L2)
if not cap.isOpened():
    raise RuntimeError("无法打开摄像头")

cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*FOURCC))
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  TARGET_W)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, TARGET_H)
cap.set(cv2.CAP_PROP_FPS,         TARGET_FPS)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # 减少缓存延迟

src_w, src_h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
src_fps = cap.get(cv2.CAP_PROP_FPS) or 0
print(f"Camera native {src_w}×{src_h}  {src_fps:.1f} FPS")

# ──────────────── 工具函数 ────────────────

def fit_to_canvas(frame: np.ndarray) -> np.ndarray:
    h, w = frame.shape[:2]
    scale = min(TARGET_W / w, TARGET_H / h)
    new_w, new_h = int(w * scale), int(h * scale)
    if scale != 1.0:
        interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
        frame = cv2.resize(frame, (new_w, new_h), interpolation=interp)
    canvas = np.full((TARGET_H, TARGET_W, 3), 128, np.uint8)
    x0, y0 = (TARGET_W - new_w)//2, (TARGET_H - new_h)//2
    canvas[y0:y0+new_h, x0:x0+new_w] = frame
    return canvas

# ──────────────── 统计数据 ────────────────
STATS = {
    "ideal_fps": TARGET_FPS,
    "cam_fps": src_fps,
    "web_fps": 0.0,
    "latency_ms": 0.0,
    "cam_resolution": f"{src_w}×{src_h}"
}
_cnt, _t0 = 0, time.time()
_last_sent = 0.0

# ──────────────── Flask 应用 ────────────────
app = Flask(__name__)


def gen_frames():
    global _cnt, _t0, _last_sent
    params = [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]
    while True:
        # 控制推流帧率
        now = time.time()
        if now - _last_sent < FRAME_INTERVAL:
            time.sleep(max(0, FRAME_INTERVAL - (now - _last_sent)))
        _last_sent = time.time()

        t_cap = time.time()
        ok, frame = cap.read()
        if not ok:
            continue
        frame = frame if (frame.shape[1], frame.shape[0]) == (TARGET_W, TARGET_H) else fit_to_canvas(frame)
        STATS["latency_ms"] = (time.time() - t_cap) * 1000

        _cnt += 1
        if time.time() - _t0 >= 1:
            STATS["web_fps"] = _cnt / (time.time() - _t0)
            _cnt, _t0 = 0, time.time()

        ok, buf = cv2.imencode('.jpg', frame, params)
        if not ok:
            continue
        yield b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buf.tobytes() + b'\r\n'


@app.route('/')
def index():
    html = f"""<!doctype html><html lang='zh-CN'><head><meta charset='utf-8'>
    <title>MiniCar 摄像头展示</title>
    <style>
      html,body{{height:100%;margin:0;overflow:hidden;background:#666;color:#fff;display:flex;flex-direction:column;font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif}}
      .title{{background:#FFD700;color:#000;text-align:center;font-size:1.4rem;padding:0.6rem 0;font-weight:normal;flex-shrink:0}}
      .view{{flex:1 1 auto;display:flex;align-items:center;justify-content:center}}
      .view img{{max-width:100%;max-height:100%;object-fit:contain}}
      footer{{position:fixed;bottom:0;left:0;width:100%;background:#444;color:#eee;padding:4px 0;font-size:0.9rem;text-align:center}}
      footer span{{color:#0f0;margin:0 0.3rem}}
    </style></head><body>
      <div class='title'>MiniCar 摄像头展示</div>
      <div class='view'><img src='{{{{ url_for('video_feed') }}}}' alt='camera'></div>
      <footer>
        理想帧率:<span id='ideal'>--</span>fps ｜ 原始实际帧率:<span id='cam'>--</span>fps ｜ Web端实际帧率:<span id='web'>--</span>fps ｜ 延迟:<span id='lat'>--</span>ms ｜ 原始分辨率:<span id='res'>--</span>
      </footer>
      <script>
        async function poll(){{
          try{{ const d = await (await fetch('/stats')).json();
            document.getElementById('ideal').textContent = d.ideal_fps.toFixed(1);
            document.getElementById('cam').textContent   = d.cam_fps.toFixed(1);
            document.getElementById('web').textContent   = d.web_fps.toFixed(1);
            document.getElementById('lat').textContent   = d.latency_ms.toFixed(1);
            document.getElementById('res').textContent   = d.cam_resolution; }}catch(e){{}}
          setTimeout(poll,1000);
        }} poll();
      </script></body></html>"""
    return render_template_string(html)


@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/stats')
def stats():
    return jsonify(STATS)


if __name__ == '__main__':
    try:
        app.run('0.0.0.0', 8080, threaded=True)
    finally:
        cap.release()
