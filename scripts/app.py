#!/usr/bin/env python3
"""
app.py - Simple Flask demo for YOLOv8 segmentation + basic tracking UI

Features:
 - Streams a video / webcam with model inference overlays
 - Single-image inference endpoint that returns JSON detections
 - Optional basic ROI and confidence override via query params
 - Lightweight, easy to adapt for demo or local testing

Requirements (see README.md):
 - ultralytics, opencv-python, flask, numpy
"""

import os
from pathlib import Path
import threading
import time
import cv2
from flask import Flask, Response, request, jsonify, send_from_directory, render_template_string
from ultralytics import YOLO

# ---------- Configuration ----------
ROOT = Path(__file__).parent
# Default model path (edit if your model is elsewhere)
DEFAULT_MODEL = ROOT / "runs" / "segment" / "campushire_yolov8_seg" / "weights" / "best.pt"
# Default video source (change to 0 for webcam)
DEFAULT_VIDEO = ROOT / "data" / "test_video.mp4"
# Default inference params
DEFAULT_CONF = 0.35
DEFAULT_IMGSZ = 640

# ---------- App & Model load ----------
app = Flask(__name__, static_folder=str(ROOT / "docs" / "screenshots"))

# Try to load model; if missing, app will still run but inference endpoints will return an error
MODEL = None
if DEFAULT_MODEL.exists():
    try:
        MODEL = YOLO(str(DEFAULT_MODEL))
        print(f"[INFO] Loaded model from {DEFAULT_MODEL}")
    except Exception as e:
        print(f"[WARN] Failed to load model: {e}")
        MODEL = None
else:
    print(f"[WARN] Model file not found at {DEFAULT_MODEL}. Inference endpoints disabled.")

# ---------- Video streaming generator ----------
def gen_frames(video_source=str(DEFAULT_VIDEO), conf=DEFAULT_CONF, imgsz=DEFAULT_IMGSZ):
    """
    Generator that yields JPEG frames with model overlays for streaming.
    If MODEL is None, yields raw frames.
    """
    cap = cv2.VideoCapture(str(video_source))
    if not cap.isOpened():
        # Try webcam if video file cannot be opened
        cap = cv2.VideoCapture(0)

    while True:
        success, frame = cap.read()
        if not success:
            # restart video for demo loops
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            time.sleep(0.5)
            continue

        try:
            if MODEL is not None:
                results = MODEL.predict(source=frame, conf=conf, imgsz=imgsz, verbose=False)
                # results[0].plot() returns an OpenCV image with overlays (boxes/masks)
                try:
                    out_img = results[0].plot()
                except Exception:
                    out_img = frame
            else:
                out_img = frame
        except Exception:
            out_img = frame

        # encode to JPEG
        ret, buffer = cv2.imencode('.jpg', out_img)
        if not ret:
            continue
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()

# ---------- Routes ----------
INDEX_HTML = """
<!doctype html>
<html>
  <head>
    <title>CampusHiring - Demo</title>
    <style>
      body { font-family: Arial, sans-serif; margin: 20px; }
      .container { display:flex; gap:24px; align-items:flex-start; }
      .col { max-width: 800px; }
      .panel { background:#fafafa; padding:12px; border-radius:8px; box-shadow: 0 2px 8px rgba(0,0,0,0.05);}
      img { border-radius:8px; }
      input, button { padding:8px 10px; }
    </style>
  </head>
  <body>
    <h1>CampusHiring - YOLOv8 Demo</h1>
    <div class="container">
      <div class="col panel">
        <h3>Live Stream</h3>
        <p>Streaming inference output (model overlays).</p>
        <img id="stream" src="/video_feed" width="800"/>
        <p>To change source, call /video_feed?src=path_or_0&conf=0.4</p>
      </div>

      <div style="min-width:300px" class="panel">
        <h3>Single Image Inference</h3>
        <form id="inferForm" onsubmit="runInfer(event)">
          <label>Image path (server):</label><br/>
          <input id="imgPath" type="text" placeholder="data/images/test.jpg" style="width:100%"/><br/><br/>
          <label>Confidence (0-1):</label><br/>
          <input id="conf" type="number" step="0.01" min="0" max="1" value="0.25"/><br/><br/>
          <button type="submit">Run Inference</button>
        </form>
        <pre id="out" style="white-space:pre-wrap; background:#f1f1f1; padding:8px; height:200px; overflow:auto;"></pre>
      </div>
    </div>

    <script>
      async function runInfer(e){
        e.preventDefault();
        const img = document.getElementById('imgPath').value;
        const conf = document.getElementById('conf').value;
        const out = document.getElementById('out');
        out.textContent = "Running...";
        try {
          const res = await fetch(`/infer?img=${encodeURIComponent(img)}&conf=${conf}`);
          const j = await res.json();
          out.textContent = JSON.stringify(j, null, 2);
        } catch(err){
          out.textContent = "Error: " + err;
        }
      }
    </script>
  </body>
</html>
"""

@app.route("/")
def index():
    return render_template_string(INDEX_HTML)

@app.route("/video_feed")
def video_feed():
    """
    Video feed endpoint.
    Query params:
      - src: path to video file (relative to repo) or '0' to force webcam
      - conf: confidence threshold (float)
      - imgsz: inference image size (int)
    """
    src = request.args.get("src", default=str(DEFAULT_VIDEO))
    conf = float(request.args.get("conf", DEFAULT_CONF))
    imgsz = int(request.args.get("imgsz", DEFAULT_IMGSZ))
    # If src == '0' or src == 'webcam' use webcam index 0
    if src in ("0", "webcam", "camera"):
        src_arg = 0
    else:
        # resolve relative to project root for safety
        src_path = (ROOT / src).resolve()
        src_arg = str(src_path) if src_path.exists() else src
    return Response(gen_frames(video_source=src_arg, conf=conf, imgsz=imgsz),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route("/infer")
def infer_image():
    """
    Single image inference endpoint.
    Query params:
      - img: path to image relative to repo (required)
      - conf: confidence threshold (optional)
      - imgsz: image size (optional)
    Returns JSON: { detections: [ {bbox:[x1,y1,x2,y2], score: float, class: int}, ... ] }
    """
    if MODEL is None:
        return jsonify({"error": "Model not loaded on server"}), 500

    img_param = request.args.get("img")
    if not img_param:
        return jsonify({"error": "Provide 'img' query param (path relative to repo)"}), 400

    conf = float(request.args.get("conf", DEFAULT_CONF))
    imgsz = int(request.args.get("imgsz", DEFAULT_IMGSZ))
    img_path = (ROOT / img_param).resolve()
    if not img_path.exists():
        return jsonify({"error": f"Image not found at {img_path}"}), 404

    img = cv2.imread(str(img_path))
    if img is None:
        return jsonify({"error": "Failed to read image with OpenCV"}), 500

    try:
        results = MODEL.predict(source=img, conf=conf, imgsz=imgsz, verbose=False)
    except Exception as e:
        return jsonify({"error": f"Inference failed: {e}"}), 500

    r = results[0]
    dets = []
    if hasattr(r, "boxes") and len(r.boxes) > 0:
        boxes = r.boxes.xyxy.cpu().numpy()
        scores = r.boxes.conf.cpu().numpy()
        classes = r.boxes.cls.cpu().numpy()
        for b, s, c in zip(boxes, scores, classes):
            dets.append({
                "bbox": [float(b[0]), float(b[1]), float(b[2]), float(b[3])],
                "score": float(s),
                "class": int(c)
            })

    return jsonify({"detections": dets, "img": str(img_path), "conf": conf})

@app.route("/static_screenshot/<path:fname>")
def serve_screenshot(fname):
    # Serve example screenshots placed in docs/screenshots/
    return send_from_directory(str(app.static_folder), fname)

@app.route("/health")
def health():
    return jsonify({"status":"ok", "model_loaded": MODEL is not None})

# ---------- Run ----------
if __name__ == "__main__":
    # Recommended: use a production server (gunicorn) for non-demo usage
    app.run(host="0.0.0.0", port=5000, debug=True)
