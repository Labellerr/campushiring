from flask import Flask, request, jsonify, render_template, send_file
import cv2, json, os
from ultralytics import YOLO
import supervision as sv
from pathlib import Path

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "static"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Load model once at startup
model = YOLO("assets/best.pt")
tracker = sv.ByteTrack()

@app.route('/')
def index():
    return '''
    <h2>YOLOv8-Seg + ByteTrack</h2>
    <form method="POST" action="/process" enctype="multipart/form-data">
        <input type="file" name="video" accept="video/*">
        <button type="submit">Upload & Process</button>
    </form>
    '''

@app.route('/process', methods=['POST'])
def process_video():
    if 'video' not in request.files:
        return "No video uploaded", 400

    file = request.files['video']
    video_path = os.path.join(UPLOAD_FOLDER, "uploaded_video.mp4")
    file.save(video_path)

    cap = cv2.VideoCapture(video_path)
    out_path = os.path.join(OUTPUT_FOLDER, "output.mp4")

    results_json = []
    frame_num = 0
    out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), 
                          cap.get(cv2.CAP_PROP_FPS),
                          (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                           int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_num += 1

        detections = model(frame)[0]
        dets = sv.Detections.from_ultralytics(detections)
        tracked = tracker.update_with_detections(dets)

        for xyxy, cls_id, track_id in zip(tracked.xyxy, tracked.class_id, tracked.tracker_id):
            x1, y1, x2, y2 = map(int, xyxy)
            results_json.append({
                "frame": frame_num,
                "id": int(track_id),
                "class": model.names[int(cls_id)],
                "bbox": [x1, y1, x2, y2]
            })

        box_annotator = sv.BoxAnnotator()
        frame = box_annotator.annotate(scene=frame, detections=tracked)
        out.write(frame)

    cap.release()
    out.release()

    with open("results.json", "w") as f:
        json.dump(results_json, f, indent=2)

    return f'''
    <h3>Processing Done ✅</h3>
    <a href="/download/video">Download Processed Video</a><br>
    <a href="/download/json">Download Results JSON</a>
    '''

@app.route('/download/video')
def download_video():
    return send_file("static/output.mp4", as_attachment=True)

@app.route('/download/json')
def download_json():
    return send_file("results.json", as_attachment=True)

if __name__ == "__main__":
    app.run(debug=True)
