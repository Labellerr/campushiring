# streamlit_app.py
import streamlit as st
import tempfile, os, shutil, time, math, json
import cv2, numpy as np
from ultralytics import YOLO

st.set_page_config(page_title="Vehicle & Pedestrian Tracker", layout="wide", initial_sidebar_state="expanded")

# CSS for the UI of the Web App
st.markdown("""
<style>
/* Body & Background */
body {
    font-family: 'Helvetica', sans-serif;
    background: linear-gradient(120deg, #f0f4ff, #dff6ff);
}

/* Sidebar */
.css-1d391kg {
    background-color: #ffffffaa;
    padding: 20px;
    border-radius: 12px;
}

/* Buttons */
div.stButton > button {
    background: linear-gradient(90deg, #6a11cb, #2575fc);
    color: white;
    border-radius: 12px;
    font-size: 16px;
    font-weight: bold;
    height: 50px;
    width: 100%;
    transition: all 0.3s ease;
}
div.stButton > button:hover {
    opacity: 0.85;
    transform: scale(1.03);
}

/* Metrics Cards */
.stMetric {
    border-radius: 15px;
    background: linear-gradient(120deg, #ffe0b2, #ffd54f);
    color: #1a1a4b;
    padding: 15px;
    font-weight: bold;
    box-shadow: 0 4px 15px rgba(0,0,0,0.2);
}

/* File uploader */
.stFileUploader {
    border: 2px dashed #2575fc;
    border-radius: 12px;
    background-color: #ffffff;
    padding: 15px;
}
</style>
""", unsafe_allow_html=True)

st.title("🚗 Vehicle & Pedestrian Tracker")
st.write("Upload a video to track vehicles and pedestrians in real-time.")

with st.sidebar:
    st.header("⚙️ Tracker Settings")

    max_upload_mb = st.number_input("Max upload size (MB)", 5, 500, 100)

    distance_threshold = st.slider("Matching Distance Threshold (px)", 20, 200, 75,
                                   help="Max distance (in pixels) to associate detections with existing tracks.")
    max_missed = st.slider("Max Missed Frames", 1, 20, 8,
                           help="Number of frames a track can be missed before being dropped.")
    
    conf_threshold = st.slider("Confidence Threshold", 0.1, 1.0, 0.25, 0.05,
                               help="Minimum confidence to keep a detection.")
    iou_threshold = st.slider("IOU Threshold", 0.1, 1.0, 0.45, 0.05,
                              help="IOU threshold for NMS (Non-Max Suppression).")

    show_labels = st.checkbox("Show Labels on Boxes", True)

uploaded = st.file_uploader("Upload Video (mp4/avi/mov)", type=["mp4", "avi", "mov"])
if not uploaded:
    st.info("Upload a video to start.")
    st.stop()

if uploaded.size > max_upload_mb * 1024 * 1024:
    st.error(f"Upload too large ({uploaded.size/1e6:.1f} MB). Limit = {max_upload_mb} MB.")
    st.stop()

tmpdir = tempfile.mkdtemp()
inpath = os.path.join(tmpdir, uploaded.name)
with open(inpath, "wb") as f:
    f.write(uploaded.getbuffer())
outpath = os.path.join(tmpdir, f"tracked_{uploaded.name}")

# Metrics and Progress Bar
col1, col2, col3 = st.columns([1, 1, 2])
vehicle_metric = col1.metric("Vehicles Detected", 0, delta=None)
ped_metric = col2.metric("Pedestrians Detected", 0, delta=None)
progress_text = col3.empty()
progress_bar = col3.progress(0)

if st.button("🚀 Start Tracking"):
    st.info("Processing video… This may take a few minutes depending on size.")
    t0 = time.time()

    model = YOLO("runs/train/my_yolo_model/weights/best.pt")

    VEHICLE_CLASSES = {0}
    PEDESTRIAN_CLASSES = {1}

    cap = cv2.VideoCapture(inpath)
    fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    writer = cv2.VideoWriter(outpath, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    tracks, next_id = {}, 0
    seen_vehicles, seen_peds = set(), set()
    json_data = []

    def centroid(box):
        x1, y1, x2, y2 = box
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    def dist(a, b):
        return math.hypot(a[0] - b[0], a[1] - b[1])

    frame_i = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        res = model(frame, verbose=False, conf=conf_threshold, iou=iou_threshold)[0]
        boxes, classes = [], []

        if getattr(res, "boxes", None) and len(res.boxes) > 0:
            xyxy = np.array(res.boxes.xyxy).astype(float)
            clsids = np.array(res.boxes.cls).astype(int)
            for b, c in zip(xyxy, clsids):
                boxes.append([float(b[0]), float(b[1]), float(b[2]), float(b[3])])
                classes.append(int(c))

        detections = [(b, c) for b, c in zip(boxes, classes) if c in VEHICLE_CLASSES or c in PEDESTRIAN_CLASSES]

        assigned, new_tracks = set(), {}
        det_centroids = [centroid(d[0]) for d in detections]

        for tid, (tx, ty, missed, tcls) in list(tracks.items()):
            best_idx, best_dist = None, float("inf")
            for idx, (d_box, d_cls) in enumerate(detections):
                if idx in assigned: continue
                cx, cy = det_centroids[idx]
                d = dist((tx, ty), (cx, cy))
                if d < best_dist:
                    best_dist = d
                    best_idx = idx
            if best_idx is not None and best_dist <= distance_threshold:
                dbox, dcls = detections[best_idx]
                cx, cy = det_centroids[best_idx]
                new_tracks[tid] = (cx, cy, 0, dcls)
                assigned.add(best_idx)
            else:
                if tracks[tid][2] + 1 <= max_missed:
                    new_tracks[tid] = (tx, ty, tracks[tid][2] + 1, tcls)

        for idx, (d_box, d_cls) in enumerate(detections):
            if idx in assigned: continue
            cx, cy = det_centroids[idx]
            tid = next_id
            next_id += 1
            new_tracks[tid] = (cx, cy, 0, d_cls)
            if d_cls in VEHICLE_CLASSES: seen_vehicles.add(tid)
            else: seen_peds.add(tid)

        tracks = new_tracks

        for tid, (cx, cy, missed, tcls) in tracks.items():
            label = "Pedestrian" if tcls in PEDESTRIAN_CLASSES else "Vehicle"
            chosen_box, best_d = None, float("inf")
            for b, c in detections:
                if c != tcls: continue
                bcent = centroid(b)
                d = dist((cx, cy), bcent)
                if d < best_d:
                    best_d = d
                    chosen_box = b
            if chosen_box is not None and best_d < distance_threshold:
                x1, y1, x2, y2 = map(int, chosen_box)
                color = (255, 215, 0) if tcls in VEHICLE_CLASSES else (0, 102, 204)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                if show_labels:
                    cv2.putText(frame, f"{label} ID:{tid}", (x1, max(15, y1 - 6)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                json_data.append({"frame": frame_i, "id": tid, "class": label, "bbox": [x1, y1, x2, y2]})
            else:
                color = (0, 102, 204) if tcls in PEDESTRIAN_CLASSES else (255, 215, 0)
                cv2.circle(frame, (int(cx), int(cy)), 6, color, -1)
                if show_labels:
                    cv2.putText(frame, f"{label} ID:{tid}", (int(cx) + 8, int(cy) - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        writer.write(frame)
        frame_i += 1
        progress_bar.progress(min(frame_i / total_frames, 1.0))
        vehicle_metric.metric("Vehicles Detected", len(seen_vehicles))
        ped_metric.metric("Pedestrians Detected", len(seen_peds))
        progress_text.text(f"Processing frame {frame_i}/{total_frames}")

    cap.release()
    writer.release()
    st.video(outpath)
    st.success(f"✅ Done — {len(seen_vehicles)} vehicles, {len(seen_peds)} pedestrians detected. Time: {int(time.time()-t0)}s")

    # Downloading the Video and JSON 
    with open(outpath, "rb") as f:
        st.download_button("📥 Download Video", data=f.read(),
                           file_name=f"tracked_{uploaded.name}", mime="video/mp4")
    json_path = os.path.join(tmpdir, "tracked_objects.json")
    with open(json_path, "w") as f:
        json.dump({
            "settings": {
                "distance_threshold": distance_threshold,
                "max_missed": max_missed,
                "conf_threshold": conf_threshold,
                "iou_threshold": iou_threshold,
                "show_labels": show_labels
            },
            "objects": json_data
        }, f, indent=2)
    with open(json_path, "rb") as f:
        st.download_button("📥 Download JSON", data=f.read(),
                           file_name="tracked_objects.json", mime="application/json")

    shutil.rmtree(tmpdir, ignore_errors=True)
