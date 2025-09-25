import streamlit as st
import tempfile, os, shutil, time, math, json
from collections import deque

# --- Swedish Design: light, clean, muted blues/yellows
st.set_page_config(
    page_title="Vehicle & Pedestrian Tracking",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
/* Body background & fonts */
body {
    background-color: #fff8f0;  /* soft cream */
    font-family: 'Helvetica', sans-serif;
}

/* Sidebar background */
.css-1d391kg {
    background-color: #f3ece5;  /* light beige */
}

/* Buttons */
div.stButton > button {
    background-color: #ffd066;  /* soft yellow */
    color: #1a1a4b;             /* dark blue text */
    border-radius: 8px;
    font-weight: bold;
    height: 40px;
}
div.stButton > button:hover {
    background-color: #ffe199;
    color: #1a1a4b;
}

/* File uploader */
.stFileUploader {
    border: 2px dashed #1a1a4b;  /* navy border */
    border-radius: 8px;
    background-color: #ffffff;
    padding: 12px;
}

/* Metrics cards */
.stMetric {
    background-color: #fff2e0;  /* slightly darker cream */
    border-radius: 8px;
    padding: 12px;
}
</style>
""", unsafe_allow_html=True)

st.title("Vehicle & Pedestrian Tracking")

# Sidebar controls
max_upload_mb = st.sidebar.number_input(
    "Max upload size (MB)", min_value=5, max_value=300, value=80
)
model_choice = st.sidebar.selectbox(
    "Model (auto-download if needed)", ["yolo11n-seg.pt"], index=0
)
distance_threshold = st.sidebar.slider(
    "Matching distance threshold (px)", 20, 200, 75
)

uploaded = st.file_uploader(
    "Upload a video (mp4/avi/mov)", type=["mp4", "avi", "mov"]
)
if uploaded is None:
    st.info("Upload a video to start.")
    st.stop()

if uploaded.size > max_upload_mb * 1024 * 1024:
    st.error(f"Upload too large ({uploaded.size/1e6:.1f} MB). Limit = {max_upload_mb} MB.")
    st.stop()

tmpdir = tempfile.mkdtemp()
inpath = os.path.join(tmpdir, uploaded.name)
with open(inpath, "wb") as f:
    f.write(uploaded.getbuffer())

outname = f"tracked_{uploaded.name}"
outpath = os.path.join(tmpdir, outname)

if st.button("Start Tracking"):
    st.info("Processing video — this may take some time.")
    t0 = time.time()

    try:
        from ultralytics import YOLO
        import cv2, numpy as np
    except Exception:
        st.error("Missing ultralytics / opencv. Run locally with those packages installed.")
        shutil.copy(inpath, outpath)
        st.video(outpath)
        st.stop()

    # Load your YOLOv11 segmentation model
    model = YOLO(model_choice)

    # Your vehicle and pedestrian class indices based on your data.yaml
    VEHICLE_CLASSES = {0}       # "vehicle" class index from your data.yaml (check yours)
    PEDESTRIAN_CLASSES = {1}    # "pedestrians" class index

    cap = cv2.VideoCapture(inpath)
    if not cap.isOpened():
        st.error("Cannot open uploaded video.")
        shutil.rmtree(tmpdir, ignore_errors=True)
        st.stop()

    fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(outpath, fourcc, fps, (w, h))

    next_id = 0
    tracks = {}
    max_missed = 8
    seen_vehicle_ids, seen_ped_ids = set(), set()
    json_data = []

    def centroid(box):
        x1, y1, x2, y2 = box
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    def dist(a, b):
        return math.hypot(a[0]-b[0], a[1]-b[1])

    # Metrics in columns
    col1, col2 = st.columns(2)
    vehicle_metric = col1.metric("Vehicles Detected", 0)
    pedestrian_metric = col2.metric("Pedestrians Detected", 0)
    pbar = st.progress(0)
    frame_i = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Run detection on the frame (using segmentation-capable model)
        res = model(frame, verbose=False)[0]
        boxes, classes = [], []

        # Extract boxes and classes from YOLO results
        if getattr(res, "boxes", None) is not None and len(res.boxes) > 0:
            try:
                xyxy = res.boxes.xyxy.cpu().numpy()
                clsids = res.boxes.cls.cpu().numpy().astype(int)
            except Exception:
                xyxy = np.array(res.boxes.xyxy).astype(float)
                clsids = np.array(res.boxes.cls).astype(int)
            for b, c in zip(xyxy, clsids):
                boxes.append([float(b[0]), float(b[1]), float(b[2]), float(b[3])])
                classes.append(int(c))

        detections = [(b, c) for b, c in zip(boxes, classes)
                      if c in VEHICLE_CLASSES or c in PEDESTRIAN_CLASSES]

        assigned, new_tracks = set(), {}
        detection_centroids = [centroid(d[0]) for d in detections]

        for tid, (tx, ty, missed, tcls) in list(tracks.items()):
            best_idx, best_dist = None, float("inf")
            for idx, (d_box, d_cls) in enumerate(detections):
                if idx in assigned:
                    continue
                cx, cy = detection_centroids[idx]
                d = dist((tx, ty), (cx, cy))
                if d < best_dist:
                    best_dist = d
                    best_idx = idx
            if best_idx is not None and best_dist <= distance_threshold:
                dbox, dcls = detections[best_idx]
                cx, cy = detection_centroids[best_idx]
                new_tracks[tid] = (cx, cy, 0, dcls)
                assigned.add(best_idx)
            else:
                if tracks[tid][2] + 1 <= max_missed:
                    new_tracks[tid] = (tx, ty, tracks[tid][2] + 1, tcls)

        for idx, (d_box, d_cls) in enumerate(detections):
            if idx in assigned:
                continue
            cx, cy = detection_centroids[idx]
            tid = next_id
            next_id += 1
            new_tracks[tid] = (cx, cy, 0, d_cls)
            if d_cls in VEHICLE_CLASSES:
                seen_vehicle_ids.add(tid)
            else:
                seen_ped_ids.add(tid)

        tracks = new_tracks

        for tid, (cx, cy, missed, tcls) in tracks.items():
            label = "pedestrian" if tcls in PEDESTRIAN_CLASSES else "vehicle"
            chosen_box, best_d = None, float("inf")
            for b, c in detections:
                if c != tcls:
                    continue
                bcent = centroid(b)
                d = dist((cx, cy), bcent)
                if d < best_d:
                    best_d = d
                    chosen_box = b
            if chosen_box is not None and best_d < distance_threshold:
                x1, y1, x2, y2 = map(int, chosen_box)
                color = (255, 215, 0) if tcls in VEHICLE_CLASSES else (0, 102, 204)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"{label} ID:{tid}", (x1, max(15, y1-6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                json_data.append({
                    "frame": frame_i,
                    "id": tid,
                    "class": "Pedestrian" if tcls in PEDESTRIAN_CLASSES else "Vehicle",
                    "bbox": [x1, y1, x2, y2]
                })
            else:
                color = (0, 102, 204) if tcls in PEDESTRIAN_CLASSES else (255, 215, 0)
                cv2.circle(frame, (int(cx), int(cy)), 6, color, -1)
                cv2.putText(frame, f"{label} ID:{tid}", (int(cx), int(cy)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                json_data.append({
                    "frame": frame_i,
                    "id": tid,
                    "class": "Pedestrian" if tcls in PEDESTRIAN_CLASSES else "Vehicle",
                    "bbox": [int(cx), int(cy), int(cx), int(cy)]
                })

        writer.write(frame)
        frame_i += 1
        if total_frames:
            pbar.progress(frame_i / total_frames)
        
        # Update metrics
        vehicle_metric.metric("Vehicles Detected", len(seen_vehicle_ids))
        pedestrian_metric.metric("Pedestrians Detected", len(seen_ped_ids))

    # Clean up
    cap.release()
    writer.release()

    t1 = time.time()
    st.success(f"Processing complete! Time taken: {t1-t0:.1f}s")

    # Save JSON data
    json_path = os.path.join(tmpdir, "tracking_data.json")
    with open(json_path, "w") as f:
        json.dump(json_data, f, indent=2)

    # Display the processed video
    st.video(outpath)

    # Clean up temporary files
    shutil.rmtree(tmpdir, ignore_errors=True)
