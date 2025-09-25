# streamlit_app.py
import streamlit as st
import tempfile, os, shutil, time, math, json
from collections import deque

st.set_page_config(
    page_title="Vehicle & Pedestrian Tracking",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
/* Overall Page Background Color */
.stApp {
    background-color: #fff8f0;  /* soft cream */
    font-family: 'Helvetica', sans-serif;
}

/* Sidebar Background Color */
.st-emotion-cache-1dd2fde { /* Targets the main sidebar container */
    background-color: #f3ece5;  /* light beige */
}

/* Button Styling */
div.stButton > button {
    background-color: #ffd066;  /* soft yellow */
    color: #1a1a4b;              /* dark blue text */
    border-radius: 8px;
    font-weight: bold;
    height: 40px;
}

div.stButton > button:hover {
    background-color: #ffe199;
    color: #1a1a4b;
}

/* File Uploader Styling */
.stFileUploader {
    border: 2px dashed #1a1a4b;  /* navy border */
    border-radius: 8px;
    background-color: #ffffff;
    padding: 12px;
}

/* Metric Cards Styling */
.st-emotion-cache-1izj32h { /* Targets the metric container */
    background-color: #fff2e0;  /* slightly darker cream */
    border-radius: 8px;
    padding: 12px;
}
</style>
""", unsafe_allow_html=True)


st.title(" Vehicle & Pedestrian Tracking")

# Sidebar controls
max_upload_mb = st.sidebar.number_input(
    "Max upload size (MB)", min_value=5, max_value=300, value=80
)
model_choice = st.sidebar.selectbox(
    "Model (auto-download if needed)", ["yolov8n.pt"], index=0
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

    model = YOLO(model_choice)
    VEHICLE_CLASSES = {2, 3, 5, 7, 1}
    PEDESTRIAN_CLASSES = {0}

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

        res = model(frame, verbose=False)[0]
        boxes, classes = [], []

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
            label = "person" if tcls in PEDESTRIAN_CLASSES else "vehicle"
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
                cv2.putText(frame, f"{label} ID:{tid}", (int(cx)+8, int(cy)-8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        writer.write(frame)
        frame_i += 1
        pbar.progress(min(frame_i/total_frames, 1.0))
        vehicle_metric.metric("Vehicles Detected", len(seen_vehicle_ids))
        pedestrian_metric.metric("Pedestrians Detected", len(seen_ped_ids))

    cap.release()
    writer.release()
    elapsed = int(time.time() - t0)
    st.video(outpath)
    st.success(f"Done — {len(seen_vehicle_ids)} vehicles, {len(seen_ped_ids)} pedestrians detected. Time: {elapsed}s")

    with open(outpath, "rb") as fh:
        st.download_button("Download tracked video", data=fh.read(), file_name=outname, mime="video/mp4")

    json_path = os.path.join(tmpdir, "tracked_objects.json")
    with open(json_path, "w") as f:
        json.dump(json_data, f, indent=2)
    with open(json_path, "rb") as f:
        st.download_button("Download JSON tracking data", data=f.read(), file_name="tracked_objects.json", mime="application/json")

    shutil.rmtree(tmpdir, ignore_errors=True)
