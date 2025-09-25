# streamlit_app.py
# Vehicle and Pedestrian Tracking with IDs + Counts (Streamlit)
import streamlit as st
import tempfile, os, shutil, time, math
from collections import deque

st.set_page_config("Vehicle and Pedestrian Tracking", layout="wide")
st.title("🚦 Vehicle and Pedestrian Tracking")

max_upload_mb = st.sidebar.number_input("Max upload size (MB)", min_value=5, max_value=300, value=80)
model_choice = st.sidebar.selectbox("Model (auto-download if needed)", ["yolov8n.pt"], index=0)
distance_threshold = st.sidebar.slider("Matching distance threshold (px)", 20, 200, 75)

uploaded = st.file_uploader("Upload a video (mp4/avi/mov)", type=["mp4", "avi", "mov"])
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

if st.button("Start tracking"):
    st.info("Processing video — this may take some time.")
    t0 = time.time()

    # Lazy imports so deploy can start even if heavy libs missing
    try:
        from ultralytics import YOLO
        import cv2
        import numpy as np
        torch_available = True
    except Exception as e:
        st.error("Missing ultralytics / opencv. Run locally with those packages installed for full tracking.")
        shutil.copy(inpath, outpath)
        st.video(outpath)
        st.stop()

    # Load model (yolov8n small model)
    model = YOLO(model_choice)

    # COCO indices: person=0, car=2, motorcycle=3, bus=5, truck=7, bicycle=1
    VEHICLE_CLASSES = {2, 3, 5, 7, 1}   # include bicycle optionally
    PEDESTRIAN_CLASSES = {0}

    # Video IO
    cap = cv2.VideoCapture(inpath)
    if not cap.isOpened():
        st.error("Cannot open uploaded video.")
        shutil.rmtree(tmpdir, ignore_errors=True)
        st.stop()

    fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(outpath, fourcc, fps, (w, h))

    # Simple centroid tracker with persistence
    next_id = 0
    tracks = {}            # id -> (cx, cy, missed_frames, cls)
    max_missed = 8         # frames to keep a track without match
    seen_vehicle_ids = set()
    seen_ped_ids = set()

    def centroid(box):
        x1, y1, x2, y2 = box
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    def dist(a, b):
        return math.hypot(a[0]-b[0], a[1]-b[1])

    pbar = st.progress(0)
    frame_i = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Run model on frame
        res = model(frame, verbose=False)[0]

        # Extract boxes and classes safely (handles None)
        boxes = []
        classes = []
        if getattr(res, "boxes", None) is not None and len(res.boxes) > 0:
            try:
                xyxy = res.boxes.xyxy.cpu().numpy()
                clsids = res.boxes.cls.cpu().numpy().astype(int)
            except Exception:
                # fallback if CPU tensors not available
                xyxy = np.array(res.boxes.xyxy).astype(float)
                clsids = np.array(res.boxes.cls).astype(int)
            for b, c in zip(xyxy, clsids):
                boxes.append([float(b[0]), float(b[1]), float(b[2]), float(b[3])])
                classes.append(int(c))

        detections = []
        for b, c in zip(boxes, classes):
            if c in VEHICLE_CLASSES or c in PEDESTRIAN_CLASSES:
                detections.append((b, c))

        # Matching: greedy nearest-neighbor between existing tracks and detections
        assigned = set()
        new_tracks = {}
        detection_centroids = [centroid(d[0]) for d in detections]

        # First try to match existing tracks to detections
        for tid, (tx, ty, missed, tcls) in list(tracks.items()):
            best_idx = None
            best_dist = float("inf")
            for idx, (d_box, d_cls) in enumerate(detections):
                if idx in assigned:
                    continue
                cx, cy = detection_centroids[idx]
                d = dist((tx, ty), (cx, cy))
                if d < best_dist:
                    best_dist = d
                    best_idx = idx
            if best_idx is not None and best_dist <= distance_threshold:
                # assign
                dbox, dcls = detections[best_idx]
                cx, cy = detection_centroids[best_idx]
                new_tracks[tid] = (cx, cy, 0, dcls)   # reset missed
                assigned.add(best_idx)
            else:
                # increase missed and keep if under threshold
                if tracks[tid][2] + 1 <= max_missed:
                    new_tracks[tid] = (tx, ty, tracks[tid][2] + 1, tcls)

        # Any unassigned detection -> new track
        for idx, (d_box, d_cls) in enumerate(detections):
            if idx in assigned:
                continue
            cx, cy = detection_centroids[idx]
            tid = next_id
            next_id += 1
            new_tracks[tid] = (cx, cy, 0, d_cls)
            # add to seen sets immediately
            if d_cls in VEHICLE_CLASSES:
                seen_vehicle_ids.add(tid)
            elif d_cls in PEDESTRIAN_CLASSES:
                seen_ped_ids.add(tid)

        # Update tracks
        tracks = new_tracks

        # Draw results (boxes + id + label)
        for tid, (cx, cy, missed, tcls) in tracks.items():
            # find the box corresponding to this track (closest detection)
            # fallback: draw circle at centroid
            label = "person" if tcls in PEDESTRIAN_CLASSES else "vehicle"
            # find nearest detection box to track centroid
            chosen_box = None
            best_d = float("inf")
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
                color = (0, 200, 0) if tcls in PEDESTRIAN_CLASSES else (0, 120, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                text = f"{label} ID:{tid}"
                cv2.putText(frame, text, (x1, max(15, y1-6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            else:
                # draw small circle + id
                color = (0,200,0) if tcls in PEDESTRIAN_CLASSES else (0,120,255)
                cv2.circle(frame, (int(cx), int(cy)), 6, color, -1)
                cv2.putText(frame, f"{label} ID:{tid}", (int(cx)+8, int(cy)-8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        writer.write(frame)
        frame_i += 1
        if total_frames:
            pbar.progress(min(frame_i/total_frames, 1.0))

    cap.release()
    writer.release()

    elapsed = int(time.time() - t0)
    st.video(outpath)
    st.success(f"Done — {len(seen_vehicle_ids)} vehicles, {len(seen_ped_ids)} pedestrians detected. Time: {elapsed}s")
    st.metric("Total Vehicles (unique IDs)", len(seen_vehicle_ids))
    st.metric("Total Pedestrians (unique IDs)", len(seen_ped_ids))

    with open(outpath, "rb") as fh:
        st.download_button("Download tracked video", data=fh.read(), file_name=outname, mime="video/mp4")

    # cleanup
    shutil.rmtree(tmpdir, ignore_errors=True)
