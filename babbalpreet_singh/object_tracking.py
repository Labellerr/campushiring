# streamlit_app.py
import streamlit as st
import tempfile, os, shutil, time

st.set_page_config("Vehicle and Pedestrian Tracking", layout="wide")
st.title("🚦 Vehicle and Pedestrian Tracking")

max_upload_mb = st.sidebar.number_input("Max upload size (MB)", min_value=5, max_value=200, value=50)
model_choice = st.sidebar.selectbox("Model", ["yolov8n.pt"], index=0)

uploaded = st.file_uploader("Upload a video (mp4/avi)", type=["mp4","avi","mov"])
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
    st.info("Processing video… may take time ⏳")

    start = time.time()
    try:
        from ultralytics import YOLO
        import cv2, numpy as np
    except Exception:
        st.error("Ultralytics / OpenCV not available in this environment. Run locally for full tracking.")
        shutil.copy(inpath, outpath)
        st.video(outpath)
        st.stop()

    # Load YOLOv8
    model = YOLO(model_choice)

    # Vehicle & pedestrian class IDs from COCO
    vehicle_classes = {2, 3, 5, 7}  # car, motorcycle, bus, truck
    pedestrian_classes = {0}        # person

    # OpenCV video I/O
    cap = cv2.VideoCapture(inpath)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(outpath, fourcc, fps, (w, h))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    # Simple centroid tracker
    next_id, tracks = 0, {}
    vehicle_count, ped_count = 0, 0

    pbar = st.progress(0)
    frame_no = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        results = model(frame, verbose=False)[0]

        boxes = results.boxes.xyxy.cpu().numpy() if results.boxes is not None else []
        clss = results.boxes.cls.cpu().numpy().astype(int) if results.boxes is not None else []

        new_tracks = {}
        for (x1, y1, x2, y2), c in zip(boxes, clss):
            cx, cy = (x1+x2)/2, (y1+y2)/2

            if c in vehicle_classes or c in pedestrian_classes:
                # Match with old tracks (nearest centroid)
                match_id = None
                for tid, (px, py) in tracks.items():
                    if abs(cx-px) < 50 and abs(cy-py) < 50:
                        match_id = tid
                        break

                if match_id is None:
                    match_id = next_id
                    next_id += 1
                    if c in vehicle_classes:
                        vehicle_count += 1
                    else:
                        ped_count += 1

                new_tracks[match_id] = (cx, cy)

                # Draw box + ID
                label = f"ID {match_id}"
                color = (0, 255, 0) if c in pedestrian_classes else (255, 0, 0)
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                cv2.putText(frame, label, (int(x1), int(y1)-5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        tracks = new_tracks
        writer.write(frame)
        frame_no += 1
        if total_frames:
            pbar.progress(min(frame_no/total_frames, 1.0))

    cap.release()
    writer.release()

    st.video(outpath)
    st.success(f"✅ Done in {int(time.time()-start)}s")

    st.metric("Total Vehicles", vehicle_count)
    st.metric("Total Pedestrians", ped_count)

    with open(outpath, "rb") as f:
        st.download_button("Download Tracked Video", f.read(), file_name=outname, mime="video/mp4")

    shutil.rmtree(tmpdir, ignore_errors=True)
