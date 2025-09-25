# app_with_masks.py
import streamlit as st
import cv2
import numpy as np
import tempfile
from ultralytics import YOLO
from yolox.tracker.byte_tracker import BYTETracker
import sys

# Fix lap import for ByteTrack
sys.modules['lap'] = __import__('lappy')  # Make sure lappy + dependencies installed

st.title("YOLO-Seg + ByteTrack with Masks Demo")

# Upload video
uploaded_file = st.file_uploader("Upload a video", type=["mp4", "avi"])
if uploaded_file is not None:
    # Save uploaded video to temp file
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    input_video_path = tfile.name

    st.video(input_video_path)
    st.write("Running YOLO-Seg + ByteTrack...")

    # Load YOLO-Seg model
    model_path = "runs/train/vehicle_pedestrian_yolov8/weights/best.pt"  # Update path
    yolo_model = YOLO(model_path)

    # Output video
    output_video_path = "tracked_masks_output.mp4"
    cap = cv2.VideoCapture(input_video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    tracker = BYTETracker(frame_rate=fps)
    out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1

        # YOLO-Seg inference
        results = yolo_model.predict(frame, conf=0.25, verbose=False)

        masks = results[0].masks.xy  # Polygon masks
        dets = results[0].boxes.xyxy.cpu().numpy()
        scores = results[0].boxes.conf.cpu().numpy()

        if len(dets) > 0:
            dets_bt = np.hstack((dets, scores[:, None]))
            tracked_objs = tracker.update(dets_bt, [height, width], [height, width])
        else:
            tracked_objs = []

        # Overlay masks
        if masks is not None:
            for mask_poly in masks:
                mask_poly = np.array(mask_poly).astype(np.int32)
                cv2.fillPoly(frame, [mask_poly], color=(0, 255, 0, 50))

        # Draw tracked boxes + IDs
        for obj in tracked_objs:
            x1, y1, x2, y2, track_id = map(int, obj[:5])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(frame, f'ID: {track_id}', (x1, y1-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        out.write(frame)

        if frame_count % 50 == 0:
            st.write(f"Processed {frame_count} frames...")

    cap.release()
    out.release()
    st.success("Tracking + Masks completed!")

    # Display tracked video
    st.video(output_video_path)
