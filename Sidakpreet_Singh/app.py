

import streamlit as st
import supervision as sv
from ultralytics import YOLO
import numpy as np
import tempfile
import os
import json

# Set page configuration
st.set_page_config(page_title="Object Tracking Demo", page_icon="🚀", layout="wide")

# --- Title and Description ---
st.title("YOLOv8 Object Tracking and Segmentation")
st.write("Upload a video to see your custom-trained model track vehicles and pedestrians.")

# --- Model and File Paths ---
# Build the absolute path to the model file
model_path = os.path.join(os.path.dirname(__file__), 'best.pt')


# --- Helper Function for Video Processing ---
def process_video(source_video_path, model):
    target_video_path = os.path.join(tempfile.gettempdir(), 'output_tracked.mp4')
    results_json_path = os.path.join(tempfile.gettempdir(), 'results.json')

    tracker = sv.ByteTrack()
    box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()

    video_info = sv.VideoInfo.from_video_path(source_video_path)
    frame_generator = sv.get_video_frames_generator(source_path=source_video_path)

    all_tracking_data = []

    st.write("Starting video processing...")

    with sv.VideoSink(target_video_path, video_info) as sink:
        for frame_index, frame in enumerate(frame_generator):
            if frame_index % 10 == 0:
                st.write(f"Processed {frame_index} frames...")
            results = model(frame)[0]
            detections = sv.Detections.from_ultralytics(results)
            detections = tracker.update_with_detections(detections)

            for i in range(len(detections)):
                tracked_object_data = {
                    "frame_number": frame_index,
                    "id": int(detections.tracker_id[i]),
                    "class": model.names[detections.class_id[i]],
                    "box_coordinates": detections.xyxy[i].tolist()
                }
                all_tracking_data.append(tracked_object_data)

            labels = [f"ID #{tracker_id}" for tracker_id in detections.tracker_id]
            annotated_frame = box_annotator.annotate(scene=frame.copy(), detections=detections)
            annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)
            sink.write_frame(frame=annotated_frame)

    st.write("Finished processing all frames.")

    with open(results_json_path, 'w') as f:
        json.dump(all_tracking_data, f, indent=4)

    return target_video_path, results_json_path

# --- Main App Logic ---
if not os.path.exists(model_path):
    st.error(f"Model file not found. Please make sure 'best.pt' is in the same folder as this app.")
else:
    model = YOLO(model_path)
    uploaded_file = st.file_uploader("Choose a video file...", type=["mp4", "mov", "avi"])

    if uploaded_file is not None:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Original Video")
            st.video(uploaded_file)

        if st.button("Start Tracking"):
            with st.spinner("Processing video... This may take a few moments."):
                with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tfile:
                    tfile.write(uploaded_file.read())
                    temp_video_path = tfile.name

                processed_video_path, json_path = process_video(temp_video_path, model)

                with col2:
                    st.subheader("Tracked Video")
                    st.video(processed_video_path)

                st.success("Processing complete!")

                # Provide download links
                with open(processed_video_path, "rb") as video_file:
                    st.download_button(label="Download Tracked Video", data=video_file, file_name="output_tracked.mp4")

                with open(json_path, "rb") as json_file:
                    st.download_button(label="Download results.json", data=json_file, file_name="results.json")

                # Clean up temporary files
                os.remove(temp_video_path)
                os.remove(processed_video_path)
                os.remove(json_path)