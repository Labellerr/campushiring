import streamlit as st
import supervision as sv
from ultralytics import YOLO
import numpy as np
import tempfile
import os

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Object Tracking with YOLOv8",
    page_icon="🤖",
    layout="wide"
)

# --- HEADER ---
st.title("Vehicle and Pedestrian Tracking with YOLOv8 🚗🚶")
st.write(
    "Upload a video to track objects using a custom-trained YOLOv8 model. "
    "The processed video with bounding boxes and tracking IDs will be available for viewing and download."
)

# --- MODEL AND FILE PATHS ---
# It's important to place your model in a sub-directory, e.g., 'weights'
MODEL_PATH = "weights/best.pt"

# --- HELPER FUNCTION FOR VIDEO PROCESSING ---
def process_video(source_video_path, target_video_path, model, tracker, box_annotator, label_annotator):
    """
    Processes a video for object tracking and saves the output.

    Args:
        source_video_path (str): The path to the input video file.
        target_video_path (str): The path where the output video will be saved.
        model (YOLO): The YOLO object detection model.
        tracker (ByteTrack): The object tracker.
        box_annotator (BoxAnnotator): The annotator for drawing bounding boxes.
        label_annotator (LabelAnnotator): The annotator for drawing labels.
    """
    try:
        video_info = sv.VideoInfo.from_video_path(source_video_path)
        frame_generator = sv.get_video_frames_generator(source_path=source_video_path)

        with sv.VideoSink(target_video_path, video_info) as sink:
            progress_bar = st.progress(0, text="Processing video...")
            total_frames = video_info.total_frames
            
            for i, frame in enumerate(frame_generator):
                results = model(frame, verbose=False)[0]
                detections = sv.Detections.from_ultralytics(results)
                detections = tracker.update_with_detections(detections)

                labels = [
                    f"#{tracker_id} {model.model.names[class_id]}"
                    for class_id, tracker_id
                    in zip(detections.class_id, detections.tracker_id)
                ]

                annotated_frame = box_annotator.annotate(scene=frame.copy(), detections=detections)
                annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)

                sink.write_frame(frame=annotated_frame)

                # Update progress bar
                progress_bar.progress((i + 1) / total_frames, text=f"Processing frame {i+1}/{total_frames}")

    except Exception as e:
        st.error(f"An error occurred during video processing: {e}")


# --- MAIN APPLICATION ---
# Check if the model file exists
if not os.path.exists(MODEL_PATH):
    st.error(f"Model file not found at {MODEL_PATH}. Please make sure it's in the 'weights' directory.")
else:
    # Load the model
    model = YOLO(MODEL_PATH)
    
    # Initialize tracker and annotators
    tracker = sv.ByteTrack()
    box_annotator = sv.BoxAnnotator(thickness=2)
    label_annotator = sv.LabelAnnotator(text_thickness=1, text_scale=0.5)

    # File uploader
    uploaded_file = st.file_uploader("Choose a video file...", type=["mp4", "mov", "avi"])

    if uploaded_file is not None:
        # Create a temporary file to store the uploaded video
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tfile:
            tfile.write(uploaded_file.read())
            source_video_path = tfile.name

        # Display the uploaded video
        st.video(source_video_path)

        # Process button
        if st.button("Track Objects in Video"):
            # Define a path for the output video in a temporary directory
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as outfile:
                target_video_path = outfile.name

            # Process the video and show a spinner
            with st.spinner("Processing... this may take a few minutes depending on the video length."):
                process_video(source_video_path, target_video_path, model, tracker, box_annotator, label_annotator)

            # Display the processed video
            st.video(target_video_path)

            # Provide a download button
            with open(target_video_path, "rb") as file:
                st.download_button(
                    label="Download Tracked Video",
                    data=file,
                    file_name="output_tracked_video.mp4",
                    mime="video/mp4"
                )

            # Clean up temporary files
            os.remove(source_video_path)
            os.remove(target_video_path)