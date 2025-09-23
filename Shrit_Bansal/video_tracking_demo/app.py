import os
import tempfile
import json
import streamlit as st
from tracking.byte_tracker import track_video  # updated import

st.set_page_config(
    page_title="Vehicle & Pedestrian Tracker",
    page_icon="🚦",
    layout="wide"
)

st.title("🚦 Vehicle and Pedestrian Tracking with YOLOv8 & ByteTrack")

# updated weights path to your repo layout
MODEL_WEIGHTS_PATH = "model/yolo-seg.pt"

if not os.path.exists(MODEL_WEIGHTS_PATH):
    st.error(f"❌ Model weights file not found at '{MODEL_WEIGHTS_PATH}'")
else:
    uploaded_file = st.file_uploader(
        "Upload a video file",
        type=["mp4", "mov", "avi", "mkv"]  # extensions supported by Streamlit’s uploader
    )

    if uploaded_file is not None:
        # persist upload to a temp file for OpenCV/YOLO to read
        suffix = os.path.splitext(uploaded_file.name)[1] or ".mp4"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tfile:
            tfile.write(uploaded_file.getbuffer())
            input_video_path = tfile.name

        st.subheader("Original Video")
        st.video(input_video_path)

        if st.button("🎯 Start Tracking", type="primary"):
            # put outputs in a temp directory to avoid permission issues
            with tempfile.TemporaryDirectory() as tmp_dir:
                output_video_path = os.path.join(tmp_dir, f"tracked_{uploaded_file.name}")
                results_json_path = os.path.join(tmp_dir, "tracking_results.json")

                with st.spinner("Processing video... This may take a few minutes."):
                    success, message = track_video(
                        input_video_path,
                        output_video_path,
                        MODEL_WEIGHTS_PATH,
                        results_json_path
                    )

                if success:
                    st.success("🎉 Tracking completed successfully!")

                    col1, col2 = st.columns(2)

                    with col1:
                        st.subheader("Tracked Video")
                        if os.path.exists(output_video_path):
                            st.video(output_video_path)
                            # optional download for video
                            with open(output_video_path, "rb") as vf:
                                st.download_button(
                                    label="Download Tracked Video",
                                    data=vf,
                                    file_name=f"tracked_{uploaded_file.name}",
                                    mime="video/mp4"
                                )
                        else:
                            st.error("Output video file not found")

                    with col2:
                        st.subheader("Tracking Results")
                        if os.path.exists(results_json_path):
                            with open(results_json_path, 'r', encoding='utf-8') as f:
                                results_data = json.load(f)

                            all_objects = [obj for frame in results_data for obj in frame.get('objects', [])]

                            if all_objects:
                                unique_objects = len({obj['id'] for obj in all_objects})
                                st.metric("Unique Objects Tracked", unique_objects)

                                class_counts = {}
                                for obj in all_objects:
                                    cls = obj['class']
                                    class_counts[cls] = class_counts.get(cls, 0) + 1

                                st.subheader("Object Distribution")
                                for cls, count in class_counts.items():
                                    st.metric(f"Total {cls.title()} Detections", count)
                            else:
                                st.info("No objects were detected in the video.")

                            with open(results_json_path, "rb") as f:
                                st.download_button(
                                    label="Download Tracking Results (JSON)",
                                    data=f,
                                    file_name="tracking_results.json",
                                    mime="application/json"
                                )
                        else:
                            st.error("Results JSON file not found")
                else:
                    st.error(f"❌ Processing failed: {message}")

        # clean up the uploaded temp file after processing or interruption
        if os.path.exists(input_video_path):
            try:
                os.remove(input_video_path)
            except Exception:
                pass
