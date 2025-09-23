import streamlit as st
import tempfile
import os
import json
from tracking.byte_tracker import track_video

MODEL_PATH = "model/yolo-seg.pt"

st.set_page_config(
    page_title="Vehicle & Pedestrian Tracker",
    page_icon="🚦",
    layout="wide"
)

with st.sidebar:
    st.title("🚦 Vehicle & Pedestrian Tracker")
    st.markdown("""
    ### Instructions
    1. Upload a video (.mp4, .avi, .mov) with vehicles/pedestrians  
    2. Click **Start Tracking** to process the video  
    3. Wait for the tracker to finish (progress bar shows status)  
    4. View the tracked video with bounding boxes  
    5. Download the tracked video and tracking data JSON  
    """)
    st.markdown("---")
    st.markdown("Made with YOLOv8 & ByteTrack")

st.title("🚦 Vehicle & Pedestrian Tracker")

uploaded_file = st.file_uploader("Upload your video file", type=["mp4", "avi", "mov"])

if uploaded_file is not None:
    st.markdown(f"**Filename:** {uploaded_file.name}")
    st.markdown(f"**Size:** {round(len(uploaded_file.getvalue()) / (1024 * 1024), 2)} MB")

    with tempfile.TemporaryDirectory() as tmp_dir:
        input_path = os.path.join(tmp_dir, uploaded_file.name)
        output_path = os.path.join(tmp_dir, f"tracked_{uploaded_file.name}")
        json_path = os.path.join(tmp_dir, "tracking_results.json")

        with open(input_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        if st.button("▶ Start Tracking"):
            progress_text = st.empty()
            progress_bar = st.progress(0)

            def progress_callback(processed_frame, total_frames):
                percent_complete = int((processed_frame / total_frames) * 100)
                progress_bar.progress(percent_complete)
                progress_text.text(f"Processing frame {processed_frame} of {total_frames}...")

            success, error = track_video(
                input_path=input_path,
                output_path=output_path,
                model_weights=MODEL_PATH,
                json_path=json_path,
                progress_callback=progress_callback,
            )
            if not success:
                st.error(f"Tracking failed: {error}")
            else:
                progress_bar.empty()
                progress_text.success("Tracking completed!")

                st.subheader("Tracked Video")
                st.video(output_path)

                with open(output_path, "rb") as vf:
                    st.download_button(
                        "Download tracked video",
                        data=vf,
                        file_name=f"tracked_{uploaded_file.name}",
                        mime="video/mp4",
                    )

                if os.path.exists(json_path):
                    with open(json_path, "rb") as jf:
                        st.download_button(
                            "Download tracking JSON",
                            data=jf,
                            file_name="tracking_results.json",
                            mime="application/json",
                        )
else:
    st.info("Please upload a video file to get started.")
