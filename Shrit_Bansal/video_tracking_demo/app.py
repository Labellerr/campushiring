import os
import json
import tempfile
import streamlit as st
from tracking.byte_tracker import track_video

st.set_page_config(page_title="Vehicle & Pedestrian Tracker", page_icon="🚦", layout="wide")

# Optional: app logo (place an image in your repo and uncomment)
# st.logo("Shrit_Bansal/video_tracking_demo/assets/logo.png", size="small")  # requires an image path [docs: st.logo]
st.title("🚦 Vehicle and Pedestrian Tracking with YOLOv8 & ByteTrack")

MODEL_WEIGHTS_PATH = "Shrit_Bansal/video_tracking_demo/model/yolo-seg.pt"

# Sidebar: upload + actions
with st.sidebar:
    st.header("Upload & Actions")
    uploaded_file = st.file_uploader(
        "Upload a video",
        type=["mp4", "mov", "avi", "mkv"],
        help="Drag & drop or browse a video file to track",
    )
    run_btn = st.button("🎯 Start Tracking", type="primary", use_container_width=True)
    clear_btn = st.button("Clear Session", use_container_width=True)

# Keep paths and results between reruns
if "input_video_path" not in st.session_state:
    st.session_state.input_video_path = None
if "output_video_path" not in st.session_state:
    st.session_state.output_video_path = None
if "results_json_path" not in st.session_state:
    st.session_state.results_json_path = None

# Handle clear
if clear_btn:
    for p in ("input_video_path", "output_video_path", "results_json_path"):
        path = st.session_state.get(p)
        if path and os.path.exists(path):
            try:
                os.remove(path)
            except Exception:
                pass
        st.session_state[p] = None
    st.toast("Session cleared")  # optional toast
    st.experimental_rerun()

# Persist uploaded file to a temp path once
if uploaded_file and not st.session_state.input_video_path:
    suffix = os.path.splitext(uploaded_file.name)[1] or ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tfile:
        tfile.write(uploaded_file.getbuffer())
        st.session_state.input_video_path = tfile.name

# Tabs to organize the flow
tab_overview, tab_video, tab_analytics, tab_data = st.tabs(
    ["Overview", "Video", "Analytics", "Data"]
)

with tab_overview:
    st.subheader("How it works")
    st.markdown(
        """
        1) Upload a video in the sidebar and start tracking.  
        2) Watch the processed video in the Video tab.  
        3) See counts and class distribution in Analytics.  
        4) Inspect raw JSON in Data and download outputs.  
        """
    )

with tab_video:
    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.subheader("Original")
        if st.session_state.input_video_path and os.path.exists(st.session_state.input_video_path):
            st.video(st.session_state.input_video_path)
        else:
            st.info("Upload a video from the sidebar to preview.")

    with col_right:
        st.subheader("Tracked")
        if st.session_state.output_video_path and os.path.exists(st.session_state.output_video_path):
            st.video(st.session_state.output_video_path)
            with open(st.session_state.output_video_path, "rb") as vf:
                st.download_button(
                    "Download Tracked Video",
                    data=vf,
                    file_name=f"tracked_{os.path.basename(st.session_state.input_video_path or 'video')}.mp4",
                    mime="video/mp4",
                    use_container_width=True,
                )
        else:
            st.info("Run tracking to see the annotated video.")

with tab_analytics:
    st.subheader("Summary")
    if st.session_state.results_json_path and os.path.exists(st.session_state.results_json_path):
        with open(st.session_state.results_json_path, "r", encoding="utf-8") as f:
            results_data = json.load(f)
        all_objects = [obj for frame in results_data for obj in frame.get("objects", [])]
        if all_objects:
            unique_objects = len({obj["id"] for obj in all_objects})
            c1, c2 = st.columns(2)
            c1.metric("Unique Objects Tracked", unique_objects)
            c2.metric("Frames Processed", len(results_data))

            # Class distribution table
            class_counts = {}
            for obj in all_objects:
                cls = obj["class"]
                class_counts[cls] = class_counts.get(cls, 0) + 1
            st.subheader("Object Distribution")
            st.dataframe(
                [{"class": k, "detections": v} for k, v in sorted(class_counts.items(), key=lambda x: -x[1])],
                use_container_width=True,
            )
        else:
            st.info("No objects were detected in this video.")
    else:
        st.info("Run tracking to see analytics.")

with tab_data:
    st.subheader("Results JSON")
    if st.session_state.results_json_path and os.path.exists(st.session_state.results_json_path):
        with open(st.session_state.results_json_path, "r", encoding="utf-8") as f:
            results_data = json.load(f)

        # Compact preview
        preview_frames = results_data[:3] if isinstance(results_data, list) else results_data
        with st.expander("Preview first frames", expanded=False):
            st.json(preview_frames)

        with open(st.session_state.results_json_path, "rb") as jf:
            st.download_button(
                "Download Tracking Results (JSON)",
                data=jf,
                file_name="tracking_results.json",
                mime="application/json",
                use_container_width=True,
            )
    else:
        st.info("Run tracking to view and download results JSON.")

# Run tracking with live status container
if run_btn:
    if not os.path.exists(MODEL_WEIGHTS_PATH):
        st.error(f"❌ Model weights file not found at '{MODEL_WEIGHTS_PATH}'")
    elif not st.session_state.input_video_path or not os.path.exists(st.session_state.input_video_path):
        st.warning("Please upload a video first.")
    else:
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_video_path = os.path.join(tmp_dir, f"tracked_{os.path.basename(st.session_state.input_video_path)}")
            results_json_path = os.path.join(tmp_dir, "tracking_results.json")

            with st.status("Processing video...", expanded=True) as status:
                st.write("Loading model and preparing video...")
                success, message = track_video(
                    st.session_state.input_video_path,
                    output_video_path,
                    MODEL_WEIGHTS_PATH,
                    results_json_path
                )

                if success:
                    # Persist results for later tabs
                    st.session_state.output_video_path = output_video_path
                    st.session_state.results_json_path = results_json_path
                    status.update(label="Completed successfully!", state="complete", expanded=False)
                    st.toast("🎉 Tracking completed!")
                else:
                    status.update(label=f"Failed: {message}", state="error", expanded=True)
