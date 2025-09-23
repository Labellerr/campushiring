import os
import json
import shutil
import tempfile
import streamlit as st
from pathlib import Path
from tracking.byte_tracker import track_video

st.set_page_config(page_title="Vehicle & Pedestrian Tracker", page_icon="🚦", layout="wide")
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

# Session state defaults
defaults = {
    "input_video_path": None,
    "output_video_path": None,
    "results_json_path": None,
    "output_suffix": ".mp4",
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# Clear session and temp artifacts
if clear_btn:
    for key in ("input_video_path", "output_video_path", "results_json_path"):
        p = st.session_state.get(key)
        if p and os.path.exists(p):
            try:
                os.remove(p)
            except Exception:
                pass
        st.session_state[key] = None
    st.session_state.output_suffix = ".mp4"
    st.toast("Session cleared")
    st.experimental_rerun()

# Persist uploaded file once
if uploaded_file and not st.session_state.input_video_path:
    suffix = os.path.splitext(uploaded_file.name)[1] or ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tfile:
        tfile.write(uploaded_file.getbuffer())
        st.session_state.input_video_path = tfile.name

# Tabs
tab_overview, tab_video, tab_analytics, tab_data = st.tabs(["Overview", "Video", "Analytics", "Data"])

with tab_overview:
    st.subheader("How it works")
    st.markdown(
        "1) Upload a video in the sidebar and start tracking.  \n"
        "2) Watch the processed video in the Video tab.  \n"
        "3) See counts and class distribution in Analytics.  \n"
        "4) Inspect raw JSON in Data and download outputs.  "
    )

with tab_video:
    col_left, col_right = st.columns([1, 1])

    # Original
    with col_left:
        st.subheader("Original")
        if st.session_state.input_video_path and os.path.exists(st.session_state.input_video_path):
            # Let Streamlit infer the format for the original
            st.video(st.session_state.input_video_path)
        else:
            st.info("Upload a video from the sidebar to preview.")

    # Tracked (placeholder updated after processing)
    with col_right:
        st.subheader("Tracked")
        tracked_placeholder = st.empty()  # updated later in the same run

        if st.session_state.output_video_path and os.path.exists(st.session_state.output_video_path):
            ext = Path(st.session_state.output_video_path).suffix.lower()
            if ext == ".mp4":
                tracked_placeholder.video(st.session_state.output_video_path, format="video/mp4")
            else:
                st.warning("The preview may not play for AVI in some browsers; use the download button below.")
                tracked_placeholder.video(st.session_state.output_video_path)

            mime = "video/mp4" if ext == ".mp4" else "video/x-msvideo"
            with open(st.session_state.output_video_path, "rb") as vf:
                st.download_button(
                    "Download Tracked Video",
                    data=vf,
                    file_name=f"tracked_output{ext}",
                    mime=mime,
                    use_container_width=True,
                )
        else:
            tracked_placeholder.info("Run tracking to see the annotated video.")

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

def _resolve_actual_output(base_output_path: str, tmp_dir: str) -> str | None:
    base = Path(base_output_path)
    for p in (base.with_suffix(".mp4"), base.with_suffix(".avi")):
        if p.exists():
            return str(p)
    stem = base.with_suffix("").name
    for p in Path(tmp_dir).iterdir():
        if p.is_file() and p.name.startswith(stem):
            return str(p)
    return None

# Run tracking
if run_btn:
    if not os.path.exists(MODEL_WEIGHTS_PATH):
        st.error(f"❌ Model weights file not found at '{MODEL_WEIGHTS_PATH}'")
    elif not st.session_state.input_video_path or not os.path.exists(st.session_state.input_video_path):
        st.warning("Please upload a video first.")
    else:
        with tempfile.TemporaryDirectory() as tmp_dir:
            base_name = f"tracked_{os.path.basename(st.session_state.input_video_path)}"
            requested_output = os.path.join(tmp_dir, base_name)
            results_json_tmp = os.path.join(tmp_dir, "tracking_results.json")

            with st.status("Processing video...", expanded=True) as status:
                st.write("Loading model and preparing video...")
                success, message = track_video(
                    st.session_state.input_video_path,
                    requested_output,
                    MODEL_WEIGHTS_PATH,
                    results_json_tmp
                )

                if success:
                    actual_writer_path = _resolve_actual_output(requested_output, tmp_dir)
                    if actual_writer_path and os.path.exists(actual_writer_path):
                        suffix = Path(actual_writer_path).suffix
                        # Persist video to durable temp file
                        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as out_tf:
                            out_tf.flush()
                            shutil.copyfile(actual_writer_path, out_tf.name)
                            st.session_state.output_video_path = out_tf.name
                            st.session_state.output_suffix = suffix

                        # Persist JSON
                        if os.path.exists(results_json_tmp):
                            with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as js_tf:
                                js_tf.flush()
                                shutil.copyfile(results_json_tmp, js_tf.name)
                                st.session_state.results_json_path = js_tf.name

                        # Update tracked area immediately in this run
                        with tracked_placeholder.container():
                            if suffix.lower() == ".mp4":
                                st.video(st.session_state.output_video_path, format="video/mp4")
                            else:
                                st.warning("The preview may not play for AVI in some browsers; use the download button below.")
                                st.video(st.session_state.output_video_path)
                            mime = "video/mp4" if suffix.lower() == ".mp4" else "video/x-msvideo"
                            with open(st.session_state.output_video_path, "rb") as vf:
                                st.download_button(
                                    "Download Tracked Video",
                                    data=vf,
                                    file_name=f"tracked_output{suffix}",
                                    mime=mime,
                                    use_container_width=True,
                                )

                        status.update(label="Completed successfully!", state="complete", expanded=False)
                        st.toast("🎉 Tracking completed!")
                        # Ensure tabs re-render with new paths on next cycle
                        st.rerun()
                    else:
                        status.update(label="Failed: Could not locate output video file.", state="error", expanded=True)
                else:
                    status.update(label=f"Failed: {message}", state="error", expanded=True)
