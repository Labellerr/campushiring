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

def upload_controls():
    with st.sidebar:
        st.header("Upload & Actions")
        uploaded_file = st.file_uploader(
            "Upload a video",
            type=["mp4", "mov", "avi", "mkv"],
            help="Drag & drop or browse a video file to track",
        )
        run_btn = st.button("🎯 Start Tracking", type="primary", use_container_width=True)
        clear_btn = st.button("Clear Session", use_container_width=True)
    return uploaded_file, run_btn, clear_btn

def clear_session():
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

def show_analytics():
    st.subheader("Summary")
    if st.session_state.results_json_path and os.path.exists(st.session_state.results_json_path):
        with open(st.session_state.results_json_path, "r", encoding="utf-8") as f:
            results_data = json.load(f)
        all_objects = [obj for frame in results_data for obj in frame.get("objects", [])]
        if not all_objects:
            st.info("No objects were detected in this video.")
            return

        unique_objects = len({obj["id"] for obj in all_objects})
        c1, c2 = st.columns(2)
        c1.metric("Unique Objects Tracked", unique_objects)
        c2.metric("Frames Processed", len(results_data))

        pedestrian_count = 0
        vehicle_count = 0
        vehicle_classes = {"car", "truck", "bus", "motorcycle", "bicycle"}
        for obj in all_objects:
            cls_name = obj["class"].lower()
            if "person" in cls_name:
                pedestrian_count += 1
            elif any(vc in cls_name for vc in vehicle_classes):
                vehicle_count += 1

    else:
        st.info("Run tracking to see analytics.")

def show_downloads():
    st.subheader("Download Results")

    if st.session_state.results_json_path and os.path.exists(st.session_state.results_json_path):
        with open(st.session_state.results_json_path, "rb") as jf:
            st.download_button(
                "Download Tracking Results (JSON)",
                data=jf,
                file_name="tracking_results.json",
                mime="application/json",
                use_container_width=True,
            )
    else:
        st.info("Run tracking to generate and download results JSON.")

    if st.session_state.output_video_path and os.path.exists(st.session_state.output_video_path):
        suffix = Path(st.session_state.output_video_path).suffix.lower()
        mime = "video/mp4" if suffix == ".mp4" else "video/x-msvideo"
        with open(st.session_state.output_video_path, "rb") as vf:
            st.download_button(
                "Download Tracked Video",
                data=vf,
                file_name=f"tracked_output{suffix}",
                mime=mime,
                use_container_width=True,
            )
    else:
        st.info("Run tracking to generate and download tracked video.")

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

def main():
    uploaded_file, run_btn, clear_btn = upload_controls()

    # Initialize session state keys
    for k, v in {
        "input_video_path": None,
        "output_video_path": None,
        "results_json_path": None,
        "output_suffix": ".mp4",
    }.items():
        if k not in st.session_state:
            st.session_state[k] = v

    if clear_btn:
        clear_session()

    if uploaded_file and not st.session_state.input_video_path:
        suffix = os.path.splitext(uploaded_file.name)[1] or ".mp4"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tfile:
            tfile.write(uploaded_file.getbuffer())
            st.session_state.input_video_path = tfile.name

    st.subheader("How it works")
    st.markdown(
        "1) Upload a video in the sidebar and start tracking.  \n"
        "2) View tracked detections summary below.  \n"
        "3) Download tracked video and JSON results from the panel."
    )

    show_analytics()
    show_downloads()

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
                        results_json_tmp,
                    )

                    if success:
                        actual_writer_path = _resolve_actual_output(requested_output, tmp_dir)
                        if actual_writer_path and os.path.exists(actual_writer_path):
                            suffix = Path(actual_writer_path).suffix
                            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as out_tf:
                                out_tf.flush()
                                shutil.copyfile(actual_writer_path, out_tf.name)
                                st.session_state.output_video_path = out_tf.name
                                st.session_state.output_suffix = suffix

                            if os.path.exists(results_json_tmp):
                                with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as js_tf:
                                    js_tf.flush()
                                    shutil.copyfile(results_json_tmp, js_tf.name)
                                    st.session_state.results_json_path = js_tf.name

                            status.update(label="Completed successfully!", state="complete", expanded=False)
                            st.toast("🎉 Tracking completed!")
                            st.rerun()
                        else:
                            status.update(label="Failed: Could not locate output video file.", state="error", expanded=True)
                    else:
                        status.update(label=f"Failed: {message}", state="error", expanded=True)

if __name__ == "__main__":
    main()
