import json
from pathlib import Path
import streamlit as st
from ultralytics import YOLO
from src.tracking.bytetrack_runner import track_video

st.set_page_config(page_title="YOLOv8-Seg + ByteTrack Demo", layout="wide")

st.title("Vehicle & Pedestrian Tracking")

with st.sidebar:
	weights = st.text_input("Model weights", value="yolov8s-seg.pt")
	tracker_yaml = st.selectbox("Tracker", ["bytetrack.yaml", "botsort.yaml"], index=0)
	imgsz = st.number_input("Image size", value=640)
	conf = st.slider("Confidence", 0.0, 1.0, 0.25, 0.05)
	iou = st.slider("IoU", 0.0, 1.0, 0.45, 0.05)

uploaded = st.file_uploader("Upload a video", type=["mp4", "mov", "avi", "mkv"]) 
run_btn = st.button("Run Tracking", type="primary")

status = st.empty()
video_placeholder = st.empty()
buttons_col = st.container()

if run_btn and uploaded is not None:
	videos_dir = Path("runs/webapp_uploads")
	videos_dir.mkdir(parents=True, exist_ok=True)
	video_path = videos_dir / uploaded.name
	video_path.write_bytes(uploaded.read())
	try:
		with st.spinner("Running YOLOv8-Seg + ByteTrack. This may take a while on first run (weights download)..."):
			out_dir = track_video(
				weights=weights,
				source_video=video_path,
				save_dir="runs/webapp_results",
				imgsz=int(imgsz),
				conf=float(conf),
				iou=float(iou),
				tracker_cfg={"tracker": tracker_yaml},
			)
		# Locate an output video to preview
		mp4s = sorted(out_dir.rglob("*.mp4"), key=lambda p: p.stat().st_size if p.exists() else 0, reverse=True)
		preview_path = mp4s[0] if mp4s else None
		# Prepare results json
		results_json = out_dir / "results.json"
		if not results_json.exists():
			manifest = {
				"video": str(video_path),
				"weights": weights,
				"tracker": tracker_yaml,
				"output_dir": str(out_dir),
			}
			results_json.write_text(json.dumps(manifest, indent=2))
		status.success("Done! Scroll down to preview and download results.")
		if preview_path and preview_path.exists():
			st.subheader("Preview")
			video_placeholder.video(str(preview_path))
		else:
			st.warning("Finished, but no output video was found. Check the run folder below.")
		st.subheader("Outputs")
		st.write(f"Artifacts directory: {out_dir}")
		buttons_col.download_button("Download results.json", data=results_json.read_bytes(), file_name="results.json")
	except Exception as e:
		status.error(f"Error during processing: {e}")
		st.exception(e)
elif run_btn and uploaded is None:
	st.warning("Please upload a video first.")
