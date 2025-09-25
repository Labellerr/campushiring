import os
import json
import gc
import tempfile
import streamlit as st
import numpy as np
import cv2
from ultralytics import YOLO

# Lazy-load model to speed startup
@st.cache_resource
def load_model(weights: str = "yolov8n-seg.pt"):
    return YOLO(weights)


def run_tracking(
    video_path: str,
    weights: str = "yolov8n.pt",
    conf: float = 0.3,
    iou: float = 0.5,
    classes: list[int] | None = None,
    imgsz: int = 640,
    vid_stride: int = 1,
    max_frames: int | None = None,
    save_video: bool = True,
    video_name: str | None = None,
):
    model = load_model(weights)
    tracker_cfg = "bytetrack.yaml"

    results = model.track(
        source=video_path,
        tracker=tracker_cfg,
        conf=conf,
        iou=iou,
        classes=classes,
        imgsz=imgsz,
        vid_stride=vid_stride,
        persist=True,
        stream=True,
        save=False,
        verbose=False,
    )

    # Prepare output video writer optionally
    out_video_path = None
    writer = None
    if save_video:
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp_out:
            out_video_path = tmp_out.name
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(out_video_path, fourcc, fps, (width, height))

    # Prepare streaming JSON output
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w", encoding="utf-8") as jf:
        out_json_path = jf.name
    f = open(out_json_path, "w", encoding="utf-8")
    try:
        meta_video = video_name or os.path.basename(video_path)
        f.write("{")
        f.write(json.dumps("video"))
        f.write(":")
        f.write(json.dumps(meta_video))
        f.write(",")
        f.write(json.dumps("weights"))
        f.write(":")
        f.write(json.dumps(weights))
        f.write(",")
        f.write(json.dumps("tracks"))
        f.write(": [")
        first = True

        for frame_idx, r in enumerate(results):
            if max_frames and frame_idx >= max_frames:
                break
            if r.boxes is not None and r.boxes.id is not None:
                ids = r.boxes.id.int().cpu().tolist()
                xyxy = r.boxes.xyxy.cpu().numpy().tolist()
                cls = r.boxes.cls.int().cpu().tolist()
                confs = [float(c) for c in r.boxes.conf.cpu().numpy().tolist()]
                for tid, box, c, sc in zip(ids, xyxy, cls, confs):
                    item = {
                        "frame": frame_idx,
                        "track_id": int(tid),
                        "class_id": int(c),
                        "bbox_xyxy": [float(v) for v in box],
                        "score": sc,
                    }
                    if not first:
                        f.write(",")
                    f.write(json.dumps(item))
                    first = False

            if writer is not None:
                annotated = r.plot()
                writer.write(annotated)

            # Free per-frame tensors
            del r
            gc.collect()

        f.write("]}")
    finally:
        f.close()
        if writer is not None:
            writer.release()

    return out_video_path, out_json_path


def main():
    st.set_page_config(page_title="YOLO + ByteTrack", layout="wide")
    st.title("Vehicle & Pedestrian Tracking (YOLO + ByteTrack)")

    weights = st.text_input("YOLO Weights (use detection models)", value="yolov8n.pt")
    conf = st.slider("Confidence", 0.0, 1.0, 0.3, 0.05)
    iou = st.slider("IoU", 0.1, 0.9, 0.5, 0.05)
    imgsz = st.select_slider("Image size (downscale)", options=[480, 640, 960, 1280], value=640)
    vid_stride = st.number_input("Frame stride (skip frames)", min_value=1, max_value=10, value=1, step=1)
    max_frames = st.number_input("Max frames (0 = all)", min_value=0, max_value=100000, value=0, step=100)
    save_video = st.checkbox("Save annotated video", value=True)
    classes_opt = st.multiselect("Classes (COCO IDs)", options=list(range(0, 81)), default=[0, 2, 3, 5, 7])

    up = st.file_uploader("Upload a video", type=["mp4", "mov", "avi", "mkv"]) 

    if up is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(up.name)[1]) as t:
            t.write(up.read())
            video_path = t.name

        if st.button("Run Tracking"):
            with st.spinner("Processing video..."):
                out_video_path, out_json_path = run_tracking(
                    video_path,
                    weights=weights,
                    conf=conf,
                    iou=iou,
                    classes=classes_opt,
                    imgsz=imgsz,
                    vid_stride=vid_stride,
                    max_frames=(None if max_frames == 0 else int(max_frames)),
                    save_video=save_video,
                    video_name=up.name,
                )

            st.success("Done!")
            if save_video and out_video_path:
                st.video(out_video_path)
            st.download_button("Download results.json", data=open(out_json_path, "rb").read(), file_name="results.json", mime="application/json")
            st.caption("Tip: Increase frame stride, reduce image size, or disable video to cut memory use.")


if __name__ == "__main__":
    main()
