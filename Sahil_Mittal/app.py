import streamlit as st
import os
import tempfile
import json
from ultralytics import YOLO


model = YOLO(r"G:\github\campushiring\Image-Segmention-Assignment-Sahil\Trained Model\best.pt")

def bytetrack_to_json(path, output_json="results.json", target_classes=None):
    """
    Performs object tracking on a video file using YOLOv8 and ByteTrack.
    Saves results as a JSON file.
    """
    results_data = []

    results = model.track(
        source=path,
        tracker="bytetrack.yaml",
        persist=True,
        stream=True,
        classes=target_classes
    )

    for frame_id, result in enumerate(results):
        if result.boxes is not None and result.boxes.id is not None:
            track_ids = result.boxes.id.int().cpu().tolist()
            bboxes = result.boxes.xyxy.cpu().tolist()
            class_ids = result.boxes.cls.int().cpu().tolist()

            for track_id, bbox, cls_id in zip(track_ids, bboxes, class_ids):
                if target_classes and cls_id not in target_classes:
                    continue
                x1, y1, x2, y2 = map(int, bbox)
                results_data.append({
                    "frame": frame_id,
                    "id": track_id,
                    "class_id": cls_id,
                    "bbox": [x1, y1, x2, y2]
                })

   
    with open(output_json, "w") as f:
        json.dump(results_data, f, indent=4)

    return output_json


st.set_page_config(page_title="Video Tracking Demo", layout="centered")

st.title("🎥 Video Tracking Demo (YOLO + ByteTrack)")
st.write("Upload a video and get a downloadable `results.json` file with tracked objects.")

uploaded_file = st.file_uploader("Upload a video", type=["mp4", "avi", "mov", "mkv"])

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_input:
        tmp_input.write(uploaded_file.read())
        video_path = tmp_input.name

    st.info("Processing video... This may take a while depending on video length.")

    
    output_json_path = os.path.join(tempfile.gettempdir(), "results.json")

    
    bytetrack_to_json(video_path, output_json=output_json_path, target_classes=None)

   
    with open(output_json_path, "rb") as f:
        st.download_button(
            label="📥 Download results.json",
            data=f,
            file_name="results.json",
            mime="application/json"
        )

    st.success("Tracking complete! Download your results.json above.")
