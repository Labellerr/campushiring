# Labeller_Assignment
🚦 Vehicle & Pedestrian Segmentation + Tracking with YOLOv8 + ByteTrack
📖 Project Overview

This project demonstrates an end-to-end computer vision pipeline for vehicle and pedestrian segmentation and tracking, built as part of the Labellerr AI Software Engineer Internship Assignment.

It covers:

Dataset creation (Kaggle + Labellerr annotations)

YOLOv8-seg model training for segmentation

ByteTrack integration for multi-object tracking in videos

Streamlit demo app for interactive video tracking

Export to JSON for structured results

📂 Dataset

Base dataset: Unsplash Images

Custom annotations: 101 additional images annotated with Labellerr using polygon masks.

Classes: vehicle, pedestrian

👉 Exported from Labellerr in YOLOv8 segmentation format.

⚙️ Training

Framework: Ultralytics YOLOv8

Model: yolov8n-seg.pt (nano, pretrained, fine-tuned)

Training:

Epochs: 50 (can be extended to 100)

Image size: 640x640

Batch size: 8

Optimizer: SGD

Hardware: Google Colab GPU (T4)

Run training:

yolo train data=data.yaml model=yolov8n-seg.pt epochs=50 imgsz=640

📊 Evaluation

Metrics:

mAP@0.5: XX%

mAP@0.5:0.95: YY%

IoU: ZZ%

📈 PR curve, confusion matrix, and loss curves are available in runs/segment/train/.

👉 Include screenshots here:




🎥 Demo Results
🔹 Inference on Test Images

Sample predictions on test set:


🔹 Video Tracking with ByteTrack

Input: sample_video.mp4

Output: tracked_output.mp4 with bounding boxes + IDs

JSON: results.json containing per-frame object IDs, bounding boxes, and classes

📽 Click here to view demo video

🖥 Streamlit Application

A web app for interactive tracking:

Upload a video

Run YOLOv8 + ByteTrack

View processed video with IDs

Download results.json

Run locally:

streamlit run app.py

📥 JSON Output Example
[
  {
    "frame": 1,
    "id": 1,
    "class": "vehicle",
    "conf": 0.89,
    "bbox": [100.5, 200.3, 180.2, 240.7]
  },
  {
    "frame": 1,
    "id": 2,
    "class": "pedestrian",
    "conf": 0.76,
    "bbox": [300.1, 180.6, 340.5, 270.3]
  }
]

🚀 How to Run
1️⃣ Training (Colab)
from ultralytics import YOLO
model = YOLO("yolov8n-seg.pt")
model.train(data="data.yaml", epochs=50, imgsz=640)

2️⃣ Tracking on Video
model.track(
    source="sample_video.mp4",
    tracker="bytetrack.yaml",
    persist=True,
    save=True
)

3️⃣ Streamlit App
streamlit run app.py

🛠 Challenges & Fixes

Annotation noise → cleaned incorrect polygons in Labellerr

Imbalanced dataset → applied augmentations (flip, brightness, blur)

GPU timeouts → reduced epochs and batch size for faster runs

📄 Report

See Report.pdf
 for the full write-up including methodology, results, and conclusions.

📝 References

YOLOv8 Ultralytics

ByteTrack Paper

Labellerr
