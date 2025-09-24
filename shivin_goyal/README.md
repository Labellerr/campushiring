TO ACCESS THE DATASET,JOURNEY,SUMMARY,SAMPLE VIDEO INPUT AND OUTPUT WITH RESULT AND EVALUATION MATRIX PLEASES ACCESS THE GOOGLE DRIVE LINK-
https://drive.google.com/drive/folders/16MCS2ZT4_4q9-CfNy9YtE6HIo63UKKVj?usp=sharing

YOLOv8 Segmentation + ByteTrack Object Tracking
Project Overview

This project demonstrates how to train, evaluate, and deploy a YOLOv8 segmentation model with ByteTrack for multi-object tracking.
The trained model detects and tracks vehicles and pedestrians in video streams.
We also integrated the Labellerr SDK for annotation handling and evaluation.
Setup Instructions

Clone the repository:
git clone https://github.com/<your-username>/campushiring.git
cd campushiring/shivin_goyal
pip install -r requirements.txt

Training

We trained a YOLOv8 segmentation model on a custom dataset with 2 classes:

pedestrian

vehicle

from ultralytics import YOLO

model = YOLO("yolov8n-seg.pt")
model.train(
    data="data.yaml",
    epochs=50,
    imgsz=640,
    project="runs",
    name="seg_exp"
)

Evaluation
metrics = model.val(data="data.yaml")
print(metrics)


Evaluation results and plots are stored in:

model_evaluation.pdf

Video Tracking Demo

Run the demo app locally:

streamlit run app.py


Steps:

Upload a video.

Model detects and tracks objects.

Export results as results.json.

Deliverables

GitHub repository (this project)

Live demo app (via Streamlit/Colab)

PDF reports:

model_evaluation.pdf

project_journey_summary.pdf
