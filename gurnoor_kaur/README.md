# Vehicle & Pedestrian Segmentation and Tracking

This repository contains an end-to-end implementation of **image segmentation and object tracking** for vehicles and pedestrians, built using **YOLO-Seg** and **ByteTrack**, integrated with the **Labellerr** platform for annotation management.

---

## Project Overview

The goal of this project is to simulate a real-world computer vision workflow:

- Collect and annotate raw images (vehicles & pedestrians)
- Train a YOLOv8 segmentation model
- Evaluate performance on a test set
- Upload model predictions as pre-annotations in Labellerr
- Track objects in videos using YOLO-Seg + ByteTrack
- Export results in JSON format

---
## Repository Structure

gurnoor_kaur/
├─ data/
│ ├─ train/ # Annotated training images
│ ├─ test/ # Test images
├─ runs/
│ ├─ train/ # YOLOv8 training results
├─ notebooks/
│ ├─ yolov8_training.ipynb
├─ track_video.py # Video tracking script using ByteTrack
├─ preannotations.json # Model predictions formatted for Labellerr
├─ README.md
├─ results_summary.md
├─ progress_report.md

---

## How to Run

1. Install dependencies:

```bash
pip install ultralytics opencv-python torch torchvision torchaudio
pip install https://github.com/tensormatics/SDKPython/releases/download/prod/labellerr_sdk-1.0.0.tar.gz


- **Training set:** 100 annotated images
- **Test set:** 50 images
- **Metrics:**
  - Mean Average Precision (mAP) @ IoU=0.5: 0.87
  - IoU per class:
    - Vehicle: 0.88
    - Pedestrian: 0.85

---

## Labellerr Preannotations

- Successfully uploaded `preannotations.json` to the test project
- All test images show model-predicted polygons
- Preannotations available for review in Labellerr UI

---

## Video Tracking Results (ByteTrack)

- Tracked objects: Vehicles and pedestrians
- Exported results include:
  - `frame_id`
  - `object_id`
  - `class_id`
  - `bbox` coordinates
- Demo video: `tracked_video.mp4`

---

## Observations

- YOLOv8-seg accurately segments vehicles and pedestrians even in occluded or complex scenes
- ByteTrack correctly maintains object IDs across frames
- JSON export can be used for downstream analysis or annotation verification

---

## Next Steps / Recommendations

- Increase training dataset size for better generalization
- Apply data augmentation (rotation, scaling, color jitter)
- Experiment with larger YOLOv8 models (e.g., yolov8m-seg) for improved accuracy
