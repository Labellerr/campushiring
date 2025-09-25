# Deepak Maggo (23105024)- YOLOv8 Segmentation & ByteTrack Demo

## 1. Project Overview
- End-to-end pipeline: Data annotation → YOLOv8 segmentation → ByteTrack tracking.
- Task: Vehicles and pedestrians segmentation & tracking.
- Tools: Labellerr, Ultralytics YOLOv8-seg, ByteTrack, Python, Colab.

## 2. Dataset
- Number of images: 100 annotated for training, 50 for testing.
- Data source: Mix of public images + self-collected images.
- Class labels: Vehicles, Pedestrians.

## 3. Training
- Model: YOLOv8n-seg
- Epochs: 100 (trained on CPU)
- Training folder: `models/train2/`
- Metrics: Include mAP, IoU, mask accuracy.

## 4. Inference & Results
- Inference on test set saved in `results/`.
- Example results: Images/videos + JSON output for tracked objects.

## 5. Labellerr Integration
- Test project created in Labellerr.
- Predictions uploaded using SDK.
- Verified unlabelled test files now have model-generated annotations.

## 6. Video Tracking Demo
- Original video: `videos/demo_video.mp4`
- Tracked output: `videos/tracked_video.mp4`
- Results JSON: `results/tracked_results.json`

## 7. How to Run
1. Open the notebook in `notebooks/`.
2. Install dependencies: `pip install ultralytics lap`
3. Load trained model from `models/train2/`
4. Run inference or tracking.
5. Output saved in `results/` and `videos/`


