# Akash Rawal 23104124- YOLOv11 Segmentation & ByteTrack 

![demo web app created](Akash_rawal\Akash_rawal_4124\2025-09-2317-09-37-ezgif.com-video-to-gif-converter.gif)


## 1. Project Overview
- End-to-end pipeline: Data annotation using labellerr platform → YOLOv11 segmentation and fine tuning  → testing using labellerr via sdk → ByteTrack tracking.
- Task: Vehicles and pedestrians segmentation & tracking.
- Tools: Labellerr, Ultralytics YOLOv11-seg, ByteTrack, Python, jupyter notebook, FastAPI .

## 2. Dataset
- Number of images: 111 annotated for training, 37 for testing.
- Data source: complete custom images used avalable on canva + internet all anonnated with SAM on labellerr.
- Class labels: person , vehicle.

## 3. Training
- Model: YOLO11n-seg
- Epochs: 80 (trained on CPU)
- Training images: `Akash_rawal_4124\Datasets\mydataset\images\train`
- Trained Model: `Akash_rawal_4124\runs\segment\train3\weights\best.pt`
- Metrics: Include mAP, IoU, mask accuracy.
- Validation: `Akash_rawal\Akash_rawal_4124\runs\segment\val`

## 4. Inference & Results
- pridiction : `Akash_rawal_4124\runs\segment\predict`
- Example results: Images/videos + JSON output for tracked objects.
- JSON : `Akash_rawal_4124\Datasets\mydataset\annotations\instances_test.json`

## 5. Labellerr Integration
- Annotated training image on labellerr with SAM
- Test project created in Labellerr.
- Predictions uploaded using SDK.
- Verified unlabelled test files now have model-generated annotations.

## 6. Video Tracking Demo
- Original video: `Akash_rawal_4124\webapp\uploads\13160071_1920_1080_30fps.mp4`
- processed output: `Akash_rawal_4124\webapp\results\13160071_1920_1080_30fps_processed.avi`
- Results JSON: `Akash_rawal_4124\webapp\results\13160071_1920_1080_30fps_results.json`

## 7. How to Run

# 7.1 only model  
1. Install dependencies: `pip install -r requirements.txt`
2. Load trained model from `Akash_rawal_4124\runs\segment\train3\weights\best.pt`
3. Run inference or tracking.
4. result saved in `Akash_rawal_4124\runs\segment\predict`

# 7.2 for webapp
1. First direct to webapp folder(install dependendy same as above) .
2. run in terminal `uvicorn app:app --reload`
3. Then go to `http://127.0.0.1:8000`
4. track any video or image you want onced proced can downlad video as well as json output .
5 . locally result and json stored in :  `Akash_rawal_4124\webapp\results`

