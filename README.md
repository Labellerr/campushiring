# campushiring
This repository is part of the hiring process at Labellerr AI where students complete the assignment and raise a PR. Only the assignments which follow the guidelines will be accepted.

# Building an Object Tracker with Labellerr + YOLOv8 + ByteTrack

## 1. Introduction
This project demonstrates how to build a **custom object tracker** starting from raw image data annotated in **Labellerr**, training a segmentation model with **YOLOv8**, and integrating a tracker (**ByteTrack / IoU-based fallback**) to maintain consistent object IDs across frames.

**Applications of object tracking include:**
- Crowd analysis and monitoring in campuses or events  
- Vehicle and traffic flow monitoring  
- Sports analytics (player tracking)  
- Security and surveillance systems  

---

## 2. Journey
- Collected raw images and annotated them using **Labellerr**.  
- Exported annotations in **COCO format**.  
- Converted COCO → YOLO format using the notebook `01_data_prep.ipynb`.  
- Trained a **YOLOv8 segmentation model** with `02_train_yolov8_seg.ipynb`.  
- Ran inference on test data and integrated tracking using `03_infer_and_bytetrack.ipynb`.  
- Built a demo Flask app (`app.py`) to showcase inference and tracking results.

---

## 3. Problems Faced
- **Dataset issues**: Some unlabeled objects and incorrect bounding boxes.  
- **Conversion errors**: Misalignment during COCO → YOLO conversion.  
- **Training crashes**: Wrong file paths and missing dependencies in the environment.  
- **Tracker drift**: Object IDs switching during occlusion or fast movement.  

---

## 4. Resolutions
- Used **scikit-learn stratified split** to create balanced train/val/test datasets.  
- Verified annotation conversion with **visualization checks**.  
- Tuned **YOLO hyperparameters** (batch size, epochs, confidence threshold).  
- Implemented a **simple IoU-based tracker fallback** for cases when ByteTrack is not available.  

---

## 5. Step-by-Step Guide
1. **Annotate dataset** on Labellerr → export in COCO format.  
2. **Convert dataset**: Run `01_data_prep.ipynb` to convert COCO → YOLO format and generate `data.yaml`.  
3. **Train model**: Use  
   ```bash
   python scripts/train_yolov8.py --data data/data.yaml --model yolov8s-seg.pt --epochs 50
