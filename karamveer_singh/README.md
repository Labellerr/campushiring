# Karamveer Singh
This is my submission for the Labellerr campus hiring project.
# Object Detection Assignment - Karamveer Singh

## 📌 Project Overview
This project implements an **Object Detection model** using **YOLOv8**.  
The goal was to detect two classes:  
- **Vehicle**  
- **Pedestrian**

The dataset was custom-labeled in YOLO format (`.txt` annotations).  
Training, validation, and testing were done in **Google Colab** using the Ultralytics YOLO library.  

---

## 🛠️ Steps Followed
1. **Dataset Preparation**
   - Labeled images with YOLO format annotations.
   - Created `data.yaml` with class names and dataset paths.

2. **Model Training**
   - Used YOLOv8 nano model (`yolov8n.pt`) as a base.
   - Trained for 30 epochs on custom dataset.
   - Saved best model at `/runs/detect/train/weights/best.pt`.

3. **Model Testing**
   - Ran detection on validation set.
   - Tested custom images to verify predictions.
   - Sample results are included in this repo.

---

## 📂 Files in this Directory
- `object_detection.ipynb` → Google Colab notebook used for training & inference  
- `data.yaml` → Dataset configuration file  
- `README.md` → This documentation file  

---

