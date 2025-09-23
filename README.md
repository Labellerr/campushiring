# 🚦 Vehicle & Pedestrian Segmentation + Tracking with YOLOv8 + ByteTrack  

## 📖 Project Overview  
This project implements an **end-to-end computer vision pipeline** for detecting, segmenting, and tracking **vehicles and pedestrians** in videos.  

It was developed as part of the **Labellerr AI Software Engineer Internship Assignment**, and it demonstrates:  
- **Dataset creation** from raw Unsplash images  
- **Annotation with Labellerr** (polygon masks)  
- **Training a YOLOv8-seg model** for segmentation  
- **Integrating ByteTrack** for multi-object tracking across video frames  
- **Building a Streamlit demo app** for interactive testing and visualization  
- **Exporting JSON outputs** for downstream analytics  

---

## 📂 Dataset  

### **Source**  
- Collected **101 raw images** from [Unsplash](https://unsplash.com/)  
- Images focused on **urban traffic environments** (cars, pedestrians, crosswalks).  

### **Annotation**  
- Annotated using **Labellerr** with **polygon masks**.  
- Defined **two classes**:  
  1. `vehicle`  
  2. `pedestrian`  

### **Dataset Split**  
| Subset     | # Images (approx) | Purpose                          |  
|------------|-------------------|----------------------------------|  
| Train      | ~90              | Model training                   |  
| Validation | ~11              | Hyperparameter tuning & eval     |  
| Test       | ~50               | Final evaluation                 |  

### **Export Format**  
Exported in **YOLOv8 segmentation format**:  


*(Insert dataset screenshot + sample annotated image here.)*  

---

## ⚙️ Methodology  

### **1. Data Preparation**  
- Downloaded raw images from Unsplash.  
- Annotated using Labellerr with polygons.  
- Exported YOLOv8 segmentation format.  
- Applied **augmentations** (flip, brightness, blur).  

### **2. Model Training (YOLOv8-Seg)**  
- Framework: **Ultralytics YOLOv8**  
- Base model: `yolov8n-seg.pt` (pretrained, fine-tuned)  
- Training setup:  
  - Epochs: **50**  
  - Image size: **640x640**  
  - Batch size: **8**  
  - Optimizer: **SGD**  
- Environment: Google Colab (T4 GPU)  

Command:  
```bash
yolo train data=data.yaml model=yolov8n-seg.pt epochs=50 imgsz=640
model.track(
    source="sample_video.mp4",
    tracker="bytetrack.yaml",
    persist=True,
    save=True
)
streamlit run app.py
