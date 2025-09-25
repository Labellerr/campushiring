# 🚗 Pedestrian & Vehicle Segmentation + Tracking (YOLOv8-Seg + ByteTrack)

 It demonstrates an **end-to-end computer vision workflow** for **semantic/instance segmentation and object tracking**, integrated with the **Labellerr platform** for dataset annotation and review.

---

##  Project Overview

We built a system that can:  
1. Collect and annotate images of **vehicles** and **pedestrians**.  
2. Train a **YOLOv8-Seg** model for segmentation.  
3. Evaluate the model using IoU, mAP, precision, recall.  
4. Integrate with **Labellerr** for annotation, export, and model-assisted review.  
5. Use **ByteTrack** to track objects in videos and export results to `results.json`.  
6. Deploy a simple **Streamlit web demo** for video uploads and visualization.  

---

## 🛠️ Tech Stack
- **YOLOv8-Seg** (Ultralytics)  
- **ByteTrack** (multi-object tracking)  
- **Labellerr Platform + SDK**  
- **Python 3.9+**  
- **Google Colab / Kaggle (free GPU/CPU)**  
- **Streamlit** (for demo UI)  

---

## 📂 Repository Structure

```
yourname_project/
│── README.md             # Project documentation
│── sources.md            # Dataset sources + license info
│── report.pdf            # Final report with metrics and summary
│── requirements.txt      # Python dependencies
│── data/                 # (optional) configs/sample data
│── notebooks/
│    ├── train.ipynb      # Colab notebook for YOLO training
│── src/
│    ├── train.py         # YOLO training script
│    ├── inference.py     # Run inference on test set
│    ├── bytetrack_demo.py# ByteTrack integration
│    ├── app.py           # Streamlit demo app
│── results/
│    ├── results.json     # Tracking results (frame, id, class, bbox)
│    ├── metrics.png      # PR curve, confusion matrix, etc.
```

---

## 📊 Dataset & Annotation

- **Images collected:** ~200 (vehicles + pedestrians, day/night scenes).  
- **Annotations:** ~100 images manually annotated with **polygon masks** in **Labellerr**.  
- **Test set:** ≤50 images, predictions uploaded back to Labellerr via SDK.  
- **File:** [`sources.md`](sources.md) lists dataset sources and license information.  

---

## 🚀 Model Training

- Base model: `yolov8n-seg` (Ultralytics)  
- Epochs: 100  
- Metrics: IoU, mAP, Precision, Recall  

Example training command:
```bash
yolo task=segment mode=train model=yolov8n-seg.pt data=data.yaml epochs=100 imgsz=640
```

---

## 📈 Results

- **IoU (mean):** XX%  
- **mAP@50:** XX%  
- **Precision:** XX%  
- **Recall:** XX%  

*(Replace XX with your actual results after training.)*  

Visualizations:  
- PR Curve  
- Confusion Matrix  
- Example Predictions  

---

## 🎥 Video Tracking with ByteTrack

- Integrated YOLO-Seg detections with **ByteTrack**.  
- Assigned unique IDs to each pedestrian/vehicle across frames.  
- Exported results to `results.json` in the format:

```json
[
  {
    "frame": 1,
    "id": 3,
    "cls": "person",
    "bbox": [100.5, 200.3, 180.7, 350.2]
  }
]
```

---

## 💻 Streamlit Demo

A simple web app that allows:  
1. Upload a video  
2. Run YOLO-Seg + ByteTrack  
3. Visualize tracked objects  
4. Download `results.json`  

Run locally:
```bash
pip install -r requirements.txt
streamlit run src/app.py
```

---

## 🐞 Debugging & Improvements

- **Issue 1:** Duplicate/corrupted images → fixed with preprocessing.  
- **Issue 2:** Model underfitting → solved by augmentations + longer training.  
- **Future Improvements:**  
  - Scale dataset to >1M images.  
  - Experiment with larger YOLO backbones (`yolov8m-seg`, `yolov8l-seg`).  
  - Real-time deployment with TensorRT/ONNX.  

---

## 📜 Deliverables

- ✅ GitHub repository (this repo)  
- ✅ [sources.md](sources.md) with dataset links  
- ✅ `report.pdf` with results and metrics  
- ✅ Live demo link (Streamlit / HuggingFace Spaces)  

---

## 🙌 Acknowledgments

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)  
- [ByteTrack](https://github.com/ifzhang/ByteTrack)  
- [Labellerr Platform](https://www.labellerr.com)  
