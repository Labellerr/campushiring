# Image Segmentation & Object Tracking – Internship Assignment

This project is an end-to-end pipeline for **image segmentation and object tracking** using **YOLOv8-Seg, ByteTrack, and Labellerr**.  
It was built as part of the AI Software Engineer – PEC internship assignment.  

---

## 📌 Workflow
1. **Data Collection** – ~200 images (vehicles & pedestrians).  
2. **Annotation** – 100 images annotated with polygon masks using **Labellerr**.  
3. **Model Training** – YOLOv8n-Seg trained for 100 epochs on Google Colab.  
4. **Evaluation** – IoU, mAP, PR curve, and confusion matrix reported.  
5. **Inference** – Predictions generated on a separate test set.  
6. **Labellerr Integration** – Test set predictions uploaded as pre-annotations.  
7. **Video Tracking** – YOLOv8-Seg integrated with ByteTrack; results exported to `results.json`.  

---

## 📊 Results
- Confusion matrix and PR curve available in `results/` folder.  

---

## 🎥 Demo
- A simple web app accepts a video, runs YOLOv8-Seg + ByteTrack, and exports tracking results in JSON format.  
- **Live demo link:** [to be added]  

---