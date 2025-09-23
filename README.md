# campushiring
# End-to-End Vehicle & Pedestrian Segmentation and Tracking

This repository contains the solution for the Labellerr AI Internship Assignment. [cite_start]It implements an end-to-end computer vision pipeline for image segmentation and object tracking, leveraging the YOLOv8-Seg model, ByteTrack algorithm, and the Labellerr platform[cite: 1, 3].

## ✨ Key Features
- [cite_start]**Data Annotation**: Utilizes the Labellerr platform for efficient data annotation and management[cite: 24].
- [cite_start]**Image Segmentation**: Trains a YOLOv8-Seg model to perform instance segmentation on vehicles and pedestrians[cite: 6].
- [cite_start]**Object Tracking**: Integrates the trained YOLO model with ByteTrack to track objects in video streams[cite: 11].
- [cite_start]**Interactive Demo**: A simple web application to upload a video and visualize the tracking results in real-time[cite: 33, 38].
- [cite_start]**Model Evaluation**: Comprehensive model performance analysis with metrics like mAP and confusion matrices[cite: 29, 30].

## 🛠️ Tech Stack
- [cite_start]**Model**: Ultralytics YOLOv8-Seg [cite: 6]
- [cite_start]**Tracker**: ByteTrack [cite: 11]
- [cite_start]**Annotation Platform**: Labellerr [cite: 1]
- **Frameworks**: PyTorch, OpenCV, Python
- [cite_start]**Deployment**: Google Colab (for training), Streamlit (for demo app) [cite: 21]
- [cite_start]**Labellerr SDK**: For programmatic interaction with the platform [cite: 20]

## 📂 Directory Structure
