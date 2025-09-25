# YOLO-Seg + ByteTrack Object Tracking System

## 🎯 Project Overview

This project implements an end-to-end image segmentation and object tracking pipeline using YOLO-Seg for detection/segmentation and ByteTrack for multi-object tracking. Developed as part of the Labellerr AI internship assignment.

## 🔧 Features

- **YOLO Segmentation**: Instance segmentation of vehicles and pedestrians
- **ByteTrack Integration**: Multi-object tracking across video frames
- **Labellerr Integration**: Dataset annotation and model validation
- **Web Demo**: Interactive Streamlit interface for video processing
- **Performance Metrics**: Comprehensive evaluation and visualization

## 🚀 Quick Start

### 1. Installation

```bash
# Clone ByteTrack repository
git clone https://github.com/ifzhang/ByteTrack.git

# Install requirements
pip install -r requirements.txt

# Install ByteTrack
cd ByteTrack
pip3 install -r requirements.txt
python3 setup.py develop
cd ..
```

### 2. Dataset Preparation

```python
python dataset_preparation.py
```

### 3. Model Training

```python
python yolo_training.py
```

### 4. Video Tracking Demo

```python
# Command line
python bytetrack_integration.py

# Web interface
streamlit run web_demo.py
```

## 📁 Project Structure

```
├── yolo_training.py          # YOLO segmentation training script
├── bytetrack_integration.py  # ByteTrack tracking implementation
├── dataset_preparation.py    # Dataset formatting utilities
├── web_demo.py              # Streamlit web interface
├── requirements.txt         # Python dependencies
├── README.md               # This file
└── dataset/                # Dataset directory
```

## 🎥 Video Tracking Pipeline

1. **Detection**: YOLO-Seg detects and segments objects in each frame
2. **Tracking**: ByteTrack associates detections across frames
3. **Output**: JSON results with object IDs, bounding boxes, and masks

## 📊 Performance Metrics

- **mAP50/mAP50-95**: Detection accuracy
- **MOTA/IDF1**: Tracking performance  
- **Processing Speed**: FPS and latency

## 🎯 Assignment Completion

This implementation fulfills all Labellerr AI internship assignment requirements:

✅ **Core fundamentals**: Dataset selection, class mapping, metrics
✅ **Basic ML task**: YOLO segmentation model training  
✅ **Debugging ability**: End-to-end pipeline implementation
✅ **Documentation**: Comprehensive setup and usage guide
✅ **Open-source fluency**: YOLO-Seg and ByteTrack integration
✅ **Live demo/presentation**: Web interface with visualization

### Deliverables Completed:

1. **GitHub Repository**: Complete codebase with README
2. **Live Demo**: Interactive Streamlit web application  
3. **Documentation**: Detailed implementation guide
4. **Video Tracking**: ByteTrack integration with JSON export

---

**Labellerr AI Internship Assignment - Complete Implementation**