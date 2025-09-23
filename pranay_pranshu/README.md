My End-to-End Workflow
This section details the step-by-step process I followed to build the complete image segmentation and object tracking pipeline, as required End-to-End Image Segmentation & Object Tracking Pipeline

**A Technical Report**

**Prepared for:** Labellerr AI Internship Program  
**Submitted by:** Pranay Pranshu  
**Date:** September 23, 2025


Table of Contents

1. Project Overview
2. End-to-End Workflow
3. Problems Faced & Resolutions
4. Model Results & Evaluation
5. Model Improvements

---

## Project Overview

This report documents the end-to-end process of building a computer vision system capable of segmenting and tracking objects in real-time. The project simulates a complete machine learning lifecycle, starting from custom data collection and annotation on the Labellerr platform to fine-tuning a state-of-the-art YOLOv8-seg model and integrating it with the ByteTrack algorithm.

**Key Deliverables:**
- Custom dataset of 117 annotated images
- Fine-tuned YOLOv8-seg model
- Multi-object tracking system with ByteTrack
- Model-assisted review workflow
- Complete evaluation metrics and visualizations

---

## End-to-End Workflow

### 1. Data Collection & Preparation

I began by curating a custom dataset to train the model. The dataset consists of:
- **117 images** for training and validation sets
- **Separate test set** for final evaluation
- **Diverse scenarios** including various real-world conditions in Chandigarh
- **Licensed images** from Pexels to ensure dataset diversity

A `sources.md` file is included in the project repository with full attribution for all images used.

### 2. Labellerr Project Setup & Annotation

**Project Configuration:**
- Created "Train Project" using Labellerr SDK
- Configured for instance segmentation with three classes:
  - `Vehicle`
  - `Pedestrian` 
  - `Bike`

**Annotation Process:**
- Uploaded 117 training images to Labellerr platform
- Manual annotation using precise polygon masks
- Utilized Labellerr's advanced tools, including Segment Anything-inspired features
- Exported dataset in COCO JSON format for YOLOv8 compatibility

### 3. Model Training

**Environment Setup:**
- Platform: Google Colab with Tesla T4 GPU
- Base Model: Pre-trained `yolov8n-seg.pt`
- Training Duration: 100 epochs

**Configuration:**
- Data split: 80% training / 20% validation
- Created `dataset.yaml` configuration file
- Monitored key performance metrics during training

### 4. Inference & Evaluation

After training completion:
- Generated predictions on separate unannotated test set
- Evaluated model performance using standard computer vision metrics
- Created confusion matrix and Precision-Recall curves for analysis

### 5. Model-Assisted Review Loop

**Quality Assurance Workflow:**
1. Created second "Test Project" in Labellerr
2. Uploaded unannotated test images
3. Used Labellerr SDK to upload model predictions as pre-annotations
4. Verified predictions in Labellerr UI
5. Completed end-to-end quality assurance workflow

### 6. Video Tracking Implementation

**Final Application Development:**
- **Integration:** Combined fine-tuned YOLO-Seg model with ByteTrack algorithm
- **Functionality:** Real-time tracking of vehicles, pedestrians, and bikes in video
- **Output:** Generated `results.json` file with detailed tracking data including:
  - Object class
  - Bounding box coordinates
  - Unique tracker ID
  - Frame number

---

## Problems Faced & Resolutions

### Problem 1: Critical Training Failure - "No Labels Found" Error

**Issue:**
- YOLOv8 training script failed to match annotation data with image files
- Found all 117 images but reported zero labels
- Model could not learn due to missing label association

**Root Cause:**
- File name mismatch between exported COCO JSON and actual filenames
- Case sensitivity issues (e.g., `image.JPG` vs `image.jpg`)

**Resolution:**
Created a Python script to synchronize filenames by reading the COCO JSON file, matching case-insensitive filenames, updating JSON with correct paths, and saving the corrected JSON file. This systematic approach resolved the data pipeline issue and enabled successful training.

### Problem 2: Initial Fine-Tuning Performance Issues

**Issue:**
- Despite resolving filename mismatch, model performance remained poor
- Training metrics showed unexpected patterns
- Object detection accuracy was below expectations

**Resolution:**
- **Systematic Debugging:** Re-exported data from Labellerr
- **Path Verification:** Meticulously checked all paths in `dataset.yaml`
- **Pipeline Validation:** Verified each component individually
- **Configuration Fix:** Identified and corrected minor path configuration error

**Key Learning:** Success in machine learning often results from rigorous process validation and careful debugging rather than just model architecture improvements.

---

## Model Results & Evaluation

### Performance Metrics

After fine-tuning for 100 epochs, the model achieved the following performance on the validation set:

| Metric | Score |
|--------|-------|
| **mAP50-95 (Box)** | 0.3496 
| **mAP50-95 (Mask)** | 0.2976 |

### Visualizations

**Confusion Matrix:**
**Precision-Recall Curve:**


## Model Improvements

To enhance the model's performance, especially after identifying weaknesses during evaluation, several strategies can be implemented:

### 1. Data Augmentation
- **Diverse Scenarios:** Collect additional data focusing on challenging conditions
- **Specific Weaknesses:** Target scenarios where model performs poorly
- **Examples:** Heavy occlusion, poor lighting, unusual angles

### 2. Hyperparameter Tuning
- **Learning Rate Optimization:** Experiment with different learning schedules
- **Batch Size Adjustment:** Find optimal batch size for hardware constraints
- **Optimizer Settings:** Test various optimizer configurations

### 3. Model Architecture Improvements
- **Larger Backbone:** Upgrade to `yolov8m-seg.pt` or `yolov8l-seg.pt`
- **Higher Capacity:** Trade inference speed for improved accuracy
- **Custom Modifications:** Implement domain-specific architectural changes

### 4. Advanced Training Techniques
- **Transfer Learning:** Fine-tune from domain-specific pre-trained models
- **Data Augmentation:** Implement advanced augmentation strategies
- **Ensemble Methods:** Combine multiple models for improved performance



* **Demo Link:** https://youtu.be/MMI4yYUkKq4

This is the 2nd link to a large yolo model, please SEE --> https://youtu.be/UuAagLCqnOQ 








