# End-to-End Image Segmentation & Object Tracking Pipeline

This project is a technical assessment for the **Labellerr AI Internship**, demonstrating a complete machine learning lifecycle from data annotation to model training and object tracking. The goal is to build an end-to-end instance segmentation workflow for vehicles and pedestrians using **YOLO-Seg** and **ByteTrack**, with the Labellerr platform at the core of the data management process.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Tech Stack](#tech-stack)
- [Dataset Information](#dataset-information)
- [Workflow Pipeline](#workflow-pipeline)
- [Problems Faced & Resolutions](#problems-faced--resolutions)
- [Results & Evaluation](#results--evaluation)
- [Video Tracking Demo](#video-tracking-demo)
- [How to Run](#how-to-run)

---

## Project Overview

This project simulates a real-world computer vision development cycle by building a complete image segmentation and object tracking system from scratch. It covers the entire machine learning lifecycle, starting with data creation and manual annotation, followed by model training, evaluation, and a model-assisted review loop for quality assurance. The final output is an application that can track vehicles and pedestrians in a video.

---

## Tech Stack

The technologies used to build this project are:

- **Model:** YOLO-Seg (specifically, Ultralytics YOLOv8-seg)
- **Tracker:** ByteTrack
- **Platform:** Labellerr AI (for annotation and data management)
- **Language:** Python
- **Environment:** Google Colab with GPU resources

---

## Dataset Information

The assignment encouraged the creation of a custom, challenging dataset rather than using a pre-labeled one. The dataset for this project consists of **117 images for training** and a separate set of images for testing.

**Data Source:** The image data is a custom mix, containing some of my own photographs and some sourced from Pexels. All images are permissibly licensed for this project.

**Sources File:** A `sources.md` file is included in this repository with links to the original images and their license notes, as required by the assignment guidelines.

---

## Workflow Pipeline

The project followed a structured, end-to-end machine learning lifecycle:

1. **Data Collection:** Gathered over 150 raw images of road scenes to form the initial dataset.
2. **Project Setup on Labellerr:** Created a project on the Labellerr platform, either via the UI or SDK, for manual annotation.
3. **Manual Annotation:** Annotated a minimum of 100 images with polygon masks for the training set using the Labellerr interface. The Segment Anything feature was utilized to speed up this process.
4. **Data Export:** Exported the completed labels from Labellerr in a format suitable for YOLO training.
5. **Model Training:** Fine-tuned a yolov8n-seg model for approximately 100 epochs.
6. **Inference and Evaluation:** Ran the trained model on the unannotated test set to generate predictions. The model's performance was evaluated using metrics like a confusion matrix and PR curve.
7. **Model-Assisted Review Loop:** Created a second "Test Project" in Labellerr and uploaded the model's predictions via the SDK as pre-annotations for review. This demonstrates the complete annotation-to-review flow.
8. **Video Tracking:** Integrated the trained YOLO-Seg model with ByteTrack to track vehicles and pedestrians in a video. The final tracking data was exported to a `results.json` file.

---

## Problems Faced & Resolutions

The assignment requires identifying and fixing at least two real issues during the process.

**Problem:** [Enter the first problem you faced, e.g., "Training Failure Due to Data Loading Errors"]

**Issue:** [Describe the issue, e.g., "The YOLOv8 training script failed with a 'No labels found' error, despite the annotation files being present."]

**Resolution:** [Explain your solution, e.g., "I discovered a case-sensitivity mismatch between the 'file_name' in the COCO JSON and the actual image filenames. I wrote a Python script to programmatically synchronize the names in the JSON file, which resolved the data loading error."]

**Problem:** [Enter the second problem you faced, e.g., "Poor Model Performance on Occluded Objects"]

**Issue:** [Describe the issue, e.g., "The initial model performed poorly on pedestrians or vehicles that were partially hidden behind other objects."]

**Resolution:** [Explain your solution, e.g., "I identified that my initial dataset lacked sufficient examples of occlusion. I collected and annotated additional images featuring partially hidden objects and added them to the training set. This data augmentation significantly improved the model's performance on these difficult cases."]