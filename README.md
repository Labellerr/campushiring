# End-to-End Image Segmentation & Object Tracking Pipeline

**Submitted by: Gurnoor Kaur**

This repository contains the completed technical assessment for the Labellerr Computer Vision Internship. It details an end-to-end machine learning workflow, covering the entire lifecycle from data acquisition and annotation to model training, evaluation, and deployment for a real-world object tracking application.

---
## 🚀 Project Overview

The goal of this project was to build a robust system capable of performing instance segmentation and multi-object tracking on videos, specifically focusing on **vehicles** and **pedestrians**. The pipeline simulates a real-world MLOps cycle, showcasing a deep understanding of modern computer vision tools and best practices.

**Tech Stack:**
* **Annotation Platform:** Labellerr
* **Segmentation Model:** YOLOv8-Seg (Ultralytics)
* **Tracking Algorithm:** ByteTrack
* **Environment:** Google Colab (GPU), Python

**Key Features:**
1.  **Data-Centric Approach:** Utilized the Labellerr platform for precise data annotation, which is foundational to the model's performance.
2.  **Advanced Deep Learning:** Fine-tuned a state-of-the-art YOLOv8-Seg model for accurate instance segmentation.
3.  **High-Performance Tracking:** Integrated the trained model with ByteTrack to maintain consistent object identities across video frames, even through occlusions.
4.  **Reproducible Workflow:** The entire process is encapsulated in a Google Colab notebook, ensuring full reproducibility.

---
## 🛠️ E2E Workflow & Methodology

The project was executed through a structured, multi-stage pipeline.

##### **1. Data Acquisition and Preparation**
A dataset of traffic images was sourced from Kaggle. A subset of 150 images was curated, with 100 designated for training/validation and 50 for a hold-out test set to ensure unbiased evaluation. This initial step involved cleaning the dataset to retain only relevant images.

##### **2. Data Annotation on Labellerr**
A project was created on the Labellerr platform, and the training images were uploaded. Using Labellerr's advanced annotation tools, precise **polygon masks** were drawn for all instances of the `Vehicle` and `Pedestrian` classes. After annotation was complete, the labels were exported in the COCO JSON format.

##### **3. Model Training (YOLOv8-Seg)**
A Python script was used to convert the exported COCO annotations into the YOLOv8-Seg label format required for training. The dataset was then split into an 80/20 train/validation ratio for robust model evaluation. A pre-trained `yolov8s-seg.pt` model was fine-tuned on this custom dataset for 100 epochs.

##### **4. Model Evaluation & Inference**
The model's performance was evaluated using standard metrics on the validation set. The final trained model was then used to run inference on the 50 unseen test images, generating high-quality segmentation masks.

##### **5. Tracking with ByteTrack**
The trained YOLOv8-Seg model was integrated as the primary detector for the ByteTrack algorithm. A video was created from the test images, and the pipeline processed it frame-by-frame. ByteTrack's association logic successfully tracked individual objects, assigning them consistent IDs. The final output is an annotated video and a `results.json` file.

---
## 📊 Results & Performance

The YOLOv8-Seg model demonstrated strong performance after 100 epochs of training. The key metrics on the validation set are:

* **Box Detection (`mAP50-95(B)`):** 0.733 (Indicates strong object localization)
* **Mask Segmentation (`mAP50-95(M)`):** 0.514 (Shows good performance in outlining the exact shape of objects)

The final tracking video showcases this performance in a dynamic, real-world application.

---
## 🐛 Debugging Journey: Problems Faced & Resolutions

This project involved significant debugging, demonstrating practical problem-solving skills:

1.  **Problem:** Invalid YOLO Class IDs during Training.
    * **Symptom:** The training script failed, reporting "corrupt image/label: negative class labels".
    * **Resolution:** Upon inspection, the COCO-to-YOLO conversion script was found to be incorrectly decrementing the class IDs. Since the Labellerr export was already 0-indexed, this created invalid `-1` labels. The fix was to remove the decrement operation, ensuring correct data formatting.

2.  **Problem:** `git push` Failures with Large Files.
    * **Symptom:** Pushing the project to GitHub failed with `HTTP 408` (timeout) and `curl 55` (connection aborted) errors.
    * **Resolution:** The root cause was the large size of the dataset images and videos (121+ MiB), which is unsuitable for standard Git. The solution was to implement **Git LFS (Large File Storage)**. This involved undoing the previous large commit, configuring LFS to track binary file types (`.jpg`, `.zip`, `.mp4`), and then creating a new, clean commit.

---
## ⚙️ How to Run the Project

1.  **Clone the Repository:**
    ```bash
    git clone [https://github.com/your-username/campushiring.git](https://github.com/your-username/campushiring.git)
    cd campushiring/gurnoor_kaur/
    ```
2.  **Open in Google Colab:**
    Upload the `Labellerr_Internship_Gurnoor_Kaur.ipynb` notebook to [Google Colab](https://colab.research.google.com/).
3.  **Set Runtime:**
    Ensure the runtime is set to use a GPU for hardware acceleration (Runtime > Change runtime type > T4 GPU).
4.  **Upload Data:**
    When prompted by the notebook, upload the training images (`train_images.zip`) and the exported annotations (`annotations.json`).
5.  **Execute All Cells:**
    Run all cells sequentially by clicking `Runtime > Run all`. The notebook will handle all dependencies, data processing, training, and tracking.
6.  **View Outputs:**
    The final outputs, including the tracked video (`tracked_video.mp4`) and the JSON results, will be available in the Colab file system for download.
