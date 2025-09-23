# End-to-End Vehicle & Pedestrian Segmentation and Tracking Pipeline

This project is a submission for the **Labellerr AI Software Engineer Internship** assignment. It demonstrates a complete, end-to-end computer vision pipeline—from data collection and annotation to model training and deployment in a live web application.

The core of the project is a **custom-trained YOLOv8-seg model** for instance segmentation of vehicles and pedestrians, which is then integrated with the **ByteTrack algorithm** for robust, real-time object tracking in videos.

---

## 🚀 Live Demo

You can access the live video tracking application here:  
[[https://huggingface.co/spaces/am-om/tracker]]

---

## 🛠️ Tech Stack

- **Model:** YOLOv8-seg (Ultralytics)  
- **Tracking Algorithm:** ByteTrack  
- **Annotation Platform:** Labellerr Platform & SDK  
- **Core Libraries:** Python, PyTorch, Supervision, OpenCV  
- **Web Framework:** FastAPI, HuggingFace Spaces
- **UI:** Gradio
- **Development Environment:** Google Colab (GPU-accelerated training)  

---

## ⚙️ Setup and Installation

To run this project on your local machine, please follow these steps.

### 📂  Clone the Repository

First, clone this GitHub repository to your local machine:

```bash
git https://github.com/OmSingh5131/campushiring
cd campushiring/Om_Singh
```


### 💻 Create and Activate a Virtual Environment

It is highly recommended to use a Python virtual environment to manage dependencies and avoid conflicts.

```bash
python -m venv venv

# Activate the environment

#  On Windows:
venv\Scripts\activate

# On macOS/Linux:

source venv/bin/activate
```

### ⬇️ Install Required Libraries
Install all necessary Python packages using the provided requirements.txt file:

```bash

python -m pip install -r requirements.txt
```


## ▶️ How to Run

### 🌐 Video Tracking Web Application

The primary deliverable is the web application, which can be launched locally on your machine, allowing you to upload a video for processing. To run it, execute the following command in your terminal from the project's `app` directory:

```bash
python app.py
```

### ✏️ Labellerr Pre-annotation Upload

The script upload_predictions.py is used to upload model predictions to the Labellerr platform for the review loop.

#### Execution:
```bash
python upload_predictions.py
```

#### Setup:

- **Create** a `.env` file in the project root.  
- **Add** your Labellerr credentials to the `.env` file:  
  ```env
  API_KEY="your_api_key"
  API_SECRET="your_api_secret"
  CLIENT_ID="your_client_id"
  TEST_PROJECT_ID="your_test_project_id"
  ```
  

### 📊 Summary of Results

The custom **YOLOv8n-seg** model was trained for **100 epochs** on a challenging, hand-annotated dataset of ~150 images, containing a mix of real-world driving scenes and high-quality stock photos.

> **Note:** The assignment explicitly encouraged using a difficult dataset where standard models might fail. Achieving a strong baseline on such data is a key success metric for this project.

**Final Model Performance on Validation Set:**

| Metric                  | Score |
|-------------------------|-------|
| Overall Mask mAP50      | 0.591 |
| Vehicle Mask mAP50      | 0.68  |
| Pedestrian Mask mAP50   | 0.503 |

These results are considered strong for a small, intentionally difficult dataset, demonstrating the model's ability to generalize well.


