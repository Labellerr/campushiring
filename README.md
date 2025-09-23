# End-to-End Vehicle & Pedestrian Tracking System

**Submission for the Labellerr AI Software Engineer Internship Assignment by [Your Name]**

[cite_start]This repository contains the complete project for the technical assessment, featuring an end-to-end image segmentation and object tracking pipeline[cite: 8]. [cite_start]The system is built using YOLOv8-Seg for segmentation and ByteTrack for multi-object tracking, with a live demonstration deployed as a Streamlit web application[cite: 9, 61].

---

## 🚀 Live Demo

You can access and interact with the live video tracking application here:

**[https://your-app-url.streamlit.app/](https://your-app-url.streamlit.app/)**

![Demo GIF](https://i.imgur.com/your-demo.gif)
*(Tip: Create a short screen recording of your app and convert it to a GIF to showcase it here.)*

---

## 📋 Project Overview

[cite_start]This project simulates a real-world computer vision development lifecycle[cite: 11]. [cite_start]It covers everything from data collection and annotation to model training, evaluation, and finally, the deployment of a live tracking application[cite: 12].

### Features
-   [cite_start]**Data Annotation:** A custom dataset of 100+ images was annotated with polygon masks for `vehicle` and `pedestrian` classes using the Labellerr platform[cite: 39, 40].
-   [cite_start]**Model Training:** A YOLOv8-Seg model was fine-tuned on the custom dataset to perform instance segmentation[cite: 44, 53].
-   [cite_start]**Video Object Tracking:** The trained YOLO model is integrated with the ByteTrack algorithm to track detected objects across video frames, assigning a unique ID to each object[cite: 27, 47].
-   [cite_start]**Interactive Web Demo:** A user-friendly web application built with Streamlit that allows users to upload a video and receive a processed video with tracking annotations and an exportable JSON file of the results[cite: 62, 66].

---

## 🛠️ Tech Stack

-   [cite_start]**Model:** YOLOv8-Seg (Ultralytics) [cite: 45]
-   [cite_start]**Tracker:** ByteTrack [cite: 9]
-   [cite_start]**Data Annotation:** Labellerr Platform & SDK [cite: 9, 40]
-   **Web Framework:** Streamlit
-   **Core Libraries:** Python, OpenCV, PyTorch, NumPy, Pandas

---

## ⚙️ Setup and Local Installation

To run this project on your local machine, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/](https://github.com/)[Your-Username]/campushiring.git
    cd campushiring/[your_first_lastname]
    ```

2.  **Install dependencies:**
    It is recommended to use a virtual environment.
    ```bash
    # Create and activate a virtual environment (optional but recommended)
    python -m venv venv
    source venv/bin/activate 

    # Install all required packages
    pip install -r requirements.txt
    ```

3.  **Run the Streamlit App:**
    ```bash
    streamlit run video_tracker_app.py
    ```
    The application will now be running on your localhost.

---

## 📁 Project Deliverables

As per the assignment guidelines, all deliverables are included in this repository:

1.  [cite_start]**Live Demo Link:** [https://vehiclehumantracker.streamlit.app/](https://vehiclehumantracker.streamlit.app/) [cite: 72]
2.  [cite_start]**Final Report:** A detailed PDF document covering the project journey, challenges faced, model evaluation metrics, and a summary of the process can be found here: **[Final_Report.pdf](./Final_Report.pdf)**[cite: 76].
3.  [cite_start]**Dataset Sources:** A complete list of data sources and their licenses is available in the **[sources.md](./sources.md)** file[cite: 36].

---

## 🧠 Challenges and Learnings

[cite_start]Throughout this assignment, several challenges were encountered and resolved, demonstrating key debugging and problem-solving abilities[cite: 22]:

-   **Deployment Dependencies:** A significant challenge was deploying the Streamlit app, which required resolving not only Python package dependencies (`requirements.txt`) but also underlying system-level library issues (e.g., `libGL.so.1`) using a `packages.txt` file on Streamlit Cloud.
-   **Video Processing Optimization:** The initial video processing logic was slow. It was refactored to use a more efficient codec (`XVID` in `.avi`) for intermediate file generation, which was then transcoded to a web-compatible MP4 using `ffmpeg`, improving reliability and performance.
-   [cite_start]**Model-Assisted Labeling Flow:** Setting up the complete feedback loop—from training a model to uploading its predictions back to a Labellerr test project via the SDK—was a great learning experience in building a scalable data pipeline[cite: 41, 59].

This assignment was an excellent, hands-on opportunity to experience the full machine learning lifecycle from data to deployment.