# End-to-End Vehicle & Pedestrian Tracking System

**Submission for the Labellerr AI Software Engineer Internship Assignment**

This repository contains the complete project for the technical assessment, featuring an end-to-end image segmentation and object tracking pipeline. The system is built using YOLOv8-Seg for segmentation and ByteTrack for multi-object tracking, with a live demonstration deployed as a Streamlit web application.

---

## 🚀 Live Demo

You can access and interact with the live video tracking application here:

**[https://vehiclehumantracker1234.streamlit.app/](https://vehiclehumantracker1234.streamlit.app/)**

**Demo Walkthrough:**

Here's a look at the application in action:

1. **Welcome to the App:** The Streamlit interface, ready for video uploads and configuration.
   
   <img width="1884" height="835" alt="image" src="https://github.com/user-attachments/assets/041a45a8-aa88-454a-9995-4a11989d48c2" />

2. **Original Video Input:** A sample video uploaded before processing.
   
   <img width="673" height="383" alt="image" src="https://github.com/user-attachments/assets/a2b2b4b1-1d9a-4812-87ce-274cdfc76088" />

3. **Processed Video with Tracking:** The same video, now with bounding boxes, segmentation masks, and unique tracking IDs for vehicles and pedestrians.
   
   <img width="1369" height="611" alt="image" src="https://github.com/user-attachments/assets/68dcfc0c-cd4f-4b23-adf4-19bddb6f4944" />

4. **Results & Analytics:** The "Detection Table" showing detailed tracking data, including object class, confidence, bounding box coordinates, and timestamps.
   
   <img width="1201" height="611" alt="image" src="https://github.com/user-attachments/assets/7c7217e7-7945-48da-9bde-a8735394c4cb" />

---

## 📋 Project Overview

This project simulates a real-world computer vision development lifecycle. It covers everything from data collection and annotation to model training, evaluation, and finally, the deployment of a live tracking application.

### Features
- **Data Annotation:** A custom dataset of 100+ images was annotated with polygon masks for `vehicle` and `pedestrian` classes using the Labellerr platform.
- **Model Training:** A YOLOv8-Seg model was fine-tuned on the custom dataset to perform instance segmentation.
- **Video Object Tracking:** The trained YOLO model is integrated with the ByteTrack algorithm to track detected objects across video frames, assigning a unique ID to each object.
- **Interactive Web Demo:** A user-friendly web application built with Streamlit that allows users to upload a video and receive a processed video with tracking annotations and an exportable JSON file of the results.

---

## 🛠️ Tech Stack

- **Model:** YOLOv8-Seg (Ultralytics)
- **Tracker:** ByteTrack
- **Data Annotation:** Labellerr Platform & SDK
- **Web Framework:** Streamlit
- **Core Libraries:** Python, OpenCV, PyTorch, NumPy, Pandas

---

## ⚙️ Setup and Local Installation

To run this project on your local machine, follow these steps:

1. **Clone the repository:**
   ```bash
   git clone https://github.com/[Your-Username]/campushiring.git
   cd campushiring/[your_first_lastname]
   ```
   *(Replace `[Your-Username]` and `[your_first_lastname]` with your actual GitHub username and project directory name.)*

2. **Install dependencies:**
   It is recommended to use a virtual environment.
   ```bash
   # Create and activate a virtual environment (optional but recommended)
   python -m venv venv
   source venv/bin/activate 

   # Install all required packages
   pip install -r requirements.txt
   ```

3. **Run the Streamlit App:**
   ```bash
   streamlit run video_tracker_app.py
   ```
   The application will now be running on your localhost.

---

## 📁 Project Deliverables

As per the assignment guidelines, all deliverables are included in this repository:

1. **Live Demo Link:** [https://vehiclehumantracker1234.streamlit.app/](https://vehiclehumantracker1234.streamlit.app/)
2. **Final Report:** A detailed PDF document covering the project journey, challenges faced, model evaluation metrics, and a summary of the process can be found here: **[Final_Report.pdf](./Final_Report.pdf)**
3. **Dataset Sources:** A complete list of data sources and their licenses is available in the **[sources.md](./sources.md)** file.

---

## 🧠 Challenges and Learnings

Throughout this assignment, several challenges were encountered and resolved, demonstrating key debugging and problem-solving abilities:

- **Deployment Dependencies:** A significant challenge was deploying the Streamlit app, which required resolving not only Python package dependencies (`requirements.txt`) but also underlying system-level library issues (e.g., `libGL.so.1`) using a `packages.txt` file on Streamlit Cloud.
- **Video Processing Optimization:** The initial video processing logic was slow. It was refactored to use a more efficient codec (`XVID` in `.avi`) for intermediate file generation, which was then transcoded to a web-compatible MP4 using `ffmpeg`, improving reliability and performance.
- **Model-Assisted Labeling Flow:** Setting up the complete feedback loop—from training a model to uploading its predictions back to a Labellerr test project via the SDK—was a great learning experience in building a scalable data pipeline.

This assignment was an excellent, hands-on opportunity to experience the full machine learning lifecycle from data to deployment.
