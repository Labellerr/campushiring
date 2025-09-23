### End-to-End Vehicle and Pedestrian Segmentation & Tracking

This project is a submission for the Labellerr AI Software Engineer Assignment. It demonstrates a complete MLOps pipeline for dataset creation, model training, evaluation, and real-time video tracking through a web-based application.

# Overview

This project builds an end-to-end computer vision system capable of:

Segmentation: Detecting and segmenting vehicles and pedestrians in images using a fine-tuned YOLOv8n-seg model.

Tracking: Assigning consistent IDs across frames using the ByteTrack multi-object tracking algorithm.

Deployment: Providing a Streamlit web application for users to upload and process videos, visualize results, and download annotated outputs.

# Project Structure
├── data/                # Annotated dataset (YOLO format)  
├── notebooks/           # Jupyter notebooks for experimentation  
├── models/              # Trained YOLOv8 weights (best.pt)  
├── streamlit_app.py     # Streamlit frontend + Backend logic with YOLO + ByteTrack  
├── requirements.txt     # Python dependencies  
├── requirements_streamlit.txt     # Python dependencies  
├── README.md            # Project documentation (this file)  
└── report.pdf           # Project report (PDF/DOCX)

# How to Run the Demo
1. Clone the repository
git clone <your-fork-url>
cd <repo-folder>

2. Create and activate a virtual environment

For Windows:

python -m venv venv
venv\Scripts\activate


For Linux/Mac:

python3 -m venv venv
source venv/bin/activate

3. Install dependencies
pip install -r requirements.txt

4. Run the Streamlit application
streamlit run streamlit_app.py


A browser tab will open showing the web interface. Upload a video file to test segmentation and tracking. The app will:

Display progress while processing.

Show sample annotated frames.

Allow downloading of both the processed video and the JSON output (tracking data).

# Model Details

Base Model: YOLOv11n-seg (Ultralytics)

Training Data: Custom dataset with polygon annotations for Vehicle and Pedestrian classes

Training Environment: Google Colab T4 GPU

Epochs: 100

Metrics:

Box mAP50-95: 0.48

Box mAP50: 0.76

Mask mAP50-95: 0.44

Mask mAP50: 0.72

# Problems Faced

Overlapping Object Confusion: In crowded scenes, segmentation masks often merged. Resolved by tuning anchors and diversifying training samples.

Tracking ID Switching: Same pedestrian occasionally received multiple IDs. Fixed by adjusting ByteTrack parameters (track buffer).

Web App Lag: Large video files caused Streamlit to freeze. Solved by compressing videos and adding progress indicators.

# Future Improvements

Add support for multi-class datasets (e.g., bicycles, buses).

Deploy the app to cloud platforms like AWS/GCP for large-scale use.

Optimize pipeline with ONNX/TensorRT for real-time inference.

Integrate Labellerr SDK fully once CLIENT_ID credentials are available.

# Author

Priyanshu Ranjan
AI/ML Enthusiast 