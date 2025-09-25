Vehicle and Pedestrian Segmentation & Tracking System

A complete end-to-end computer vision pipeline designed to detect, segment, and track vehicles and pedestrians from videos — accessible via an interactive Streamlit web application.

This project was developed as part of the Labellerr AI Software Engineer Assignment, showcasing practical MLOps workflows from dataset annotation to model deployment.

# Features

Instance Segmentation: Utilizes a fine-tuned YOLOv11n-seg model to precisely segment vehicles and pedestrians.

Multi-Object Tracking: Maintains consistent object IDs across frames using a custom distance-based tracking algorithm.

Interactive Web Interface: Users can upload videos, monitor processing progress, view annotated outputs, and download both processed videos and tracking metadata (JSON).

Live Metrics: Real-time counts of vehicles and pedestrians displayed during processing for quick insights.

# Repository Structure
data/                 # Custom annotated datasets (YOLO format)
notebooks/            # Experimental Jupyter notebooks
models/               # Trained YOLOv11n-seg weights
streamlit_app.py      # Main Streamlit app with detection & tracking logic
requirements.txt      # Python package dependencies
README.md             # Project documentation
report.pdf            # Comprehensive project report

Installation & Usage

Clone the repository:

git clone <your-fork-url>
cd <repo-folder>


Set up a virtual environment and activate it:

Windows

python -m venv venv
venv\Scripts\activate


Linux / MacOS

python3 -m venv venv
source venv/bin/activate


Install dependencies:

pip install -r requirements.txt


Launch the Streamlit app:

streamlit run streamlit_app.py


Upload supported video formats (.mp4, .avi, .mov).

View annotated video outputs and download JSON tracking logs.

Model Information

Architecture: YOLOv11n-seg by Ultralytics, fine-tuned on a custom vehicle & pedestrian dataset.

Training: 100 epochs on Google Colab using a T4 GPU.

Performance Metrics:

Box mAP50-95: 0.48

Box mAP50: 0.76

Mask mAP50-95: 0.44

Mask mAP50: 0.72

 #  Challenges Faced
Problem	Approach to Resolve
Overlapping masks in crowded scenes	Improved anchor tuning and training data diversity
ID switching during tracking	Adjusted tracking parameters (distance threshold, max buffer)
UI lag on large video files	Added video compression and progress bar indicators
Roadmap & Future Enhancements

Extend detection to more classes (e.g., bicycles, buses).

Deploy application on cloud services like AWS or GCP.

Optimize inference speed with ONNX or TensorRT.

Full integration with Labellerr SDK upon receiving client credentials.

Author

SURAJ MITTAL
AI/ML Practitioner | Computer Vision Engineer