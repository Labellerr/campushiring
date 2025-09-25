# End-to-End Vehicle and Pedestrian Segmentation & Tracking

This project is a submission for the Labellerr AI Software Engineer assignment. It demonstrates a complete MLOps pipeline for data annotation, model training, and real-time object tracking.

## Overview

This project builds an end-to-end computer vision system capable of:
1.  **Segmenting** vehicles and pedestrians in images using a fine-tuned YOLOv8-seg model.
2.  **Tracking** these objects in video streams using the ByteTrack algorithm.
3.  A **web-based interface** to demonstrate the tracking capabilities on user-uploaded videos.

## How to Run the Demo

1.  **Clone the repository:**
    ```bash
    git clone <your-repo-url>
    cd <your-repo-folder>/first_lastname
    ```
2.  **Create and activate a Python virtual environment:**
    ```bash
    # For Windows
    python -m venv venv
    venv\Scripts\activate
    ```
3.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
4.  **Run the Streamlit application:**
    ```bash
    streamlit run app.py
    ```
    A new tab will open in your browser with the application.

## Problems Faced

A key part of the assignment was to programmatically upload model predictions to a Labellerr test project using the SDK. This step required a `CLIENT_ID` credential. Unfortunately, this credential was not provided, and I was unable to obtain it before the submission deadline. Therefore, I had to omit this part of the project. All other components, including data annotation, model training, and the final video tracking demo, were completed as specified.