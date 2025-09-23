# End-to-End Image Segmentation & Object Tracking Pipeline (YOLO-Seg & ByteTrack)

This project implements an end-to-end computer vision pipeline for image segmentation and object tracking, focusing on vehicles and pedestrians. It utilizes YOLO-Seg for segmentation and is designed to be integrated with the Labellerr platform for data management and model-assisted review. The project was developed as a technical assessment.

## Assignment Details

*   **Assignment Type:** Technical Assessment - Computer Vision
*   **Theme:** End-to-End Image Segmentation & Object Tracking Pipeline
*   **Tech Stack:** YOLO-Seg, ByteTrack, Labellerr Platform, Python, Flask, OpenCV, Ultralytics
*   **Objective:** Build a small end-to-end semantic/instance segmentation workflow for vehicles and pedestrians using YOLO-Seg, integrated with Labellerr, and potentially integrate with ByteTrack for tracking.

## Project Structure

The project includes the following key components:

1.  **Data Understanding and Preparation:** Processing the provided dataset from `Labeller_Test_zip.zip`.
2.  **Model Training:** Training a YOLOv8-seg model on the prepared dataset.
3.  **Web Application:** A Flask-based web application for uploading videos, processing them with the trained model, and downloading the results.
4.  **Potential Object Tracking:** Integration with ByteTrack (attempted, but noted potential import issues in development).

## Setup and Installation

1.  Clone this repository (or save the code files).
2.  Ensure you have Python installed (3.6+ recommended).
3.  Install the required dependencies. You can use the `requirements.txt` file generated:
