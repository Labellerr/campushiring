# Vehicle and Pedestrian Tracking using YOLOv8 Segmentation

This repository hosts the code, datasets, and resources for a comprehensive traffic scene object segmentation and tracking project. Using state-of-the-art deep learning techniques with Ultralytics YOLOv8, this project offers an end-to-end pipeline starting from raw video data collection to deployment of a real-time web application for multi-class object tracking.

## Project Overview

The project aims to develop an automated system capable of detecting, segmenting, and tracking vehicles, bikes, and pedestrians in traffic videos. The core components include:

- **Data Acquisition:** Collection of diverse traffic videos from free sources, including Pexels and YouTube, ensuring coverage of various traffic environments and scenarios. 
- **Dataset Curation:** Conversion of video data into images through frame extraction, followed by duplicate frame removal to enhance dataset quality and reduce redundancy. Manual annotation was conducted using the Labeller platform to label objects in three categories: vehicles, bikes, and pedestrians.
- **Model Development:** Fine-tuning of the YOLOv8n segmentation model on the curated dataset within the Google Colab environment, leveraging GPU acceleration for efficient model training. Custom hyperparameters and data splits were used to maximize detection accuracy.
- **Performance Evaluation:** Comprehensive evaluation using key metrics such as precision, recall, and mean average precision (mAP) across classes to ensure robustness and generalization.
- **Web Application:** Development of an intuitive web app powered by Streamlit that enables users to upload videos for real-time object tracking. The app provides annotated videos with overlay segmentation masks and outputs a JSON file containing detailed tracking information.

## Datset and Demo Video 

All the images used to train and test the model as well as the video can be found here:

[Dataset and Demo Video · Drive](https://drive.google.com/drive/folders/1h7qCufIjyN6CgaXMBJbC2nqjeYB6EieY?usp=sharing)

## Live Demo

Experience the live demo of the traffic object tracking application here:

[Vehicle and Pedestrian Tracking · Streamlit](https://trafficlivetracking.streamlit.app/)

## Features

- Real-time processing of traffic videos
- Instance segmentation of vehicles, bikes, and pedestrians
- Object tracking across frames
- Exportable JSON summary of detected objects with spatial-temporal data
- User-friendly web interface with drag-and-drop video upload

# Repository Structure

- `metrics/` - Contains training  datasets and model metrics.
- `assets/` - Pretrained and fine-tuned model weights.
- `notebooks/` - Jupyter notebooks for training and evaluation.
- `streamlit_app.py` - Web app source code.

## Credits

- Videos sourced from [Pexels](https://www.pexels.com/videos/) and YouTube.
- Labels created using Labeller platform.
- Model trained using Ultralytics YOLOv8 framework.

---

For any issues or contributions, please open an issue or pull request on this repo.

To clone and run the project locally:

