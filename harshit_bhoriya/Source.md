# Video Object Tracking Project

This project is a web application built with **Streamlit** that performs real-time object detection and tracking on uploaded videos. It leverages powerful deep learning models to identify and follow objects throughout the video, providing an annotated output for viewing and download.

---

## Technologies Used

- **Streamlit**: The framework used to create the interactive web interface. It allows for rapid development of data applications in Python.
- **Ultralytics YOLOv8**: The core computer vision model for object detection. It is highly efficient and accurate at identifying a wide range of objects.
- **ByteTrack**: The tracking algorithm integrated with YOLOv8. It is responsible for assigning and maintaining unique IDs to each detected object across video frames, ensuring consistent tracking.

---

## How it Works

1. **File Upload**: Users upload a video file via the Streamlit interface.
2. **Model Processing**: The application uses the `yolov8n.pt` model to analyze the video frame by frame.
3. **Object Tracking**: The **ByteTrack** algorithm processes the detected objects, assigning unique IDs and drawing bounding boxes around them.
4. **Display & Download**: The annotated video is displayed directly in the web browser, with an option to download the processed video file.

---

## Getting Started

To run this application locally, you will need to install the required Python libraries. You can use `pip` to install the dependencies:

```bash
pip install streamlit ultralytics
```
