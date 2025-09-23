YOLOv8 Vehicle and Pedestrian Tracking App
This is a web application built with Streamlit that uses a custom-trained YOLOv8 model to perform real-time object detection and tracking on uploaded videos.

Features
Upload video files (.mp4, .mov, .avi).

Detects and tracks 'pedestrian' and 'vehicle' objects.

Adjustable confidence threshold to fine-tune detections.

Displays the processed video with bounding boxes and tracking IDs.

Allows downloading the final tracked video.

Project Structure
.
├── weights/
│   └── best.pt       (Note: This large file is excluded by .gitignore)
├── .gitignore
├── app.py
└── requirements.txt

Setup and Usage
Follow these steps to run the application locally.

1. Clone the Repository
git clone [https://github.com/YOUR_USERNAME/YOUR_REPOSITORY_NAME.git](https://github.com/YOUR_USERNAME/YOUR_REPOSITORY_NAME.git)
cd YOUR_REPOSITORY_NAME

2. Install Dependencies
Create a virtual environment (recommended) and install the required libraries.

pip install -r requirements.txt

3. Add the Model File
This repository does not include the trained model file (best.pt) due to its size. You must download the best.pt file manually and place it inside the weights/ directory.

4. Run the Streamlit App
Once the dependencies are installed and the model is in place, run the application from your terminal:

streamlit run app.py

The application will open in your web browser.