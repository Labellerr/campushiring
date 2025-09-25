# End-to-End Vehicle and Pedestrian Tracker

**Live Demo Link:** https://live-traffic-tracker.streamlit.app
**Backup Demo Video:** [View on Google Drive](https://drive.google.com/file/d/1qs6GU5qztQhxtQTM-ETqRbGUZ6B05xSh/view?usp=drive_link) *(Note: A video recording of the app is provided in case the free cloud deployment is sleeping or temporarily unavailable.)*

---

## Project Overview

This project is an end-to-end computer vision pipeline for segmenting and tracking vehicles and pedestrians in video footage. It was built as a technical assessment for the Labellerr AI internship. The entire machine learning lifecycle was covered, from manual data annotation on the Labellerr platform to custom model training and final deployment as an interactive web application.

## Features

-   Real-time object tracking for `vehicle` and `pedestrian` classes.
-   Trained on a custom-annotated dataset using the Labellerr platform.
-   Ability to upload a video and view live tracking results.
-   Exports detailed tracking data (frame number, object ID, class, and coordinates) to a `results.json` file.

## Tech Stack

-   **Model:** YOLOv8-seg
-   **Tracker:** ByteTrack
-   **Annotation Platform:** Labellerr
-   **Web App:** Streamlit
-   **Core Language:** Python

## Local Setup and Usage

To run this project on your local machine, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/Sid-singh415/campushiring.git](https://github.com/Sid-singh415/campushiring.git)
    cd campushiring/Sidakpreet_Singh
    ```

2.  **Install dependencies from the requirements file:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the Streamlit app:**
    ```bash
    streamlit run app.py
    ```

## Challenges and Learnings

During this project, I encountered several real-world challenges that provided significant learning opportunities:

-   **Challenge 1: Complex Data Collection & Curation:** As encouraged by the assignment, I focused on collecting a difficult and diverse dataset. This involved sourcing images with challenging conditions like nighttime scenes, rainy weather, and heavy object occlusion. Curating and then properly shuffling this data to ensure a balanced distribution across the train, validation, and test sets was a critical lesson in building a robust model.

-   **Challenge 2: Data Cleaning and Formatting:** Many of the sourced images were in unsupported formats (like `.HEIC` or `.WEBP`). This required a data pre-processing step to standardize all images into a compatible JPEG format before annotation could begin. This was a practical lesson in the importance of data cleaning in any serious ML pipeline.

-   **Challenge 3: Local Python Environment Setup:** When setting up the Streamlit application on my local machine, I ran into the `externally-managed-environment` error. This was a valuable learning experience in modern Python dependency management on macOS. I learned how to use virtual environments (`venv`) to create isolated project spaces and, when necessary, how to safely install packages directly for a user with override flags.

-   **Challenge 4: Adapting to Evolving Libraries:** While integrating the tracker, the code failed with `AttributeError`s related to the `supervision` library. After debugging, I discovered this was due to recent, breaking API changes in the library. This was a crucial lesson in the reality of working with fast-moving open-source tools and the importance of adapting code to the latest documentation.