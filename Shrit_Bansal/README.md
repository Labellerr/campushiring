
# Live demo
https://campushiring-h9uqxszmobdjmcfk58hnd2.streamlit.app/

## Demo video (Google Drive)
https://drive.google.com/file/d/1UjMnmuFf0Ul2E0QPMIGhucEY9mllzKE3/view?usp=sharing

# End-to-End Vehicle & Pedestrian Segmentation and Tracking
This repository contains an end-to-end workflow for semantic or instance segmentation and multi-object tracking of vehicles and pedestrians.

## Project summary
- Collected diverse street images via the Unsplash API.  
- Annotated images in Labellerr using AI-assisted polygon masks for efficient labeling.  
- Fine-tuned a YOLO-Seg model on the labeled dataset.  
- Integrated ByteTrack for multi-object tracking across video frames.  
- Built a Streamlit app for video upload, tracking, and artifact downloads.

## Features 
- Real-time tracking with persistent identity assignment.  
- Exports a tracked video and a detailed JSON of tracking results.  
- Modular structure for training, evaluation, and deployment.

## Getting started
- Python 3.8 or newer.  
- Install dependencies:
```
pip install -r requirements.txt
```

## Data and labeling
- Curated 150–200 permissibly licensed images of street scenes and maintained a Sources.md with attributions.  
- Created a Labellerr project, defined classes for vehicles and pedestrians, and label with polygon masks using the AI-assist tool.  
- Exported labels in a YOLO-Seg compatible format and keep disjoint train, val, and test splits.

## Training and validation
- Train a lightweight YOLO-Seg model for fast iteration and log metrics and curves.  
- Validate on a held-out set and capture mAP, precision/recall, F1, PR curves, confusion matrix, and representative qualitative results.

## Tracking and JSON export
- Used YOLO tracking with ByteTrack to assign stable IDs across frames.  
- Generated a results.json capturing frame index, class, confidence, and bounding box per tracked object.

## App usage
- Launch the Streamlit app, upload a video, run tracking, and download the processed video and results.json.  

## Results
- mAP50: 0.5499  
- mAP50-95: 0.3438  
- Precision: 0.5612  
- Recall: 0.6165  
- F1-Score: 0.5875

## Labellerr review loop
- Created a separate test project, upload test images, attach model predictions as pre-annotations, and verify suggestions in the UI.


## PDF report
- Included a consolidated PDF covering the journey, how-to guide, model results, metrics, and a concise final summary.

