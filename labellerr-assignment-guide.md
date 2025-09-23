# Labellerr AI Software Engineer Assignment - Complete Implementation Guide

## Assignment Overview

**Assignment Type:** Technical Assessment - Computer Vision  
**Duration:** 3 Days (72 hours)  
**Time Commitment:** 12-16 hours (flexible scheduling)  
**Difficulty Level:** Intermediate to Advanced  
**Theme:** End-to-End Image Segmentation & Object Tracking Pipeline  
**Tech Stack:** YOLO-Seg, ByteTrack, Labellerr Platform, Python, Google Colab  
**Format:** Individual Project  

## Core Requirements

### Dataset Requirements
- **Minimum training images:** 100 annotated images
- **Test set:** ≤50 images  
- **Total raw images:** 150-200 images
- **Focus classes:** Vehicles and pedestrians
- **Challenge level:** Difficult dataset where publicly available models fail

### Technical Components
1. **Data Collection:** Download raw images with proper licensing
2. **Annotation:** Use Labellerr for polygon mask annotation
3. **Training:** YOLO-Seg model for ~100 epochs
4. **Evaluation:** Model performance metrics (IoU/mAP, confusion matrix, PR curves)
5. **Tracking:** Integrate YOLO-Seg with ByteTrack for video tracking
6. **Web App:** Simple web app for video upload and tracking demo

### Labellerr Integration Requirements
- Create training project via SDK and annotate ≤100 images
- Export labels in YOLO-compatible format
- Create test project and upload model predictions via SDK
- Verify predictions appear in Labellerr UI

## Detailed Implementation Plan

### Phase 1: Data Collection (4-6 hours)

**Objectives:**
- Research and select challenging vehicle/pedestrian datasets
- Download 150-200 raw images with proper licensing
- Create comprehensive documentation of data sources

**Tasks:**
1. **Dataset Research:**
   - COCO dataset (vehicles: car, truck, bus, motorcycle; person class)
   - Open Images (car, truck, bus, person classes)
   - Custom web scraping with proper licenses
   - Synthetic data generation for edge cases

2. **Data Collection Process:**
   ```python
   # Example COCO dataset download
   !wget http://images.cocodataset.org/zips/val2017.zip
   !wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
   
   # Filter for vehicle and person classes
   vehicle_classes = ['car', 'motorcycle', 'airplane', 'bus', 'train', 'truck']
   person_class = ['person']
   ```

3. **Data Organization:**
   ```
   dataset/
   ├── raw_images/
   ├── train/
   │   └── images/
   ├── val/
   │   └── images/
   ├── test/
   │   └── images/
   └── sources.md
   ```

4. **Create sources.md:**
   - List all data sources with URLs
   - Include license information
   - Document any modifications or filtering applied

### Phase 2: Labellerr Setup (2-3 hours)

**Objectives:**
- Set up Labellerr workspace and SDK
- Create training project with proper configuration

**Tasks:**
1. **Account Setup:**
   - Create Labellerr account at https://www.labellerr.com
   - Obtain API credentials from workspace settings

2. **SDK Installation:**
   ```bash
   pip install https://github.com/tensormatics/SDKPython/releases/download/prod/labellerr_sdk-1.0.0.tar.gz
   ```

3. **Project Creation:**
   ```python
   from labellerr.client import LabellerrClient
   from labellerr.exceptions import LabellerrError
   
   # Initialize client
   api_key = "your-api-key"
   api_secret = "your-api-secret"
   client_id = "your-client-id"
   email = "your-email"
   
   client = LabellerrClient(api_key, api_secret)
   
   # Create project
   project_payload = {
       'client_id': client_id,
       'dataset_name': 'Vehicle_Pedestrian_Segmentation',
       'dataset_description': 'Vehicle and pedestrian instance segmentation dataset',
       'data_type': 'image',
       'created_by': email,
       'project_name': 'YOLO-Seg Training Project',
       'annotation_guide': [
           {
               "question_number": 1,
               "question": "Segment vehicles and pedestrians",
               "question_id": "vehicle-pedestrian-seg",
               "option_type": "polygon",
               "required": True,
               "options": [
                   {"option_name": "Vehicle"},
                   {"option_name": "Pedestrian"}
               ]
           }
       ],
       'rotation_config': {
           'annotation_rotation_count': 1,
           'review_rotation_count': 1,
           'client_review_rotation_count': 1
       },
       'autolabel': False,
       'folder_to_upload': 'path/to/train/images'
   }
   
   result = client.initiate_create_project(project_payload)
   ```

### Phase 3: Annotation (3-4 hours)

**Objectives:**
- Annotate 100+ training images with high-quality polygon masks
- Utilize Labellerr's advanced annotation tools

**Tasks:**
1. **Annotation Strategy:**
   - Use Segment Anything tool for faster annotation
   - Ensure consistent annotation quality across classes
   - Focus on challenging scenarios (partial occlusion, multiple objects)

2. **Annotation Guidelines:**
   - **Vehicles:** Include cars, trucks, buses, motorcycles
   - **Pedestrians:** Include fully visible and partially occluded persons
   - **Quality control:** Verify polygon accuracy and completeness

3. **Export Process:**
   ```python
   # Export annotations in YOLO format
   export_result = client.export_project(
       project_id=project_id,
       export_format='yolo_segmentation'
   )
   ```

### Phase 4: Model Training (2-3 hours)

**Objectives:**
- Train YOLOv8 segmentation model on annotated dataset
- Monitor training progress and optimize hyperparameters

**Tasks:**
1. **Environment Setup:**
   ```bash
   # Google Colab setup
   !pip install ultralytics
   !pip install roboflow
   ```

2. **Dataset Configuration (dataset.yaml):**
   ```yaml
   train: ./dataset/train/images
   val: ./dataset/val/images
   test: ./dataset/test/images
   
   nc: 2  # number of classes
   names: ['vehicle', 'pedestrian']
   ```

3. **Training Command:**
   ```bash
   yolo task=segment mode=train model=yolov8n-seg.pt \
        data=dataset.yaml epochs=100 imgsz=640 batch=16 \
        name=vehicle_pedestrian_seg exist_ok=True
   ```

4. **Training Monitoring:**
   - Track mAP (mean Average Precision)
   - Monitor loss curves
   - Save best model weights

### Phase 5: Model Evaluation (1-2 hours)

**Objectives:**
- Comprehensive evaluation of trained model
- Generate detailed performance metrics

**Tasks:**
1. **Inference on Test Set:**
   ```bash
   yolo task=segment mode=predict \
        model=./runs/segment/vehicle_pedestrian_seg/weights/best.pt \
        source=./dataset/test/images conf=0.25 save=true
   ```

2. **Metrics Generation:**
   ```bash
   yolo task=segment mode=val \
        model=./runs/segment/vehicle_pedestrian_seg/weights/best.pt \
        data=dataset.yaml
   ```

3. **Evaluation Report:**
   - Confusion matrix analysis
   - Precision-Recall curves
   - mAP scores for each class
   - Qualitative analysis of failure cases

### Phase 6: Labellerr Integration (1-2 hours)

**Objectives:**
- Create test project and upload model predictions
- Demonstrate end-to-end Labellerr workflow

**Tasks:**
1. **Test Project Creation:**
   ```python
   test_project_payload = {
       'client_id': client_id,
       'dataset_name': 'Vehicle_Pedestrian_Test',
       'dataset_description': 'Test dataset for model predictions',
       'data_type': 'image',
       'created_by': email,
       'project_name': 'YOLO-Seg Test Project',
       'folder_to_upload': 'path/to/test/images'
   }
   
   test_result = client.initiate_create_project(test_project_payload)
   ```

2. **Upload Predictions:**
   ```python
   # Convert YOLO predictions to Labellerr format
   # Upload via SDK as pre-annotations
   client.upload_predictions(
       project_id=test_project_id,
       predictions=formatted_predictions
   )
   ```

3. **Verification:**
   - Check predictions appear in Labellerr UI
   - Verify annotation quality and accuracy

### Phase 7: Video Tracking Implementation (2-3 hours)

**Objectives:**
- Integrate YOLOv8-Seg with ByteTrack for multi-object tracking
- Implement robust tracking pipeline

**Tasks:**
1. **ByteTrack Setup:**
   ```bash
   git clone https://github.com/ifzhang/ByteTrack.git
   cd ByteTrack
   pip install -r requirements.txt
   pip install -e .
   ```

2. **Integration Code:**
   ```python
   import cv2
   from ultralytics import YOLO
   from yolox.tracker.byte_tracker import BYTETracker
   
   # Load trained model
   model = YOLO('path/to/best.pt')
   
   # Initialize tracker
   tracker = BYTETracker(frame_rate=30, track_thresh=0.5)
   
   def process_video(video_path):
       cap = cv2.VideoCapture(video_path)
       tracking_results = []
       
       while cap.isOpened():
           ret, frame = cap.read()
           if not ret:
               break
           
           # YOLO inference
           results = model(frame)
           
           # Convert to tracker format
           detections = convert_yolo_to_tracker_format(results)
           
           # Update tracker
           tracks = tracker.update(detections)
           
           # Store results
           frame_results = extract_tracking_info(tracks, frame_idx)
           tracking_results.append(frame_results)
       
       return tracking_results
   ```

3. **JSON Export:**
   ```python
   def export_tracking_results(tracking_results, output_path):
       json_output = {
           "video_info": {"fps": 30, "total_frames": len(tracking_results)},
           "tracking_data": tracking_results
       }
       
       with open(output_path, 'w') as f:
           json.dump(json_output, f, indent=2)
   ```

### Phase 8: Web Application Development (3-4 hours)

**Objectives:**
- Create intuitive web interface for video tracking demo
- Implement real-time processing and visualization

**Tasks:**
1. **Framework Selection:**
   - Streamlit for rapid prototyping
   - Flask for more control over UI/UX

2. **Streamlit Implementation:**
   ```python
   import streamlit as st
   import tempfile
   import json
   
   def main():
       st.title("Vehicle & Pedestrian Tracking Demo")
       st.markdown("Upload a video to track vehicles and pedestrians")
       
       uploaded_file = st.file_uploader("Choose a video file", type=['mp4', 'avi', 'mov'])
       
       if uploaded_file is not None:
           # Save uploaded file
           with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
               tmp_file.write(uploaded_file.read())
               video_path = tmp_file.name
           
           # Process video
           with st.spinner('Processing video...'):
               tracking_results = process_video(video_path)
           
           # Display results
           st.success("Tracking completed!")
           
           # Show tracking statistics
           display_tracking_stats(tracking_results)
           
           # Download results
           json_results = json.dumps(tracking_results, indent=2)
           st.download_button(
               label="Download Tracking Results (JSON)",
               data=json_results,
               file_name="tracking_results.json",
               mime="application/json"
           )
   
   if __name__ == "__main__":
       main()
   ```

3. **Features to Implement:**
   - Video upload functionality
   - Real-time processing progress
   - Tracking visualization
   - Results export (JSON format)
   - Statistics dashboard

### Phase 9: Documentation and Submission (2-3 hours)

**Objectives:**
- Create comprehensive documentation
- Prepare professional submission

**Tasks:**
1. **README.md Structure:**
   ```markdown
   # Vehicle & Pedestrian Tracking with YOLO-Seg and ByteTrack
   
   ## Project Overview
   ## Installation Instructions
   ## Usage Guide
   ## Model Performance
   ## Technical Architecture
   ## Results and Analysis
   ## Future Improvements
   ## Acknowledgments
   ```

2. **Technical Report:**
   - Journey documentation
   - Problems encountered and solutions
   - Model performance analysis
   - Lessons learned
   - Recommendations for scale

3. **Submission Checklist:**
   - [ ] GitHub repository with complete codebase
   - [ ] Comprehensive README with setup instructions
   - [ ] Live demo link (deployed web app)
   - [ ] PDF technical report with evaluation metrics
   - [ ] Pull request to Labellerr/campushiring repository
   - [ ] Clean code with proper documentation
   - [ ] Model weights and configuration files

## Technical Specifications

### Model Configuration
- **Architecture:** YOLOv8n-seg or YOLOv8s-seg
- **Input size:** 640x640 pixels
- **Training epochs:** 100
- **Batch size:** 16
- **Classes:** ['vehicle', 'pedestrian']

### Tracking Configuration
- **Tracker:** ByteTrack
- **Confidence threshold:** 0.5
- **High confidence threshold:** 0.6
- **Low confidence threshold:** 0.1
- **Track buffer:** 30 frames

### Output Formats
- **Annotations:** YOLO segmentation format
- **Tracking results:** JSON with track IDs, classes, bounding boxes, frame numbers
- **Model metrics:** mAP, IoU, confusion matrix, PR curves

## Expected Challenges and Solutions

### 1. Data Quality Issues
**Challenge:** Finding challenging dataset where public models fail  
**Solution:** Mix challenging scenarios (occlusion, weather conditions, lighting variations) with synthetic data generation

### 2. Annotation Time Constraints
**Challenge:** Time-intensive polygon annotation process  
**Solution:** Utilize Labellerr's Segment Anything tool for faster, more accurate annotations

### 3. Model Performance Optimization
**Challenge:** Achieving good performance on difficult dataset  
**Solution:** Implement data augmentation, ensure proper train/val/test splits, tune hyperparameters systematically

### 4. Integration Complexity
**Challenge:** Integrating multiple tools (YOLO, ByteTrack, Labellerr)  
**Solution:** Follow official documentation, use provided examples, test components incrementally

### 5. Deployment Challenges
**Challenge:** Creating functional web demonstration  
**Solution:** Use proven frameworks like Streamlit or Flask for rapid prototyping and deployment

## Suggested Timeline

### Day 1: Foundation (6-8 hours)
- **Morning:** Data collection and organization (3-4 hours)
- **Afternoon:** Labellerr setup and initial annotation (3-4 hours)

### Day 2: Core Development (6-8 hours)
- **Morning:** Complete annotation and model training (4-5 hours)
- **Afternoon:** Model evaluation and Labellerr integration (2-3 hours)

### Day 3: Integration and Deployment (6-8 hours)
- **Morning:** Video tracking implementation (3-4 hours)
- **Afternoon:** Web app development and documentation (3-4 hours)

**Total Estimated Time:** 18-24 hours

## Success Metrics

### Technical Metrics
- **Model Performance:** mAP > 0.4 for segmentation task
- **Tracking Accuracy:** Successful object ID consistency across frames
- **Processing Speed:** Real-time or near real-time video processing

### Deliverable Quality
- **Code Quality:** Clean, well-documented, reproducible code
- **Documentation:** Comprehensive README and technical report
- **Demo Functionality:** Working web application with intuitive UI
- **Integration Success:** Seamless Labellerr workflow demonstration

## Additional Resources

### Documentation Links
- [Labellerr SDK Documentation](https://docs.labellerr.com/sdk/getting-started)
- [YOLOv8 Segmentation Guide](https://docs.ultralytics.com/tasks/segment/)
- [ByteTrack Implementation](https://github.com/ifzhang/ByteTrack)
- [COCO Dataset](https://cocodataset.org/#download)

### Code Examples
- [YOLOv8 Training Notebook](https://colab.research.google.com/github/roboflow-ai/notebooks/blob/main/notebooks/train-yolov8-instance-segmentation-on-custom-dataset.ipynb)
- [ByteTrack Tutorial](https://www.labellerr.com/blog/how-to-implement-bytetrack)

## Conclusion

This comprehensive guide provides a structured approach to completing the Labellerr AI Software Engineer assignment. By following the detailed phases and implementing the suggested solutions, you'll create a professional-quality end-to-end computer vision pipeline that demonstrates proficiency in modern AI/ML technologies and tools.

Remember to document your journey, challenges, and solutions throughout the process, as this documentation will be valuable for both your submission and future reference.

Good luck with your implementation!