# YOLOv8 Object Detection & Segmentation



# Project Overview
This project uses YOLOv8 for object detection and segmentation.  
The model was trained to detect Cars and Persons in images.  

The workflow includes:
- Preparing a custom dataset
- Annotation using Labellerr 
- Training YOLOv8 with segmentation  
- Running inference on new test images  
- Visualizing predictions
- bytetracking using model



# Dataset
- Custom dataset with two classes:  
  - car  
  - person



# Training 
from ultralytics import YOLO

model = YOLO("yolov8n-seg.pt")

model.train(
    data="dataset/data.yaml",
    epochs=100,
    imgsz=640,
    batch=16
)


# Prediction

-- for test images
from ultralytics import YOLO

model = YOLO("runs/segment/train/weights/best.pt")
results = model.predict("test_images/example.jpg")
results[0].show()

-- folder of images

results = model.predict("test_images/*")
for r in results:
    r.save()

-- output is saved in runs/predict/



