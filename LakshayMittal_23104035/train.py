from ultralytics import YOLO

# Load YOLOv8 detection model
model = YOLO("yolov8n.pt")  # Use detection model instead of segmentation model

# Train
model.train(
    data="Vehicles-OpenImages.v1-416x416.yolov8/data.yaml",  # Correct dataset path
    epochs=20,  # Reduced to 20 epochs
    imgsz=416,  # Reduced image size
    batch=4,  # Reduced batch size
    mosaic=0.0  # Disable mosaic augmentation
)
