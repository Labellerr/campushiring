from ultralytics import YOLO

# Load a pre-trained YOLOv8 segmentation model
model = YOLO('yolov8n-seg.pt')

# Train the model using the dataset.yaml configuration file
results = model.train(
    data='dataset.yaml',
    epochs=100,
    imgsz=640,
    project='YOLOv8-Training',
    name='fine_tuning_run'
)

print("\n✅ Fine-tuning complete!")
print("Your trained model is saved in the 'YOLOv8-Training/fine_tuning_run/weights/' directory.")
print("The best performing model is named 'best.pt'.")