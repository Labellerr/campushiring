from ultralytics import YOLO

# Load trained YOLO-Seg model
model = YOLO("runs/segment/train/weights/best.pt")  # Use trained weights

# Run inference on test set
results = model.predict(
    source="dataset/test/images",  # Path to test images
    save=True
)

# Save predictions
for r in results:
    r.save_txt()  # Save predictions in YOLO format
