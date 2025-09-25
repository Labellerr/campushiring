from ultralytics import YOLO

if __name__ == "__main__":
    # Loading YOLOv11 segmentation model
    model = YOLO("yolo11n-seg.pt")  

    # Training the model
    model.train(
        data="data.yaml",      
        epochs=100,             
        imgsz=640,             
        batch=8,               
        project="runs/train",  
        name="my_yolo_model",  
        exist_ok=True         
    )
