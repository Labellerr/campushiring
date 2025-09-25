import os
import shutil
from ultralytics import YOLO

def main():
    data_yaml = os.path.join(os.path.dirname(__file__), "configs", "data.yaml")
    model = YOLO("yolov8s-seg.pt")

    model.train(data=data_yaml, epochs=100, imgsz=640, batch=8)

    src_weights = os.path.join(os.path.dirname(os.path.dirname(__file__)), "runs", "segment", "train2", "weights", "best.pt")
    dest_weights = os.path.join(os.path.dirname(os.path.dirname(__file__)), "Labeller_Assignment_Dataset", "weights", "best.pt")

    shutil.copy(src_weights, dest_weights)
    print(f"Model weights copied to {dest_weights}")


if __name__ == "__main__":
    main()
