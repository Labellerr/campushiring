# train_yolov8_seg.py
import argparse
from ultralytics import YOLO

def main(data_yaml, epochs=100, imgsz=640, batch=8, model="yolov8n-seg.pt", name="yolov8_seg_run"):
    model = YOLO(model)
    model.train(data=data_yaml, epochs=epochs, imgsz=imgsz, batch=batch, name=name)
    print("Training completed. Weights in runs/seg/{}/weights/".format(name))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data_seg.yaml")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--model", default="yolov8n-seg.pt")
    parser.add_argument("--name", default="yolov8_seg_run")
    args = parser.parse_args()
    main(args.data, args.epochs, args.imgsz, args.batch, args.model, args.name)
