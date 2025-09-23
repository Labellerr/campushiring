import argparse
from ultralytics import YOLO

def prepare_data(path)
    return {"train": path + "/train", "val": path + "/val"}

def create_model(name="yolov8n-seg.pt"):
    model = YOLO(name)
    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data")
    args = parser.parse_args()
    d = prepare_data(args.data_dir)
    train_model(m, d, epochs=5)
