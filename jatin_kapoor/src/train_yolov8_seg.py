#!/usr/bin/env python3
import argparse
from ultralytics import YOLO
from src.utils import resolve_dataset_yaml

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='yolov8n-seg.pt')
    ap.add_argument('--data', default='data/dataset.yaml')
    ap.add_argument('--epochs', type=int, default=5)
    ap.add_argument('--img', type=int, default=640)
    ap.add_argument('--batch', type=int, default=2)
    ap.add_argument('--project', default='runs/segment')
    ap.add_argument('--name', default='labeller_seg_v2')
    ap.add_argument('--device', default='')
    args = ap.parse_args()

    resolved = resolve_dataset_yaml(args.data, 'data/_resolved.yaml')
    print(f"[INFO] Using dataset YAML: {resolved}")

    model = YOLO(args.model)
    model.train(data=resolved, epochs=args.epochs, imgsz=args.img, batch=args.batch,
                project=args.project, name=args.name, device=args.device or None,
                patience=10, cos_lr=True, augment=True, verbose=True)

if __name__ == '__main__':
    main()
