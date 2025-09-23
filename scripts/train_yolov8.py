#!/usr/bin/env python3
"""
train_yolov8.py
Train YOLOv8 (detect or segment) using ultralytics API.

Usage (example, run from project root):
python scripts/train_yolov8.py \
  --data /content/drive/MyDrive/campushiring/arvind_mankotia_labellerr/data.yaml \
  --model yolov8n-seg.pt \
  --task segment \
  --name labellerr_run \
  --epochs 100 \
  --imgsz 640 \
  --batch 8

Notes:
- For bbox-only labels use model=yolov8n.pt and --task detect
- Requires `ultralytics` package.
"""

import argparse
import os
from ultralytics import YOLO
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True, help="Path to data.yaml (project root)")
    p.add_argument("--model", default="yolov8n.pt", help="Base model or weights (yolov8n.pt or yolov8n-seg.pt)")
    p.add_argument("--task", choices=["detect", "segment"], default="detect", help="Task: detect or segment")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--name", default="labellerr_yolov8", help="Run name under runs/train/")
    p.add_argument("--resume", action="store_true", help="Resume from last training run if present")
    p.add_argument("--device", default=None, help="Device string for ultralytics (e.g. 0, 0,1 or cpu). Leave None for default.")
    return p.parse_args()

def main():
    args = parse_args()
    logging.info("Starting YOLOv8 training")
    logging.info(f"Args: {args}")

    # Validate paths
    if not os.path.exists(args.data):
        raise FileNotFoundError(f"data.yaml not found: {args.data}")

    # Prepare YOLO object
    logging.info(f"Loading model: {args.model}")
    model = YOLO(args.model)

    # train
    train_kwargs = dict(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        name=args.name,
        device=args.device,
    )

    # ultralytics CLI uses task as separate argument if needed via model file (yolov8n-seg.pt)
    # We'll call model.train with the args; for segmentation the model file should be seg variant.
    try:
        model.train(**train_kwargs)
    except Exception as e:
        logging.exception("Training failed. See error above.")
        raise e

    # Save info
    finished_at = datetime.utcnow().isoformat() + "Z"
    logging.info(f"Training finished at {finished_at}. Check runs/train/{args.name}/")

if __name__ == "__main__":
    main()
