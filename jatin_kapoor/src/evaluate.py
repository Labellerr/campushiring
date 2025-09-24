#!/usr/bin/env python3
import argparse, os, shutil
from ultralytics import YOLO
from src.utils import resolve_dataset_yaml

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--weights', default='models/best.pt')
    ap.add_argument('--data', default='data/dataset.yaml')
    ap.add_argument('--img', type=int, default=640)
    ap.add_argument('--split', default='val')
    ap.add_argument('--out_dir', default='reports')
    args = ap.parse_args()

    resolved = resolve_dataset_yaml(args.data, 'data/_resolved.yaml')
    model = YOLO(args.weights)
    metrics = model.val(data=resolved, imgsz=args.img, split=args.split, plots=True)
    os.makedirs(args.out_dir, exist_ok=True)
    try:
        save_dir = model.trainer.save_dir
        for p in ['PR_curve.png', 'confusion_matrix.png', 'results.png']:
            src = os.path.join(save_dir, p)
            if os.path.exists(src):
                shutil.copy(src, os.path.join(args.out_dir, p))
    except Exception as e:
        print('[WARN] Could not copy plots:', e)
    print(metrics)

if __name__ == '__main__':
    main()
