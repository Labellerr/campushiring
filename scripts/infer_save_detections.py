#!/usr/bin/env python3
"""
infer_save_detections.py
Run YOLOv8 inference on a video OR a folder of images and save per-frame detections.

Outputs:
- detections_raw.json : list of dicts, each dict: {
    "frame": int, "image": "filename", "bbox": [x1,y1,x2,y2], "score": float, "class": int, "class_name": str, "mask": optional polygon list
  }

Usage:
python scripts/infer_save_detections.py --weights runs/train/labellerr_yolov8/weights/best.pt --video /content/test_video.mp4 --output results/detections_raw.json --conf 0.25
or
python scripts/infer_save_detections.py --weights runs/train/.../weights/best.pt --images_dir data/images/test --output results/detections_raw.json
"""

import argparse, os, json, cv2, math
from ultralytics import YOLO
from tqdm import tqdm
import numpy as np

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", required=True, help="Path to trained weights (best.pt)")
    p.add_argument("--video", default=None, help="Path to input video. If omitted, use --images_dir")
    p.add_argument("--images_dir", default=None, help="Path to folder of images (alternative to --video)")
    p.add_argument("--output", default="results/detections_raw.json", help="Output JSON path")
    p.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    p.add_argument("--device", default=None, help="Device for ultralytics (e.g. 0 or 'cpu')")
    p.add_argument("--max_frames", type=int, default=None, help="Limit number of frames (for debugging)")
    return p.parse_args()

def xyxy_from_box(box):
    # box: ultralytics 'box' object with .xyxy and .conf and .cls
    # If box.xyxy is tensor 1x4: convert to list
    try:
        xy = box.xyxy[0].cpu().numpy().tolist()
    except Exception:
        # fallback if already numpy
        xy = np.array(box.xyxy).reshape(-1).tolist()
    return [float(x) for x in xy]

def process_video(model, video_path, conf_thresh, device, max_frames):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    frame_no = 0
    results_list = []
    pbar = tqdm(total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_no += 1
        if max_frames and frame_no > max_frames:
            break
        # ultralytics model.predict supports passing numpy images
        preds = model.predict(frame, conf=conf_thresh, verbose=False)[0]
        # preds.boxes is a Boxes object; preds.masks if seg model
        boxes = getattr(preds, "boxes", [])
        masks = getattr(preds, "masks", None)
        for i, box in enumerate(boxes):
            xyxy = box.xyxy.cpu().numpy()[0].tolist()
            conf = float(box.conf.cpu().numpy()[0])
            cls = int(box.cls.cpu().numpy()[0])
            entry = {
                "frame": frame_no,
                "image": None,
                "bbox": [float(x) for x in xyxy],
                "score": conf,
                "class": cls
            }
            # masks: if seg, masks.data or preds.masks.xy might be available; we attempt a polygon extraction
            if masks is not None:
                try:
                    # masks.xy exists in ultralytics results as list of polygons per instance (list[list[np.array]])
                    polys = masks.xy[i].cpu().numpy().tolist()
                    entry["mask_polygon"] = polys
                except Exception:
                    pass
            results_list.append(entry)
        pbar.update(1)
    pbar.close()
    cap.release()
    # write results
    return results_list, fps

def process_images_dir(model, images_dir, conf_thresh, max_frames):
    img_files = sorted([os.path.join(images_dir, f) for f in os.listdir(images_dir) if f.lower().endswith(('.jpg','.jpeg','.png'))])
    if max_frames:
        img_files = img_files[:max_frames]
    results_list = []
    for idx, imgp in enumerate(tqdm(img_files)):
        img = cv2.imread(imgp)
        if img is None:
            continue
        preds = model.predict(img, conf=conf_thresh, verbose=False)[0]
        boxes = getattr(preds, "boxes", [])
        masks = getattr(preds, "masks", None)
        for i, box in enumerate(boxes):
            xyxy = box.xyxy.cpu().numpy()[0].tolist()
            conf = float(box.conf.cpu().numpy()[0])
            cls = int(box.cls.cpu().numpy()[0])
            entry = {
                "frame": idx + 1,
                "image": os.path.basename(imgp),
                "bbox": [float(x) for x in xyxy],
                "score": conf,
                "class": cls
            }
            if masks is not None:
                try:
                    polys = masks.xy[i].cpu().numpy().tolist()
                    entry["mask_polygon"] = polys
                except Exception:
                    pass
            results_list.append(entry)
    return results_list, None

def main():
    args = parse_args()
    model = YOLO(args.weights)
    if args.video and args.images_dir:
        raise ValueError("Provide either --video or --images_dir, not both.")
    if args.video:
        results, fps = process_video(model, args.video, args.conf, args.device, args.max_frames)
    elif args.images_dir:
        results, fps = process_images_dir(model, args.images_dir, args.conf, args.max_frames)
    else:
        raise ValueError("Provide --video or --images_dir")
    # Save JSON
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump({"fps": fps, "detections": results}, f, indent=2)
    print(f"Wrote {len(results)} detection entries to {args.output}")

if __name__ == "__main__":
    main()
