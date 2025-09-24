#!/usr/bin/env python3
import argparse, os, cv2
from pathlib import Path
from ultralytics import YOLO
from src.utils import ensure_dir, save_json

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--weights', default='models/best.pt')
    ap.add_argument('--source', default='samples/sample.mp4')
    ap.add_argument('--out_dir', default='outputs')
    ap.add_argument('--conf', type=float, default=0.25)
    ap.add_argument('--iou', type=float, default=0.45)
    ap.add_argument('--device', default='')
    ap.add_argument('--tracker', default='bytetrack.yaml')
    args = ap.parse_args()

    ensure_dir(args.out_dir)
    model = YOLO(args.weights)

    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {args.source}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 20
    W  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_video_path = os.path.join(args.out_dir, Path(args.source).stem + '_tracked.mp4')
    writer = cv2.VideoWriter(out_video_path, fourcc, fps, (W, H))

    results_json = {"video": os.path.basename(args.source), "model": args.weights, "tracks": []}
    frame_idx = -1
    for result in model.track(source=args.source, stream=True, conf=args.conf, iou=args.iou,
                              tracker=args.tracker, device=args.device or None, persist=True, verbose=False):
        frame_idx += 1
        frame = result.plot()
        writer.write(frame)
        boxes = result.boxes
        if boxes is None:
            continue
        ids = boxes.id.cpu().tolist() if boxes.id is not None else [None]*len(boxes)
        xyxy = boxes.xyxy.cpu().tolist()
        clss = boxes.cls.cpu().tolist()
        confs = boxes.conf.cpu().tolist()
        for _id, bb, c, cf in zip(ids, xyxy, clss, confs):
            x1,y1,x2,y2 = bb
            results_json["tracks"].append({
                "frame": frame_idx, "id": int(_id) if _id is not None else -1,
                "cls": int(c), "conf": float(cf),
                "bbox_xyxy": [float(x1), float(y1), float(x2), float(y2)]
            })
    writer.release()
    out_json = os.path.join(args.out_dir, Path(args.source).stem + '_results.json')
    save_json(results_json, out_json)
    print("Saved:", out_video_path, "\nSaved:", out_json)

if __name__ == '__main__':
    main()
