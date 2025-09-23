#!/usr/bin/env python3
"""
run_bytetrack_export_json.py
Run ByteTrack on detections and export results.json

Usage examples:
# 1) Using precomputed detections (from infer_save_detections.py)
python scripts/run_bytetrack_export_json.py --detections results/detections_raw.json --output results/results.json --fps 30

# 2) Run YOLO on video and track in one step
python scripts/run_bytetrack_export_json.py --video /content/test_video.mp4 --weights runs/train/.../weights/best.pt --output results/results.json
"""

import argparse, os, json, numpy as np, sys
from pathlib import Path
import importlib.util
from tqdm import tqdm

# Minimal fallback tracker (IOU-based) if ByteTrack not available
class SimpleIOUTracker:
    def __init__(self, iou_threshold=0.3, max_lost=30):
        self.next_id = 1
        self.tracks = {}  # id -> {bbox, lost}
        self.iou_thr = iou_threshold
        self.max_lost = max_lost

    @staticmethod
    def iou(a, b):
        # a,b = [x1,y1,x2,y2]
        xa1,ya1,xa2,ya2 = a; xb1,yb1,xb2,yb2 = b
        xi1 = max(xa1, xb1); yi1 = max(ya1, yb1)
        xi2 = min(xa2, xb2); yi2 = min(ya2, yb2)
        iw = max(0, xi2 - xi1); ih = max(0, yi2 - yi1)
        inter = iw * ih
        A = max(0, (xa2 - xa1)) * max(0, (ya2 - ya1))
        B = max(0, (xb2 - xb1)) * max(0, (yb2 - yb1))
        union = A + B - inter
        return inter/union if union>0 else 0

    def update(self, detections):
        # detections: list of (bbox,score,class)
        assigned = {}
        out = []
        # match existing tracks to detections by IOU greedy
        for tid, t in list(self.tracks.items()):
            best_iou = 0; best_idx = -1
            for i, det in enumerate(detections):
                if i in assigned: continue
                iouv = self.iou(t["bbox"], det[0])
                if iouv > best_iou:
                    best_iou = iouv; best_idx = i
            if best_iou >= self.iou_thr and best_idx >=0:
                # update track
                bbox, score, cls = detections[best_idx]
                t["bbox"] = bbox; t["lost"] = 0; t["cls"] = cls; t["score"]=score
                assigned[best_idx] = tid
                out.append((tid, bbox, score, cls))
            else:
                t["lost"] += 1
                if t["lost"] > self.max_lost:
                    del self.tracks[tid]
        # create tracks for unassigned detections
        for i, det in enumerate(detections):
            if i in assigned: continue
            bbox,score,cls = det
            tid = self.next_id; self.next_id += 1
            self.tracks[tid] = {"bbox": bbox, "lost": 0, "cls": cls, "score": score}
            out.append((tid,bbox,score,cls))
        return out

def try_import_bytetrack(bytetrack_repo_path):
    # try the canonical ByteTrack path
    # if success returns BYTETracker class constructor (callable)
    repo = Path(bytetrack_repo_path)
    if not repo.exists():
        return None
    # typical path: ByteTrack/yolox/tracking_utils/byte_tracker.py with class BYTETracker
    bt_path = repo / "yolox" / "tracking_utils" / "byte_tracker.py"
    if not bt_path.exists():
        # other forks have different structure; try to locate "byte_tracker" anywhere
        candidates = list(repo.rglob("byte_tracker.py"))
        if candidates:
            bt_path = candidates[0]
        else:
            return None
    spec = importlib.util.spec_from_file_location("byte_tracker", str(bt_path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if hasattr(mod, "BYTETracker"):
        return mod.BYTETracker
    return None

def load_detections_json(detections_json):
    with open(detections_json) as f:
        d = json.load(f)
    # expect dict with key 'detections' list or a list directly
    if isinstance(d, dict) and "detections" in d:
        entries = d["detections"]
    else:
        entries = d
    # group by frame
    frames = {}
    for e in entries:
        fno = int(e.get("frame", e.get("frame_id", 0)))
        bbox = e["bbox"]  # [x1,y1,x2,y2]
        score = float(e.get("score", 1.0))
        cls = int(e.get("class", 0))
        frames.setdefault(fno, []).append((bbox, score, cls))
    return frames

def run_with_precomputed(dets_frames, fps, bytetrack_cls, output_path, class_names):
    # dets_frames: dict frame->list of (bbox,score,cls)
    results = []
    # init tracker
    if bytetrack_cls:
        # instantiate with default params (these repos expect args dict)
        try:
            tracker = bytetrack_cls(track_thresh=0.5, track_buffer=30)
        except Exception:
            # some versions require a config object; fall back to simple tracker
            tracker = SimpleIOUTracker()
    else:
        tracker = SimpleIOUTracker()
    # iterate frames in order
    for frame in tqdm(sorted(dets_frames.keys())):
        dets = dets_frames[frame]
        # convert to numpy detections if ByteTrack expects np array Nx5 (x1,y1,x2,y2,score)
        if hasattr(tracker, "update") and not isinstance(tracker, SimpleIOUTracker):
            # try ByteTrack update semantics: tracker.update( np.array([...]), img_info, frames=frame )
            try:
                import numpy as np
                dets_np = np.array([[d[0][0], d[0][1], d[0][2], d[0][3], d[1]] for d in dets], dtype=float)
                online_targets = tracker.update(dets_np, None, [0,0])  # some forks expect different args
                # adapt to reading online_targets
                for t in online_targets:
                    # typical object has attributes tlwh / tlbr / track_id
                    try:
                        track_id = int(t.track_id)
                        tlbr = getattr(t, "tlbr", None)
                        if tlbr is None:
                            # some use t.tlwh
                            tlwh = getattr(t, "tlwh", None)
                            if tlwh is not None:
                                x, y, w, h = tlwh
                                bbox = [x, y, x+w, y+h]
                            else:
                                continue
                        else:
                            bbox = [float(x) for x in tlbr]
                        class_id = int(getattr(t, "cls", 0)) if hasattr(t, "cls") else 0
                        score = float(getattr(t, "score", 0.0)) if hasattr(t, "score") else 0.0
                        results.append({"frame": frame, "track_id": track_id, "class": class_names[class_id] if class_id < len(class_names) else str(class_id), "bbox": bbox, "score": score})
                    except Exception:
                        continue
                continue
            except Exception:
                # fallthrough to SimpleIOU tracker logic
                pass
        # fallback simple tracker update
        out = tracker.update(dets)
        for tid, bbox, score, cls in out:
            results.append({"frame": frame, "track_id": int(tid), "class": class_names[cls] if cls < len(class_names) else str(cls), "bbox": [float(x) for x in bbox], "score": float(score)})
    # write results
    out_json = {"video": None, "frame_rate": fps, "tracks": results}
    with open(output_path, "w") as f:
        json.dump(out_json, f, indent=2)
    return out_json

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--detections", help="Path to detections_raw.json (from infer_save_detections.py). If not provided, specify --video and --weights")
    parser.add_argument("--video", help="Path to input video (optional)")
    parser.add_argument("--weights", help="YOLO weights (if feeding YOLO -> bytetrack in one script)")
    parser.add_argument("--bytetrack_repo", default="/content/ByteTrack", help="Path to cloned ByteTrack repo")
    parser.add_argument("--output", default="results/results.json", help="Path to output results.json")
    parser.add_argument("--fps", type=int, default=30, help="Frame rate for video / export")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold when computing detections from YOLO")
    parser.add_argument("--classes_txt", default="data/classes.txt", help="Path to classes.txt for mapping class ids to names")
    args = parser.parse_args()

    # load class names
    if os.path.exists(args.classes_txt):
        with open(args.classes_txt) as f:
            class_names = [x.strip() for x in f if x.strip()]
    else:
        class_names = []

    # try import ByteTrack
    bytetrack_cls = try_import_bytetrack(args.bytetrack_repo)
    if bytetrack_cls:
        print("ByteTrack BYTETracker class found:", bytetrack_cls)
    else:
        print("ByteTrack BYTETracker class NOT found. Falling back to SimpleIOUTracker (less robust).")

    # If detections json provided -> load and run tracker
    if args.detections:
        frames = load_detections_json(args.detections)
        out = run_with_precomputed(frames, args.fps, bytetrack_cls, args.output, class_names)
        print("Wrote tracking results to", args.output)
        return

    # Else fallback: run YOLO on video then track
    if args.video and args.weights:
        # perform on-the-fly inference per frame using ultralytics (avoid writing intermediate file)
        try:
            from ultralytics import YOLO
        except Exception as e:
            print("Install ultralytics first: pip install ultralytics"); raise e
        model = YOLO(args.weights)
        import cv2
        cap = cv2.VideoCapture(args.video)
        if not cap.isOpened():
            raise RuntimeError("Cannot open video")
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        detections_by_frame = {}
        frame_no = 0
        pbar = tqdm(total=frame_count)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_no += 1
            preds = model.predict(frame, conf=args.conf, verbose=False)[0]
            boxes = getattr(preds, "boxes", [])
            arr = []
            for b in boxes:
                xy = b.xyxy[0].cpu().numpy().tolist()
                conf = float(b.conf[0])
                cls = int(b.cls[0])
                arr.append((xy, conf, cls))
            detections_by_frame[frame_no] = arr
            pbar.update(1)
        pbar.close(); cap.release()
        out = run_with_precomputed(detections_by_frame, args.fps, bytetrack_cls, args.output, class_names)
        print("Wrote tracking results to", args.output)
        return

    parser.error("Either --detections OR both --video and --weights must be provided")

if __name__ == "__main__":
    main()
