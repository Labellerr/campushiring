# bytetrack_integration.py
import cv2, json, os
import numpy as np
from ultralytics import YOLO

# For this script, use the ByteTrack wrapper you installed; import its tracker class
# Example import (adjust based on the package you installed):
try:
    from bytetrack.trackers.byte_tracker import BYTETracker
except Exception as e:
    # Provide a simple local tracker fallback (not as good)
    BYTETracker = None
    print("ByteTrack import failed; install a compatible bytetrack package. Error:", e)

def run_video_tracking(video_path, model_path, output_json, tracker_cfg=None, conf=0.25):
    model = YOLO(model_path)
    cap = cv2.VideoCapture(video_path)
    frame_idx = 0
    tracks_output = []
    tracker = None
    if BYTETracker:
        # tracker_cfg example - adjust according to the ByteTrack API you're using:
        tracker = BYTETracker(track_thresh=0.5, match_thresh=0.8, min_box_area=10)
    else:
        # Very simple tracker fallback: track by nearest bbox between consecutive frames
        tracker = None

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1
        res = model.predict(source=frame, conf=conf, imgsz=640)
        r = res[0]
        if hasattr(r, "masks") and r.masks is not None and r.masks.data is not None:
            boxes = r.boxes.xyxy.cpu().numpy()  # x1,y1,x2,y2
            scores = r.boxes.conf.cpu().numpy()
            classes = r.boxes.cls.cpu().numpy().astype(int)
            dets = []
            for i, box in enumerate(boxes):
                x1,y1,x2,y2 = box.tolist()
                w = x2-x1; h = y2-y1
                dets.append([x1, y1, w, h, float(scores[i]), int(classes[i])])

            # feed to BYTETracker - different APIs expect different formats (tlwh and score)
            if BYTETracker:
                # Convert to tlwh + score
                tlwhs = [[d[0], d[1], d[2], d[3]] for d in dets]
                scores_list = [d[4] for d in dets]
                dets_for_tracker = np.array([[d[0],d[1],d[2],d[3],d[4]] for d in dets])
                online_targets = tracker.update(dets_for_tracker, frame.shape[:2])
                for t in online_targets:
                    tlwh = t.tlwh
                    track_id = t.track_id
                    cls = int(t.cls) if hasattr(t, "cls") else -1
                    tracks_output.append({
                        "frame": frame_idx,
                        "track_id": int(track_id),
                        "class": cls,
                        "bbox": [float(tlwh[0]), float(tlwh[1]), float(tlwh[2]), float(tlwh[3])],
                    })
            else:
                # Very naive single-frame tracking (no persistence)
                for i, d in enumerate(dets):
                    tracks_output.append({
                        "frame": frame_idx,
                        "track_id": i,
                        "class": d[5],
                        "bbox": [d[0], d[1], d[2], d[3]],
                    })

    with open(output_json, "w") as f:
        json.dump(tracks_output, f, indent=2)
    print("Saved tracking results to", output_json)
    cap.release()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--model", default="runs/seg/yolov8n_seg_labellerr/weights/last.pt")
    parser.add_argument("--out", default="results/tracking_results.json")
    args = parser.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    run_video_tracking(args.video, args.model, args.out)
