"""
Run ByteTrack on a video with YOLO detections. Expect per-frame detections in a simple JSONL:
{"frame": 0, "detections": [{"bbox_xyxy":[x1,y1,x2,y2],"score":0.9,"cls":0}, ...]}
"""
import os, json, cv2

def run_tracking(video_path, det_jsonl_path, out_video="outputs/tracked.mp4"):
    os.makedirs("outputs", exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)); FPS = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(out_video, fourcc, FPS if FPS>0 else 25, (W,H))
    # TODO: init ByteTrack tracker object here (per repo examples)
    frame_idx = 0
    with open(det_jsonl_path) as f:
        for line in f:
            dets = json.loads(line)["detections"]
            # TODO: feed dets to tracker, get tracks
            # draw tracks
            ret, frame = cap.read()
            if not ret: break
            # TODO: draw boxes/IDs
            vw.write(frame)
            frame_idx += 1
    cap.release(); vw.release()
    print("Wrote", out_video)

if __name__ == "__main__":
    import sys
    if len(sys.argv)<3:
        print("Usage: python src/bytetrack_integration.py input.mp4 detections.jsonl")
        raise SystemExit(1)
    run_tracking(sys.argv[1], sys.argv[2])
