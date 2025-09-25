# track_video.py
import cv2
import numpy as np
from ultralytics import YOLO
from yolox.tracker.byte_tracker import BYTETracker
import sys
import argparse

# Fix lap import for ByteTrack
sys.modules['lap'] = __import__('lappy')  # Make sure lappy and dependencies are installed locally

def run_tracking(video_path, model_path, output_path, conf=0.25):
    # Load YOLO-Seg model
    model = YOLO(model_path)

    # Open video
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Initialize tracker
    tracker = BYTETracker(frame_rate=fps)

    # Output video
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # YOLO inference
        results = model.predict(frame, conf=conf, verbose=False)
        if len(results[0].boxes) > 0:
            dets = results[0].boxes.xyxy.cpu().numpy()  # bounding boxes
            scores = results[0].boxes.conf.cpu().numpy()  # confidence scores
            dets_bt = np.hstack((dets, scores[:, None]))
            tracked_objs = tracker.update(dets_bt, [height, width], [height, width])
        else:
            tracked_objs = []

        # Draw tracked objects
        for obj in tracked_objs:
            x1, y1, x2, y2, track_id = map(int, obj[:5])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f'ID: {track_id}', (x1, y1-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        out.write(frame)

        if frame_count % 50 == 0:
            print(f"Processed {frame_count} frames...")

    cap.release()
    out.release()
    print(f"Tracking complete. Output saved at: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, required=True, help="Path to input video")
    parser.add_argument("--model", type=str, required=True, help="Path to YOLO-Seg model weights")
    parser.add_argument("--output", type=str, default="tracked_output.mp4", help="Output video path")
    parser.add_argument("--conf", type=float, default=0.25, help="YOLO confidence threshold")
    args = parser.parse_args()

    run_tracking(args.video, args.model, args.output, args.conf)
