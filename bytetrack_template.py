
# ByteTrack Integration Template
import cv2
import numpy as np
import json
from collections import defaultdict
from ultralytics import YOLO
import torch

class YOLOByteTrackIntegration:
    def __init__(self, model_path, conf_thresh=0.5, high_thresh=0.6, low_thresh=0.1):
        self.model = YOLO(model_path)
        self.conf_thresh = conf_thresh
        self.high_thresh = high_thresh
        self.low_thresh = low_thresh
        self.tracker = self._initialize_tracker()

    def _initialize_tracker(self):
        """Initialize ByteTrack tracker"""
        try:
            from yolox.tracker.byte_tracker import BYTETracker
            tracker_args = {
                'track_thresh': self.high_thresh,
                'track_buffer': 30,
                'match_thresh': 0.8,
                'frame_rate': 30
            }
            return BYTETracker(**tracker_args)
        except ImportError:
            print("ByteTrack not installed. Installing...")
            os.system("pip install yolox")
            from yolox.tracker.byte_tracker import BYTETracker
            tracker_args = {
                'track_thresh': self.high_thresh,
                'track_buffer': 30,
                'match_thresh': 0.8,
                'frame_rate': 30
            }
            return BYTETracker(**tracker_args)

    def convert_yolo_to_tracker_format(self, results, frame_idx):
        """Convert YOLO results to ByteTracker format"""
        detections = []

        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for i, box in enumerate(boxes):
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = box.conf[0].cpu().numpy()
                    cls = int(box.cls[0].cpu().numpy())

                    # ByteTracker expects [x1, y1, x2, y2, conf, class]
                    detection = [x1, y1, x2, y2, conf, cls]
                    detections.append(detection)

        return np.array(detections) if detections else np.empty((0, 6))

    def process_video(self, video_path, output_path=None, visualize=True):
        """Process video with YOLO detection and ByteTrack tracking"""
        cap = cv2.VideoCapture(video_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        tracking_results = []
        frame_idx = 0

        # Setup video writer if output path provided
        if output_path and visualize:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        print(f"Processing video: {video_path}")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # YOLO inference
            results = self.model(frame, conf=self.conf_thresh, verbose=False)

            # Convert to tracker format
            detections = self.convert_yolo_to_tracker_format(results, frame_idx)

            # Update tracker
            if len(detections) > 0:
                tracks = self.tracker.update(detections, frame.shape[:2], frame.shape[:2])
            else:
                tracks = []

            # Extract tracking information
            frame_tracks = self.extract_tracking_info(tracks, frame_idx)
            tracking_results.extend(frame_tracks)

            # Visualize if requested
            if visualize:
                annotated_frame = self.draw_tracks(frame, tracks)

                if output_path:
                    out.write(annotated_frame)

                # Display frame (optional)
                cv2.imshow('Tracking', cv2.resize(annotated_frame, (960, 540)))
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            frame_idx += 1

            if frame_idx % 100 == 0:
                print(f"Processed {frame_idx} frames...")

        cap.release()
        if output_path and visualize:
            out.release()
        cv2.destroyAllWindows()

        print(f"Video processing complete. Processed {frame_idx} frames.")
        return tracking_results

    def extract_tracking_info(self, tracks, frame_idx):
        """Extract tracking information for JSON export"""
        frame_tracks = []

        for track in tracks:
            track_info = {
                'frame_number': frame_idx,
                'track_id': int(track.track_id),
                'class_id': int(track.cls),
                'class_name': self.model.names[int(track.cls)],
                'confidence': float(track.score),
                'bbox': {
                    'x1': float(track.tlbr[0]),
                    'y1': float(track.tlbr[1]), 
                    'x2': float(track.tlbr[2]),
                    'y2': float(track.tlbr[3])
                },
                'center': {
                    'x': float((track.tlbr[0] + track.tlbr[2]) / 2),
                    'y': float((track.tlbr[1] + track.tlbr[3]) / 2)
                }
            }
            frame_tracks.append(track_info)

        return frame_tracks

    def draw_tracks(self, frame, tracks):
        """Draw tracking results on frame"""
        annotated_frame = frame.copy()
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]

        for track in tracks:
            x1, y1, x2, y2 = map(int, track.tlbr)
            track_id = track.track_id
            class_name = self.model.names[int(track.cls)]
            confidence = track.score

            # Choose color based on track ID
            color = colors[track_id % len(colors)]

            # Draw bounding box
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)

            # Draw label
            label = f"ID:{track_id} {class_name} {confidence:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(annotated_frame, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), color, -1)
            cv2.putText(annotated_frame, label, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        return annotated_frame

    def export_tracking_results(self, tracking_results, output_path):
        """Export tracking results to JSON"""
        # Organize results by frame
        frames_data = defaultdict(list)
        for track in tracking_results:
            frames_data[track['frame_number']].append(track)

        # Create final JSON structure
        json_output = {
            "video_info": {
                "total_frames": max(frames_data.keys()) + 1 if frames_data else 0,
                "total_tracks": len(set(track['track_id'] for track in tracking_results)),
                "classes_detected": list(set(track['class_name'] for track in tracking_results))
            },
            "tracking_data": dict(frames_data)
        }

        with open(output_path, 'w') as f:
            json.dump(json_output, f, indent=2)

        print(f"Tracking results exported to: {output_path}")
        return json_output

# Usage example
if __name__ == "__main__":
    # Initialize tracker with trained model
    tracker = YOLOByteTrackIntegration(
        model_path='runs/segment/vehicle_pedestrian_seg/weights/best.pt'
    )

    # Process video
    video_path = 'test_video.mp4'
    tracking_results = tracker.process_video(
        video_path=video_path,
        output_path='tracked_output.mp4',
        visualize=True
    )

    # Export results
    tracker.export_tracking_results(tracking_results, 'tracking_results.json')
