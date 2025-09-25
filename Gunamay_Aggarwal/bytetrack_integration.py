
"""
ByteTrack Integration with YOLO-Seg for Object Tracking
Labellerr AI Internship Assignment
"""

import cv2
import numpy as np
import torch
from ultralytics import YOLO
import sys
import os
from pathlib import Path
import json
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Tuple
import logging

# Add ByteTrack to path (assuming ByteTrack is cloned)
bytetrack_path = Path("ByteTrack")
if bytetrack_path.exists():
    sys.path.append(str(bytetrack_path))
    from yolox.tracker.byte_tracker import BYTETracker
    from yolox.tracker.basetrack import TrackState
else:
    logging.warning("ByteTrack not found. Please clone ByteTrack repository.")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class TrackingResult:
    """Data class for tracking results"""
    frame_id: int
    track_id: int
    bbox: List[float]  # [x1, y1, x2, y2]
    confidence: float
    class_id: int
    class_name: str
    mask: Optional[List] = None  # Segmentation mask

class YOLOByteTracker:
    def __init__(self, yolo_model_path: str, conf_thresh: float = 0.5, 
                 track_thresh: float = 0.5, track_buffer: int = 30,
                 match_thresh: float = 0.8, frame_rate: int = 30):
        """
        Initialize YOLO + ByteTrack pipeline

        Args:
            yolo_model_path: Path to trained YOLO segmentation model
            conf_thresh: Confidence threshold for detections
            track_thresh: Tracking threshold
            track_buffer: Buffer frames for lost tracks
            match_thresh: Matching threshold for tracking
            frame_rate: Video frame rate
        """
        self.yolo_model = YOLO(yolo_model_path)
        self.conf_thresh = conf_thresh

        # ByteTracker arguments
        class Args:
            track_thresh = track_thresh
            track_buffer = track_buffer
            match_thresh = match_thresh
            frame_rate = frame_rate
            mot20 = False

        self.tracker = BYTETracker(Args())
        self.tracking_results: List[TrackingResult] = []
        self.class_names = self.yolo_model.names

    def detect_and_track(self, frame: np.ndarray, frame_id: int) -> Tuple[np.ndarray, List[TrackingResult]]:
        """
        Run detection and tracking on a single frame

        Args:
            frame: Input frame
            frame_id: Frame number

        Returns:
            Annotated frame and tracking results
        """
        # YOLO detection
        results = self.yolo_model(frame, conf=self.conf_thresh, verbose=False)

        frame_results = []
        if len(results[0].boxes) > 0:
            # Extract detections
            boxes = results[0].boxes.xyxy.cpu().numpy()
            scores = results[0].boxes.conf.cpu().numpy()
            class_ids = results[0].boxes.cls.cpu().numpy().astype(int)

            # Prepare detections for ByteTracker
            # Format: [x1, y1, x2, y2, score]
            detections = np.column_stack([boxes, scores])

            # Update tracker
            online_tracks = self.tracker.update(detections, 
                                              [frame.shape[0], frame.shape[1]], 
                                              [frame.shape[0], frame.shape[1]])

            # Process tracking results
            for track in online_tracks:
                if track.state == TrackState.Tracked:
                    bbox = track.tlbr.tolist()  # [x1, y1, x2, y2]
                    track_id = track.track_id

                    # Find corresponding detection for class info
                    # This is a simplified matching - in practice you might want more sophisticated matching
                    best_match_idx = 0
                    best_iou = 0
                    for i, det_box in enumerate(boxes):
                        iou = self._calculate_iou(bbox, det_box)
                        if iou > best_iou:
                            best_iou = iou
                            best_match_idx = i

                    if best_iou > 0.3:  # Minimum IoU for matching
                        class_id = class_ids[best_match_idx]
                        confidence = scores[best_match_idx]
                        class_name = self.class_names[class_id]

                        # Extract mask if available
                        mask = None
                        if hasattr(results[0], 'masks') and results[0].masks is not None:
                            if best_match_idx < len(results[0].masks.data):
                                mask = results[0].masks.data[best_match_idx].cpu().numpy().tolist()

                        # Create tracking result
                        track_result = TrackingResult(
                            frame_id=frame_id,
                            track_id=track_id,
                            bbox=bbox,
                            confidence=confidence,
                            class_id=class_id,
                            class_name=class_name,
                            mask=mask
                        )

                        frame_results.append(track_result)
                        self.tracking_results.append(track_result)

        # Annotate frame
        annotated_frame = self._annotate_frame(frame.copy(), frame_results)

        return annotated_frame, frame_results

    def _calculate_iou(self, box1: List[float], box2: List[float]) -> float:
        """Calculate IoU between two bounding boxes"""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        if x2 <= x1 or y2 <= y1:
            return 0.0

        intersection = (x2 - x1) * (y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0.0

    def _annotate_frame(self, frame: np.ndarray, tracks: List[TrackingResult]) -> np.ndarray:
        """Annotate frame with tracking results"""
        for track in tracks:
            x1, y1, x2, y2 = map(int, track.bbox)

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Draw track ID and class
            label = f"ID:{track.track_id} {track.class_name} {track.confidence:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]

            cv2.rectangle(frame, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), (0, 255, 0), -1)
            cv2.putText(frame, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

        return frame

    def process_video(self, video_path: str, output_path: str = None) -> List[TrackingResult]:
        """
        Process entire video for tracking

        Args:
            video_path: Path to input video
            output_path: Path to save output video (optional)

        Returns:
            List of all tracking results
        """
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            logger.error(f"Cannot open video: {video_path}")
            return []

        # Video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        logger.info(f"Processing video: {video_path}")
        logger.info(f"Resolution: {width}x{height}, FPS: {fps}, Frames: {total_frames}")

        # Setup video writer if output path provided
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        frame_id = 0
        self.tracking_results = []  # Reset results

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Process frame
            annotated_frame, frame_tracks = self.detect_and_track(frame, frame_id)

            # Write frame if output path provided
            if writer:
                writer.write(annotated_frame)

            # Progress update
            if frame_id % 100 == 0:
                logger.info(f"Processed frame {frame_id}/{total_frames}")

            frame_id += 1

        # Cleanup
        cap.release()
        if writer:
            writer.release()

        logger.info(f"Video processing completed. Total tracks: {len(set(t.track_id for t in self.tracking_results))}")
        return self.tracking_results

    def export_results_json(self, output_path: str):
        """Export tracking results to JSON"""
        results_dict = {
            'video_info': {
                'total_frames': len(set(t.frame_id for t in self.tracking_results)),
                'total_tracks': len(set(t.track_id for t in self.tracking_results)),
                'classes': list(set(t.class_name for t in self.tracking_results))
            },
            'tracks': [asdict(track) for track in self.tracking_results]
        }

        with open(output_path, 'w') as f:
            json.dump(results_dict, f, indent=2)

        logger.info(f"Results exported to {output_path}")

def main():
    """Main demo function"""
    # Initialize tracker
    model_path = "runs/segment/yolo_seg_exp/weights/best.pt"  # Update with your model path
    tracker = YOLOByteTracker(model_path, conf_thresh=0.5)

    # Process video
    video_path = "demo_video.mp4"  # Update with your video path
    output_video = "tracked_output.mp4"
    output_json = "tracking_results.json"

    if Path(video_path).exists():
        # Process video
        results = tracker.process_video(video_path, output_video)

        # Export results
        tracker.export_results_json(output_json)

        # Print summary
        print(f"\nTracking Summary:")
        print(f"Total frames processed: {len(set(r.frame_id for r in results))}")
        print(f"Total unique tracks: {len(set(r.track_id for r in results))}")
        print(f"Classes detected: {set(r.class_name for r in results)}")
    else:
        logger.error(f"Video file not found: {video_path}")

if __name__ == "__main__":
    main()
