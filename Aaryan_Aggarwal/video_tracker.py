# video_tracker.py
import cv2
import json
import os
from ultralytics import YOLO

def track_video(input_path, output_path, model_weights, json_path):
    """
    Track vehicles and pedestrians in a video using YOLOv8 with ByteTrack
    """
    try:
        # Load model
        model = YOLO(model_weights)
        
        # Open video
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            return False, f"Could not open video file: {input_path}"
            
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Video writer
        fourcc = cv2.VideoWriter_fourcc(*'avc1')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        results_data = []
        frame_id = 0
        
        # Process video with tracking
        results = model.track(
            source=input_path,
            imgsz=320,
            conf=0.5,
            iou=0.7,
            tracker="bytetrack.yaml",
            stream=True,
            verbose=False
        )
        
        for result in results:
            frame_id += 1
            frame = result.orig_img.copy()
            frame_objects = []
            
            if result.boxes is not None and result.boxes.id is not None:
                boxes = result.boxes.xyxy.cpu().numpy()
                track_ids = result.boxes.id.cpu().numpy().astype(int)
                confidences = result.boxes.conf.cpu().numpy()
                classes = result.boxes.cls.cpu().numpy().astype(int)
                
                for box, track_id, conf, cls_id in zip(boxes, track_ids, confidences, classes):
                    x1, y1, x2, y2 = map(int, box)
                    
                    # Get class name
                    class_name = model.names.get(cls_id, f'class_{cls_id}')
                    
                    # Choose color based on class
                    if 'person' in class_name.lower():
                        color = (0, 255, 0)  # Green for pedestrians
                    else:
                        color = (255, 0, 0)  # Blue for vehicles
                    
                    # Draw bounding box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    
                    # Add label
                    label = f"{class_name}-{track_id} ({conf:.2f})"
                    text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                    
                    # Draw background for text
                    cv2.rectangle(frame, (x1, y1 - text_size[1] - 10), 
                                (x1 + text_size[0], y1), color, -1)
                    
                    # Add text
                    cv2.putText(frame, label, (x1, y1 - 5),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    
                    # Store results
                    frame_objects.append({
                        "id": int(track_id),
                        "class": class_name,
                        "confidence": float(conf),
                        "bbox": [x1, y1, x2, y2]
                    })
            
            # Add frame info to results
            results_data.append({
                "frame_id": frame_id,
                "timestamp": frame_id / fps if fps > 0 else 0,
                "objects": frame_objects
            })
            
            # Write frame to output video
            out.write(frame)

        # Clean up
        cap.release()
        out.release()

        # Save JSON results
        with open(json_path, "w") as f:
            json.dump(results_data, f, indent=2)

        return True, f"Successfully processed {frame_id} frames"

    except Exception as e:
        return False, f"An error occurred during processing: {str(e)}"