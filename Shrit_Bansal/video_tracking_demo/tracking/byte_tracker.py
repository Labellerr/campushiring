import cv2
import json
from ultralytics import YOLO

def track_video(input_path, output_path, model_weights, json_path, progress_callback=None):
    """
    Runs YOLOv8 + ByteTrack on input_path, writes tracked video to output_path,
    JSON to json_path. Calls progress_callback(processed_frames, total_frames)
    if provided to report progress.
    """
    try:
        model = YOLO(model_weights)

        # Open video and get metadata
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            return False, f"Cannot open video: {input_path}"
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        # Video writer with H.264 codec
        fourcc = cv2.VideoWriter_fourcc(*'avc1')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        if not out.isOpened():
            return False, f"Cannot write output video: {output_path}"

        results_data = []
        frame_id = 0

        # Run tracker
        results = model.track(
            source=input_path,
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

                for box, tid, conf, cid in zip(boxes, track_ids, confidences, classes):
                    x1, y1, x2, y2 = map(int, box)
                    cls_name = model.names.get(cid, f"class_{cid}")
                    color = (0,255,0) if "pedestrian" in cls_name.lower() else (255,0,0)

                    # Draw box and label
                    cv2.rectangle(frame, (x1,y1),(x2,y2), color, 2)
                    label = f"{cls_name}#{tid}"
                    text_sz = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                    cv2.rectangle(frame, (x1,y1-text_sz[1]-6),(x1+text_sz[0],y1), color, -1)
                    cv2.putText(frame, label, (x1, y1-4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

                    frame_objects.append({
                        "id": int(tid),
                        "class": cls_name,
                        "confidence": float(conf),
                        "bbox": [x1,y1,x2,y2]
                    })

            results_data.append({"frame_id": frame_id, "objects": frame_objects})
            out.write(frame)

            # Update progress
            if progress_callback:
                progress_callback(frame_id, total_frames)

        out.release()
        with open(json_path, "w") as jf:
            json.dump(results_data, jf, indent=2)

        return True, None

    except Exception as e:
        return False, str(e)
