# byte_tracker.py
import cv2
import json
import math
from pathlib import Path
from ultralytics import YOLO

def _safe_fps(val):
    try:
        if val is None or math.isnan(val) or val <= 0:
            return 25.0  # sane default
        return float(val)
    except Exception:
        return 25.0

def _open_writer(base_path: str, fps: float, size: tuple[int, int]):
    """
    Try multiple codecs/containers in order of broadest availability.
    Returns (writer, actual_output_path, codec_name) or (None, None, None).
    """
    base = Path(base_path)
    candidates = [
        ("mp4v", ".mp4"),   # widely available for MP4 without OpenH264
        ("XVID", ".avi"),   # fallback to AVI on Windows
        ("avc1", ".mp4"),   # H.264; requires OpenH264 DLL on Windows
    ]
    for fourcc_name, ext in candidates:
        out_path = base.with_suffix(ext)
        writer = cv2.VideoWriter(
            str(out_path),
            cv2.VideoWriter_fourcc(*fourcc_name),
            max(fps, 1.0),
            size
        )
        if writer.isOpened():
            return writer, str(out_path), fourcc_name
    return None, None, None

def track_video(input_path, output_path, model_weights, json_path, progress_callback=None):
    """
    Runs YOLOv8 + ByteTrack on input_path, writes tracked video to output_path/json_path.
    Calls progress_callback(processed_frames, total_frames) if provided.
    """
    try:
        model = YOLO(model_weights)

        # Read metadata once
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            return False, f"Cannot open video: {input_path}"
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = _safe_fps(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        # Initialize writer with fallback codecs
        out, actual_out_path, codec_used = _open_writer(output_path, fps, (width, height))
        if out is None:
            return False, "Cannot initialize VideoWriter with mp4v/XVID/avc1"

        results_data = []
        frame_id = 0

        # Run tracker (use default bytetrack.yaml accessible to Ultralytics)
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
                    # Ultralytics v8 uses "person" for class 0 in COCO; keep generic fallback
                    cls_name = model.names.get(cid, f"class_{cid}")
                    color = (0, 255, 0) if "person" in cls_name.lower() else (255, 0, 0)

                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    label = f"{cls_name}#{tid}"
                    text_sz = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                    cv2.rectangle(frame, (x1, y1 - text_sz[1] - 6), (x1 + text_sz[0], y1), color, -1)
                    cv2.putText(frame, label, (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                    frame_objects.append({
                        "id": int(tid),
                        "class": cls_name,
                        "confidence": float(conf),
                        "bbox": [x1, y1, x2, y2],
                    })

            results_data.append({"frame_id": frame_id, "objects": frame_objects})
            out.write(frame)

            if progress_callback:
                progress_callback(frame_id, max(total_frames, frame_id))

        out.release()
        with open(json_path, "w", encoding="utf-8") as jf:
            json.dump(results_data, jf, indent=2)

        # If the actual output path extension differs from requested, signal no error (Streamlit just loads the file path)
        return True, None

    except Exception as e:
        return False, str(e)
