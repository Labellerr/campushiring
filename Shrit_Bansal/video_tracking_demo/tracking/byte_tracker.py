import cv2
import json
import math
from pathlib import Path
from ultralytics import YOLO

def _safe_fps(val):
    try:
        v = float(val)
        if not math.isfinite(v) or v <= 0:
            return 25.0
        return v
    except Exception:
        return 25.0

def _open_writer(base_output_path: str, fps: float, size: tuple[int, int]):
    base = Path(base_output_path)
    candidates = [
        ("mp4v", ".mp4"),
        ("XVID", ".avi"),
        ("avc1", ".mp4"),
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

def _get_class_name(model, cls_id: int) -> str:
    """
    Robustly fetch a class name from Ultralytics model.names which can be a list/tuple
    or a dict with int or string keys.
    """
    names = getattr(model, "names", None)
    if isinstance(names, (list, tuple)):
        if 0 <= cls_id < len(names):
            return str(names[cls_id])
    elif isinstance(names, dict):
        # try int key, then string variants
        return str(
            names.get(cls_id)
            or names.get(int(cls_id), None)
            or names.get(str(int(cls_id)), None)
            or names.get(str(cls_id), f"class_{cls_id}")
        )
    return f"class_{cls_id}"

def _map_binary(name: str) -> tuple[str, tuple]:
    """
    Map any detector name to strictly two labels:
    - 'pedestrian' if the name indicates a person
    - 'vehicle' otherwise
    Colors are BGR as required by OpenCV (green for pedestrians, blue for vehicles).
    """
    n = name.strip().lower()
    if any(tok in n for tok in ("person", "pedestrian", "people", "human")):
        return "pedestrian", (0, 255, 0)   # green (BGR)
    return "vehicle", (255, 0, 0)          # blue (BGR)

def track_video(input_path, output_path, model_weights, json_path):
    try:
        model = YOLO(model_weights)

        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            return False, f"Could not open video file: {input_path}"

        fps = _safe_fps(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        out, actual_out_path, used_codec = _open_writer(output_path, fps, (width, height))
        if out is None:
            return False, "Failed to initialize VideoWriter with mp4v/XVID/avc1"

        results_data = []
        frame_id = 0

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

                for box, track_id, conf, cls_id in zip(boxes, track_ids, confidences, classes):
                    x1, y1, x2, y2 = map(int, box)

                    # Robust class-name fetch, then strict two-class mapping
                    raw_name = _get_class_name(model, int(cls_id))          # model.names access can vary [Ultralytics]
                    normalized, color = _map_binary(raw_name)               # person -> pedestrian, else vehicle

                    # Draw (BGR color order for OpenCV)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    label = f"{normalized.title()}-#{track_id}"
                    text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                    cv2.rectangle(frame, (x1, y1 - text_size[1] - 10), (x1 + text_size[0], y1), color, -1)
                    cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                    frame_objects.append({
                        "id": int(track_id),
                        "class": normalized,     # strictly 'pedestrian' or 'vehicle'
                        "confidence": float(conf),
                        "bbox": [x1, y1, x2, y2]
                    })

            results_data.append({"frame_id": frame_id, "objects": frame_objects})
            out.write(frame)

        out.release()
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(results_data, f, indent=2)

        return True, f"Saved {frame_id} frames using {used_codec} at {actual_out_path}"
    except Exception as e:
        return False, f"An error occurred during processing: {str(e)}"
