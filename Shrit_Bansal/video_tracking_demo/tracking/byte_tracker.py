import cv2
import json
import math
import shutil
import subprocess
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
    """
    Prefer browser-playable H.264 (avc1) if available; fall back to mp4v/XVID.
    Returns (writer, actual_output_path, codec_name) or (None, None, None).
    """
    base = Path(base_output_path)
    candidates = [
        ("avc1", ".mp4"),  # H.264 (browser-friendly)
        ("mp4v", ".mp4"),  # MPEG-4 Part 2 (not browser-friendly)
        ("XVID", ".avi"),  # AVI fallback (not browser-friendly)
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

def _has_ffmpeg():
    return shutil.which("ffmpeg") is not None

def _transcode_to_h264(src_path: str, dst_path: str) -> bool:
    # Fast, broadly compatible H.264 for web playback
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-i", src_path,
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
        "-preset", "veryfast", "-crf", "23",
        "-an", dst_path
    ]
    return subprocess.run(cmd).returncode == 0

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
            return False, "Failed to initialize VideoWriter with avc1/mp4v/XVID"

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
                    class_name = model.names.get(cls_id, f'class_{cls_id}')
                    color = (0, 255, 0) if 'person' in class_name.lower() else (255, 0, 0)

                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    label = f"{class_name}-#{track_id}"
                    text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                    cv2.rectangle(frame, (x1, y1 - text_size[1] - 10), (x1 + text_size[0], y1), color, -1)
                    cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                    frame_objects.append({
                        "id": int(track_id),
                        "class": class_name,
                        "confidence": float(conf),
                        "bbox": [x1, y1, x2, y2]
                    })

            results_data.append({"frame_id": frame_id, "objects": frame_objects})
            out.write(frame)

        out.release()

        # If we didn't get avc1, try to transcode to H.264 MP4 for browser playback
        final_out_path = actual_out_path
        dst_mp4 = str(Path(output_path).with_suffix(".mp4"))
        need_transcode = (Path(actual_out_path).suffix.lower() != ".mp4") or (used_codec.lower() != "avc1")
        if need_transcode and _has_ffmpeg():
            if _transcode_to_h264(actual_out_path, dst_mp4):
                final_out_path = dst_mp4

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(results_data, f, indent=2)

        return True, f"Saved {frame_id} frames using {used_codec}; final: {final_out_path}"
    except Exception as e:
        return False, f"An error occurred during processing: {str(e)}"
