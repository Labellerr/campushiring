import os
import threading
from typing import Any, Dict, List, Optional
import json


_model_lock = threading.Lock()
_model = None


def _default_model_path() -> str:
    # Default to the fast 1-epoch model path created earlier
    return (
        r"C:\\Users\\kiran\\Downloads\\Detecting cars and pedestrians.v1i.yolov8"
        r"\\runs\\yolov8n-seg-fast-1ep-cli\\weights\\best.pt"
    )


def get_model(model_path: Optional[str] = None):
    global _model
    if _model is not None:
        return _model
    with _model_lock:
        if _model is None:
            from ultralytics import YOLO  # lazy import
            path = model_path or _default_model_path()
            if not os.path.exists(path):
                raise FileNotFoundError(f"Model weights not found at: {path}")
            _model = YOLO(path)
    return _model


def run_segment(
    image_path: str,
    conf: float = 0.15,
    iou: float = 0.5,
    imgsz: int = 416,
    device: str = "cpu",
    save: bool = True,
    project: Optional[str] = None,
    name: str = "segment-api",
) -> Dict[str, Any]:
    model = get_model()
    kwargs = {
        "conf": conf,
        "iou": iou,
        "imgsz": imgsz,
        "device": device,
        "save": save,
        "project": project,
        "name": name,
        "verbose": False,
        "stream": False,
    }
    results = model.predict(source=image_path, **{k: v for k, v in kwargs.items() if v is not None})
    if not results:
        return {"detections": [], "save_dir": None}
    r0 = results[0]
    save_dir = getattr(r0, "save_dir", None)
    detections: List[Dict[str, Any]] = []
    names = r0.names if hasattr(r0, "names") else {}
    boxes = r0.boxes
    if boxes is not None:
        for i in range(len(boxes)):
            b = boxes[i]
            cls_id = int(b.cls.item()) if hasattr(b, "cls") else -1
            conf_v = float(b.conf.item()) if hasattr(b, "conf") else None
            xyxy = b.xyxy.tolist()[0] if hasattr(b, "xyxy") else None
            detections.append(
                {
                    "class_id": cls_id,
                    "class_name": names.get(cls_id, str(cls_id)) if isinstance(names, dict) else cls_id,
                    "confidence": conf_v,
                    "box_xyxy": xyxy,
                }
            )
    return {"detections": detections, "save_dir": str(save_dir) if save_dir else None}


def run_track(
    source: str,
    conf: float = 0.15,
    iou: float = 0.5,
    imgsz: int = 416,
    device: str = "cpu",
    tracker: str = "bytetrack.yaml",
    save: bool = True,
    project: Optional[str] = None,
    name: str = "track-api",
    persist: bool = True,
) -> Dict[str, Any]:
    model = get_model()
    kwargs = {
        "conf": conf,
        "iou": iou,
        "imgsz": imgsz,
        "device": device,
        "tracker": tracker,
        "save": save,
        "project": project,
        "name": name,
        "persist": persist,
        "verbose": False,
        "stream": False,
    }
    results = model.track(source=source, **{k: v for k, v in kwargs.items() if v is not None})
    # Ultralytics returns list-like; paths are under save_dir
    save_dir = None
    if results and hasattr(results[0], "save_dir"):
        save_dir = str(results[0].save_dir)
    return {"save_dir": save_dir}


def track_video_collect(
    source: str,
    conf: float = 0.15,
    iou: float = 0.5,
    imgsz: int = 416,
    device: str = "cpu",
    tracker: str = "bytetrack.yaml",
    save: bool = True,
    project: Optional[str] = None,
    name: str = "track-api",
    persist: bool = True,
) -> Dict[str, Any]:
    """Run ByteTrack tracking and collect per-frame objects with IDs.

    Returns a dict with keys: save_dir, results_json, tracks
    Each track item: {frame, id, class_id, class_name, confidence, box_xyxy}
    """
    model = get_model()
    kwargs = {
        "conf": conf,
        "iou": iou,
        "imgsz": imgsz,
        "device": device,
        "tracker": tracker,
        "save": save,
        "project": project,
        "name": name,
        "persist": persist,
        "verbose": False,
        "stream": True,
    }

    tracks: List[Dict[str, Any]] = []
    save_dir: Optional[str] = None
    names_map: Dict[int, str] = {}

    # Prepare progress file
    total_frames = None
    try:
        import cv2  # type: ignore
        cap = cv2.VideoCapture(source)
        if cap.isOpened():
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or None
        cap.release()
    except Exception:
        total_frames = None

    progress_path = None
    # Save under project/name even before results appear
    tmp_save_dir = os.path.join(project or "runs", name)
    os.makedirs(tmp_save_dir, exist_ok=True)
    progress_path = os.path.join(tmp_save_dir, "progress.json")

    def write_progress(idx: int):
        try:
            with open(progress_path, "w", encoding="utf-8") as f:
                json.dump({"processed": idx, "total": total_frames}, f)
        except Exception:
            pass

    for frame_idx, r in enumerate(model.track(source=source, **{k: v for k, v in kwargs.items() if v is not None})):
        if save_dir is None and hasattr(r, "save_dir") and r.save_dir is not None:
            save_dir = str(r.save_dir)
        if hasattr(r, "names") and isinstance(r.names, dict):
            names_map = r.names
        boxes = getattr(r, "boxes", None)
        if boxes is None:
            write_progress(frame_idx)
            continue
        ids = getattr(boxes, "id", None)
        for i in range(len(boxes)):
            b = boxes[i]
            cls_id = int(b.cls.item()) if hasattr(b, "cls") else -1
            conf_v = float(b.conf.item()) if hasattr(b, "conf") else None
            xyxy = b.xyxy.tolist()[0] if hasattr(b, "xyxy") else None
            obj_id = int(ids[i].item()) if ids is not None else None
            tracks.append(
                {
                    "frame": frame_idx,
                    "id": obj_id,
                    "class_id": cls_id,
                    "class_name": names_map.get(cls_id, str(cls_id)),
                    "confidence": conf_v,
                    "box_xyxy": xyxy,
                }
            )
        write_progress(frame_idx)

    # Fallback save_dir if not set
    if save_dir is None:
        base = (project or "runs")
        save_dir = os.path.join(base, name)
        os.makedirs(save_dir, exist_ok=True)

    results_json = os.path.join(save_dir, "results.json")
    # Build summary: counts per class and unique IDs per class
    summary: Dict[str, Dict[str, int]] = {}
    class_to_ids: Dict[str, set] = {}
    for t in tracks:
        cls_name = str(t.get("class_name"))
        summary.setdefault(cls_name, {"detections": 0, "unique_ids": 0})
        summary[cls_name]["detections"] += 1
        if t.get("id") is not None:
            class_to_ids.setdefault(cls_name, set()).add(int(t["id"]))
    for cls_name, ids in class_to_ids.items():
        summary.setdefault(cls_name, {"detections": 0, "unique_ids": 0})
        summary[cls_name]["unique_ids"] = len(ids)
    try:
        with open(results_json, "w", encoding="utf-8") as f:
            json.dump({"tracks": tracks, "summary": summary}, f, ensure_ascii=False)
    except Exception:
        pass

    # Try find a preview media file
    preview_file = None
    try:
        for ext in (".mp4", ".avi", ".mov", ".mkv", ".gif", ".jpg", ".png"):
            cand = [f for f in os.listdir(save_dir) if f.lower().endswith(ext)]
            if cand:
                preview_file = os.path.join(save_dir, cand[0])
                break
    except Exception:
        preview_file = None

    # Write 100% on completion
    write_progress(total_frames or len(tracks))

    return {
        "save_dir": save_dir,
        "results_json": results_json,
        "tracks": tracks,
        "summary": summary,
        "preview_file": preview_file,
        "progress_path": progress_path,
    }


