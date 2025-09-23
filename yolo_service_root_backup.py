import os
import threading
from typing import Any, Dict, List, Optional


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


