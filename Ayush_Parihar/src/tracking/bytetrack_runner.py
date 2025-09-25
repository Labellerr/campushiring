from pathlib import Path
from typing import Any, Dict
from loguru import logger
from ultralytics import YOLO


DEFAULT_TRACKER_CFG = {
    "tracker": "botsort.yaml",  # Ultralytics supports bytetrack.yaml and botsort.yaml
}


def track_video(
    weights: str,
    source_video: str | Path,
    save_dir: str | Path = "runs/track",
    imgsz: int = 640,
    conf: float = 0.25,
    iou: float = 0.45,
    tracker_cfg: Dict[str, Any] | None = None,
) -> Path:
    out = Path(save_dir)
    out.mkdir(parents=True, exist_ok=True)
    model = YOLO(weights)
    logger.info("Starting tracking on video")
    cfg = dict(DEFAULT_TRACKER_CFG)
    if tracker_cfg:
        cfg.update(tracker_cfg)
    results = model.track(
        source=str(source_video),
        stream=False,
        save=True,
        save_txt=True,
        save_conf=True,
        project=str(out),
        imgsz=imgsz,
        conf=conf,
        iou=iou,
        tracker=cfg["tracker"],
    )
    # Ultralytics returns a list; results[0].save_dir contains the run folder
    try:
        save_dir_res = Path(results[0].save_dir)
    except Exception:
        save_dir_res = out
    logger.info(f"Tracking complete. Artifacts in {save_dir_res}")
    return save_dir_res


__all__ = ["track_video"]
