from pathlib import Path
from typing import Dict, Any

# Placeholder interfaces for Labellerr SDK integration.
# Replace with actual Labellerr SDK calls when available.


def download_labellerr_project(project_id: str, output_dir: str | Path) -> Path:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    # TODO: Integrate Labellerr SDK download here.
    # For now, assume dataset is prepared manually.
    return out


def ensure_yolo_seg_structure(dataset_dir: str | Path, classes: list[str]) -> Dict[str, Any]:
    base = Path(dataset_dir)
    (base / "images" / "train").mkdir(parents=True, exist_ok=True)
    (base / "images" / "val").mkdir(parents=True, exist_ok=True)
    (base / "images" / "test").mkdir(parents=True, exist_ok=True)
    (base / "labels" / "train").mkdir(parents=True, exist_ok=True)
    (base / "labels" / "val").mkdir(parents=True, exist_ok=True)
    (base / "labels" / "test").mkdir(parents=True, exist_ok=True)
    data_yaml = base / "data.yaml"
    if not data_yaml.exists():
        data_yaml.write_text(
            "\n".join(
                [
                    f"path: {base.as_posix()}",
                    "train: images/train",
                    "val: images/val",
                    "test: images/test",
                    f"names: {classes}",
                ]
            )
        )
    return {"base": base, "data_yaml": data_yaml}


__all__ = ["download_labellerr_project", "ensure_yolo_seg_structure"]
