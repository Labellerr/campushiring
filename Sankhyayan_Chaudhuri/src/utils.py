import os
import json
import random
from pathlib import Path
from typing import List, Dict, Any, Optional

import numpy as np
import cv2
import pandas as pd
from tqdm import tqdm

try:
    from labellerr.client import LabellerrClient  # type: ignore
    from labellerr.exceptions import LabellerrError  # type: ignore
except Exception:  # SDK optional import
    LabellerrClient = Any  # type: ignore
    LabellerrError = Exception  # type: ignore


COCO_PERSON_ID = 0
COCO_VEHICLE_IDS = [2, 3, 5, 7]  # car, motorcycle, bus, truck
TARGET_CLASS_IDS = [COCO_PERSON_ID] + COCO_VEHICLE_IDS


def ensure_dir(p: str | Path) -> None:
    Path(p).mkdir(parents=True, exist_ok=True)


def save_json(path: str | Path, data: Any):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def load_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def split_list(items: List[Any], ratios=(0.7, 0.2, 0.1), seed: int = 42):
    random.seed(seed)
    n = len(items)
    idx = list(range(n))
    random.shuffle(idx)
    n_train = int(ratios[0] * n)
    n_val = int(ratios[1] * n)
    train_idx = idx[:n_train]
    val_idx = idx[n_train:n_train + n_val]
    test_idx = idx[n_train + n_val:]
    return [items[i] for i in train_idx], [items[i] for i in val_idx], [items[i] for i in test_idx]


def init_labellerr_from_env() -> Optional[Any]:
    if LabellerrClient is None:
        return None
    api_key = os.getenv("LABELLERR_API_KEY")
    api_secret = os.getenv("LABELLERR_API_SECRET")
    if not api_key or not api_secret:
        return None
    client = LabellerrClient(api_key, api_secret)
    return client


def yolo_seg_data_yaml(root: Path, names: Optional[Dict[int, str]] = None) -> Dict[str, Any]:
    names = names or {0: "person", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
    return {
        "path": str(root),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": names,
        "nc": len(names),
    }


def collect_images_from_coco(sample_count: int = 200, out_dir: str | Path = "data/raw") -> List[str]:
    """Download a small set of COCO 2017 val images via OpenCV sample URLs.
    This is a lightweight placeholder; users can replace with their sources.
    """
    ensure_dir(out_dir)
    # A few public sample images with people/vehicles (not full COCO API usage)
    urls = [
        "https://ultralytics.com/images/zidane.jpg",
        "https://ultralytics.com/images/bus.jpg",
        "https://ultralytics.com/images/bus-stops.jpg",
        "https://ultralytics.com/images/cars.jpg",
        "https://ultralytics.com/images/motorbike.jpg",
        "https://ultralytics.com/images/traffic.jpg",
    ]
    paths = []
    for i in range(sample_count):
        url = urls[i % len(urls)]
        p = Path(out_dir) / f"img_{i:04d}.jpg"
        try:
            import requests
            r = requests.get(url, timeout=20)
            if r.status_code == 200:
                with open(p, "wb") as f:
                    f.write(r.content)
                paths.append(str(p))
        except Exception:
            continue
    return paths