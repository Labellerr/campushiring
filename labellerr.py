import os
import sys
import glob
import shutil
from typing import Dict, List, Tuple, Optional


def load_yaml(yaml_path: str) -> Dict:
    """Load a minimal YOLO-style YAML file with a safe fallback if PyYAML isn't installed."""
    try:
        import yaml  # type: ignore
        with open(yaml_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception:
        # Minimal fallback parser for simple key: value YAML (single-level plus lists)
        data: Dict[str, object] = {}
        key_order = ["path", "train", "val", "test", "nc", "names"]
        with open(yaml_path, "r", encoding="utf-8") as f:
            lines = [ln.strip() for ln in f.readlines() if ln.strip() and not ln.strip().startswith("#")]
        i = 0
        while i < len(lines):
            ln = lines[i]
            if ":" in ln:
                key, value = ln.split(":", 1)
                key = key.strip()
                value = value.strip()
                if value == "":
                    if key == "names":
                        names: List[str] = []
                        i += 1
                        while i < len(lines) and lines[i].startswith("-"):
                            names.append(lines[i][1:].strip().strip("'\"") )
                            i += 1
                        data[key] = names
                        continue
                else:
                    if value.isdigit():
                        data[key] = int(value)
                    elif value.lower() in {"true", "false"}:
                        data[key] = value.lower() == "true"
                    elif value.startswith("[") and value.endswith("]"):
                        inner = value[1:-1].strip()
                        data[key] = [v.strip().strip("'\"") for v in inner.split(",") if v.strip()]
                    else:
                        data[key] = value.strip("'\"")
            i += 1
        return {k: data.get(k) for k in key_order if k in data} | {k: v for k, v in data.items() if k not in key_order}


def resolve_path(base_dir: str, p: Optional[str]) -> Optional[str]:
    if p is None:
        return None
    p = os.path.expanduser(os.path.expandvars(p))
    if os.path.isabs(p):
        return p
    return os.path.normpath(os.path.join(base_dir, p))


def detect_images_dir(split_path: str) -> str:
    if os.path.isdir(split_path):
        base_name = os.path.basename(split_path).lower()
        if base_name == "images":
            return split_path
        candidate = os.path.join(split_path, "images")
        if os.path.isdir(candidate):
            return candidate
    return split_path


def list_images(images_dir: str) -> List[str]:
    exts = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff"]
    files: List[str] = []
    if os.path.isdir(images_dir):
        for ext in exts:
            files.extend(glob.glob(os.path.join(images_dir, "**", ext), recursive=True))
    else:
        if any(ch in images_dir for ch in ["*", "?", "["]):
            for ext in exts:
                files.extend(glob.glob(os.path.join(images_dir, ext) if os.path.isdir(images_dir) else images_dir, recursive=True))
        else:
            files = []
    return sorted(files)


def image_to_label_path(image_path: str) -> str:
    parts = image_path.replace("\\", "/").split("/")
    try:
        idx = len(parts) - 1 - parts[::-1].index("images")
        parts[idx] = "labels"
        label_dir_swapped = "/".join(parts)
    except ValueError:
        label_dir_swapped = image_path
    base, _ = os.path.splitext(label_dir_swapped)
    return base + ".txt"


def validate_pairs(images: List[str]) -> Tuple[int, int, List[str]]:
    ok = 0
    missing = 0
    missing_list: List[str] = []
    for img in images:
        lbl = image_to_label_path(img)
        if os.path.exists(lbl):
            ok += 1
        else:
            missing += 1
            missing_list.append(img)
    return ok, missing, missing_list


def parse_label_line(line: str) -> Optional[int]:
    line = line.strip()
    if not line:
        return None
    parts = line.split()
    try:
        return int(parts[0])
    except Exception:
        return None


def compute_class_distribution(images: List[str]) -> Dict[int, int]:
    counts: Dict[int, int] = {}
    for img in images:
        lbl_path = image_to_label_path(img)
        if not os.path.exists(lbl_path):
            continue
        try:
            with open(lbl_path, "r", encoding="utf-8") as f:
                for ln in f:
                    cid = parse_label_line(ln)
                    if cid is None:
                        continue
                    counts[cid] = counts.get(cid, 0) + 1
        except Exception:
            continue
    return counts


def draw_previews(images: List[str], out_dir: str, names: List[str], num: int = 3) -> List[str]:
    try:
        from PIL import Image, ImageDraw  # type: ignore
    except Exception:
        print("Pillow (PIL) not installed; skipping previews.")
        return []

    os.makedirs(out_dir, exist_ok=True)
    saved: List[str] = []

    def yolo_to_xyxy(w: int, h: int, cls: int, x: float, y: float, bw: float, bh: float) -> Tuple[int, int, int, int]:
        cx = x * w
        cy = y * h
        box_w = bw * w
        box_h = bh * h
        x1 = int(max(0, cx - box_w / 2))
        y1 = int(max(0, cy - box_h / 2))
        x2 = int(min(w - 1, cx + box_w / 2))
        y2 = int(min(h - 1, cy + box_h / 2))
        return x1, y1, x2, y2

    for img_path in images[:num]:
        lbl_path = image_to_label_path(img_path)
        try:
            im = Image.open(img_path).convert("RGB")
        except Exception:
            continue
        draw = ImageDraw.Draw(im)
        W, H = im.size
        try:
            with open(lbl_path, "r", encoding="utf-8") as f:
                for ln in f:
                    parts = ln.strip().split()
                    if len(parts) < 5:
                        continue
                    try:
                        cls_id = int(parts[0])
                    except Exception:
                        continue
                    if len(parts) in (5, 6):
                        x, y, bw, bh = map(float, parts[1:5])
                        x1, y1, x2, y2 = yolo_to_xyxy(W, H, cls_id, x, y, bw, bh)
                        color = (0, 255, 0) if cls_id % 2 == 0 else (255, 0, 0)
                        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
                        label = str(cls_id)
                        draw.text((x1 + 2, y1 + 2), label, fill=color)
                    else:
                        # segmentation polygon (x1 y1 x2 y2 ... normalized)
                        pts = parts[1:]
                        if len(pts) % 2 != 0:
                            continue
                        xy = []
                        for i in range(0, len(pts), 2):
                            try:
                                px = float(pts[i]) * W
                                py = float(pts[i+1]) * H
                                xy.append((px, py))
                            except Exception:
                                xy = []
                                break
                        if xy:
                            color = (0, 255, 0) if cls_id % 2 == 0 else (255, 0, 0)
                            draw.polygon(xy, outline=color)
                            draw.text(xy[0], str(cls_id), fill=color)
        except FileNotFoundError:
            pass
        out_path = os.path.join(out_dir, os.path.basename(img_path))
        try:
            im.save(out_path)
            saved.append(out_path)
        except Exception:
            continue

    return saved


def compute_image_quality(images: List[str]) -> Dict[str, object]:
    stats: Dict[str, object] = {
        "count": len(images),
        "width_min": None,
        "width_max": None,
        "height_min": None,
        "height_max": None,
        "width_mean": None,
        "height_mean": None,
        "file_size_min": None,
        "file_size_max": None,
        "file_size_mean": None,
        "common_resolutions": {},
        "channels": {},
        "aspect_ratio_mode": None,
        "aspect_ratio_within_5pct": None,
    }

    try:
        from PIL import Image  # type: ignore
    except Exception:
        print("Pillow (PIL) not installed; skipping image quality stats.")
        return stats

    if not images:
        return stats

    widths: List[int] = []
    heights: List[int] = []
    sizes: List[int] = []
    res_counts: Dict[str, int] = {}
    channel_counts: Dict[str, int] = {}
    aspect_ratios: List[float] = []

    for p in images:
        try:
            sz = os.path.getsize(p)
            with Image.open(p) as im:
                w, h = im.size
                mode = im.mode
        except Exception:
            continue
        widths.append(w)
        heights.append(h)
        sizes.append(sz)
        key = f"{w}x{h}"
        res_counts[key] = res_counts.get(key, 0) + 1
        channel_counts[mode] = channel_counts.get(mode, 0) + 1
        if h > 0:
            aspect_ratios.append(round(w / h, 4))

    if not widths:
        return stats

    def mean(vals: List[float]) -> float:
        return round(sum(vals) / max(1, len(vals)), 2)

    stats["width_min"] = int(min(widths))
    stats["width_max"] = int(max(widths))
    stats["height_min"] = int(min(heights))
    stats["height_max"] = int(max(heights))
    stats["width_mean"] = mean([float(x) for x in widths])
    stats["height_mean"] = mean([float(x) for x in heights])
    stats["file_size_min"] = int(min(sizes))
    stats["file_size_max"] = int(max(sizes))
    stats["file_size_mean"] = mean([float(x) for x in sizes])

    top_res = dict(sorted(res_counts.items(), key=lambda kv: kv[1], reverse=True)[:5])
    stats["common_resolutions"] = top_res

    stats["channels"] = dict(sorted(channel_counts.items(), key=lambda kv: kv[1], reverse=True))

    if aspect_ratios:
        bins: Dict[float, int] = {}
        for ar in aspect_ratios:
            b = round(ar, 3)
            bins[b] = bins.get(b, 0) + 1
        dominant_ar = max(bins.items(), key=lambda kv: kv[1])[0]
        within = sum(1 for ar in aspect_ratios if abs(ar - dominant_ar) / dominant_ar <= 0.05)
        stats["aspect_ratio_mode"] = dominant_ar
        stats["aspect_ratio_within_5pct"] = f"{round(100.0 * within / len(aspect_ratios), 2)}%"

    return stats


def labels_dir_from_images_dir(images_dir: str) -> str:
    if images_dir.replace("\\", "/").endswith("/images"):
        return images_dir[:-len("images")] + "labels"
    parent = os.path.dirname(images_dir)
    return os.path.join(parent, "labels")


def convert_bbox_labels_to_segmentation(labels_dir: str) -> Tuple[int, int, int]:
    converted = 0
    skipped_poly = 0
    files = 0

    if not os.path.isdir(labels_dir):
        return converted, skipped_poly, files

    backup_dir = labels_dir.rstrip("/\\") + "_bbox"
    if not os.path.exists(backup_dir):
        shutil.copytree(labels_dir, backup_dir)
        print(f"Backed up labels to: {backup_dir}")

    for path in glob.glob(os.path.join(labels_dir, "**", "*.txt"), recursive=True):
        files += 1
        try:
            with open(path, "r", encoding="utf-8") as f:
                lines = [ln.strip() for ln in f.readlines() if ln.strip()]
        except Exception:
            continue
        new_lines: List[str] = []
        changed = False
        for ln in lines:
            parts = ln.split()
            if len(parts) < 5:
                new_lines.append(ln)
                continue
            try:
                cls_id = int(parts[0])
            except Exception:
                new_lines.append(ln)
                continue
            if len(parts) > 6:
                # Already polygon; keep as-is
                new_lines.append(ln)
                skipped_poly += 1
                continue
            # bbox -> polygon
            try:
                x, y, w, h = map(float, parts[1:5])
            except Exception:
                new_lines.append(ln)
                continue
            x1 = x - w / 2
            y1 = y - h / 2
            x2 = x + w / 2
            y2 = y + h / 2
            # Four corner polygon normalized: top-left, top-right, bottom-right, bottom-left
            poly = [
                f"{cls_id}",
                f"{max(0.0, min(1.0, x1))}", f"{max(0.0, min(1.0, y1))}",
                f"{max(0.0, min(1.0, x2))}", f"{max(0.0, min(1.0, y1))}",
                f"{max(0.0, min(1.0, x2))}", f"{max(0.0, min(1.0, y2))}",
                f"{max(0.0, min(1.0, x1))}", f"{max(0.0, min(1.0, y2))}",
            ]
            new_lines.append(" ".join(poly))
            changed = True
        if changed:
            try:
                with open(path, "w", encoding="utf-8") as f:
                    f.write("\n".join(new_lines) + "\n")
                converted += 1
            except Exception:
                pass
    return converted, skipped_poly, files


def summarize_split(name: str, split_value: Optional[str], base_dir: str, yaml_dir: str) -> Dict[str, object]:
    if split_value is None:
        return {"split": name, "defined": False}

    split_path = resolve_path(yaml_dir, split_value)
    images_dir = detect_images_dir(split_path)

    if not os.path.exists(images_dir):
        alt = resolve_path(base_dir, split_value.lstrip("./").lstrip("../"))
        alt_images_dir = detect_images_dir(alt)
        if os.path.exists(alt_images_dir):
            images_dir = alt_images_dir

    images = list_images(images_dir)
    ok, missing, missing_list = validate_pairs(images)
    class_counts = compute_class_distribution(images)
    img_quality = compute_image_quality(images)

    return {
        "split": name,
        "defined": True,
        "images_dir": images_dir,
        "labels_dir": labels_dir_from_images_dir(images_dir),
        "num_images": len(images),
        "labels_ok": ok,
        "labels_missing": missing,
        "class_counts": class_counts,
        "image_quality": img_quality,
        "sample_missing": missing_list[:10],
        "images": images,
    }


def run_conversion_over_summaries(summaries: List[Dict[str, object]]) -> None:
    total_converted = 0
    total_skipped_poly = 0
    total_files = 0
    for s in summaries:
        if not s.get("defined"):
            continue
        labels_dir = s.get("labels_dir")
        if not isinstance(labels_dir, str):
            continue
        conv, skipped, files = convert_bbox_labels_to_segmentation(labels_dir)
        total_converted += conv
        total_skipped_poly += skipped
        total_files += files
        print(f"Converted in {labels_dir}: changed={conv}, already_polygon={skipped}, files={files}")
    print(f"Total: changed={total_converted}, already_polygon={total_skipped_poly}, files={total_files}")


def main(dataset_dir: Optional[str] = None) -> None:
    dataset_root = dataset_dir or r"C:\\Users\\kiran\\Downloads\\Detecting cars and pedestrians.v1i.yolov8"
    yaml_path = os.path.join(dataset_root, "data.yaml")
    if not os.path.exists(yaml_path):
        print(f"data.yaml not found at: {yaml_path}")
        sys.exit(1)

    data = load_yaml(yaml_path)

    yaml_dir = os.path.dirname(yaml_path)
    base_path = dataset_root

    names = data.get("names") or []
    if isinstance(names, dict):
        names = [names[k] for k in sorted(names.keys())]

    train_val = str(data.get("train")) if data.get("train") is not None else None
    val_val = str(data.get("val")) if data.get("val") is not None else None
    test_val = str(data.get("test")) if data.get("test") is not None else None

    print("Loaded dataset YAML:")
    print(f"  root: {dataset_root}")
    print(f"  yaml_dir: {yaml_dir}")
    print(f"  nc: {data.get('nc', '(unknown)')}")
    print(f"  names: {names}")

    summaries = [
        summarize_split("train", train_val, base_path, yaml_dir),
        summarize_split("val", val_val, base_path, yaml_dir),
        summarize_split("test", test_val, base_path, yaml_dir),
    ]

    convert_flag = any(arg.lower() in ("--convert-seg", "--seg", "--to-seg") for arg in sys.argv[1:])

    print("\nSplit summary:")
    for s in summaries:
        if not s.get("defined"):
            print(f"- {s['split']}: not defined in YAML")
            continue
        print(f"- {s['split']}:")
        print(f"    images_dir: {s['images_dir']}")
        print(f"    num_images: {s['num_images']}")
        print(f"    labels_ok: {s['labels_ok']}")
        print(f"    labels_missing: {s['labels_missing']}")
        class_counts = s.get("class_counts", {})
        if class_counts:
            print("    class_counts:")
            for k in sorted(class_counts.keys()):
                print(f"      {k}: {class_counts[k]}")
        iq = s.get("image_quality", {}) or {}
        if iq:
            print("    image_quality:")
            print(f"      resolution_width: min={iq.get('width_min')} mean={iq.get('width_mean')} max={iq.get('width_max')}")
            print(f"      resolution_height: min={iq.get('height_min')} mean={iq.get('height_mean')} max={iq.get('height_max')}")
            print(f"      file_size_bytes: min={iq.get('file_size_min')} mean={iq.get('file_size_mean')} max={iq.get('file_size_max')}")
            common_res = iq.get("common_resolutions", {}) or {}
            if common_res:
                print("      common_resolutions:")
                for res, cnt in common_res.items():
                    print(f"        {res}: {cnt}")
            channels = iq.get("channels", {}) or {}
            if channels:
                print("      channels:")
                for mode, cnt in channels.items():
                    print(f"        {mode}: {cnt}")
            if iq.get("aspect_ratio_mode") is not None:
                print(f"      aspect_ratio_mode: {iq.get('aspect_ratio_mode')}")
                print(f"      within_5pct_of_mode: {iq.get('aspect_ratio_within_5pct')}")

    if convert_flag:
        print("\nConverting bbox labels to segmentation polygons (backup will be created)...")
        run_conversion_over_summaries(summaries)

    # Previews
    from_previews = any(arg.lower() == "--no-previews" for arg in sys.argv[1:])
    out_root = os.path.join(dataset_root, "previews")
    if not from_previews:
        os.makedirs(out_root, exist_ok=True)
        for s in summaries:
            if not s.get("defined"):
                continue
            split_name = s["split"]  # type: ignore
            split_out = os.path.join(out_root, str(split_name))
            saved = draw_previews(s.get("images", [])[:10], split_out, names, num=3)  # type: ignore
            if saved:
                print(f"Saved {len(saved)} previews for {split_name} to: {split_out}")
            else:
                print(f"No previews saved for {split_name} (Pillow missing or no images)")


if __name__ == "__main__":
    user_arg = None
    # Optional dataset root as first non-flag arg
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    if args:
        user_arg = args[0]
    main(user_arg)
