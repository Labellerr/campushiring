import argparse
import random
import shutil
from pathlib import Path
from typing import List

from .labellerr_utils import ensure_yolo_seg_structure


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def find_images(input_dir: Path, limit: int | None = None) -> List[Path]:
	files = []
	for p in input_dir.rglob("*"):
		if p.suffix.lower() in IMAGE_EXTS:
			files.append(p)
	if limit is not None:
		random.shuffle(files)
		files = files[:limit]
	return files


def copy_files(files: List[Path], dst_dir: Path) -> None:
	dst_dir.mkdir(parents=True, exist_ok=True)
	for f in files:
		shutil.copy2(f, dst_dir / f.name)


def prepare_dataset(input_images: Path, out_dir: Path, classes: List[str], total: int = 200, val_ratio: float = 0.2, test_ratio: float = 0.1) -> None:
	random.seed(42)
	imgs = find_images(input_images, limit=total)
	assert imgs, f"No images found in {input_images}"
	ensure = ensure_yolo_seg_structure(out_dir, classes)
	base = ensure["base"]
	# split
	n = len(imgs)
	n_test = int(n * test_ratio)
	n_val = int(n * val_ratio)
	n_train = n - n_val - n_test
	train = imgs[:n_train]
	val = imgs[n_train:n_train + n_val]
	test = imgs[n_train + n_val:]
	copy_files(train, base / "images" / "train")
	copy_files(val, base / "images" / "val")
	copy_files(test, base / "images" / "test")
	# Labels will be created after annotation export from Labellerr
	print(f"Prepared dataset at {base}. Train/Val/Test: {len(train)}/{len(val)}/{len(test)}")


def parse_args() -> argparse.Namespace:
	p = argparse.ArgumentParser()
	p.add_argument("--input", required=True, help="Folder with raw images")
	p.add_argument("--out", required=True, help="Output dataset root")
	p.add_argument("--classes", nargs="+", default=["person", "car"], help="Class names")
	p.add_argument("--total", type=int, default=200)
	p.add_argument("--val", type=float, default=0.2)
	p.add_argument("--test", type=float, default=0.1)
	return p.parse_args()


def main() -> None:
	args = parse_args()
	prepare_dataset(Path(args.input), Path(args.out), args.classes, args.total, args.val, args.test)


if __name__ == "__main__":
	main()

