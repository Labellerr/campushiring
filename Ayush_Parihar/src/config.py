from dataclasses import dataclass
from pathlib import Path


@dataclass
class TrainConfig:
	data_yaml: str
	epochs: int = 100
	img_size: int = 640
	batch: int = 8
	workers: int = 2
	project: str = "runs/seg_train"
	model: str = "yolov8s-seg.pt"

	def to_args(self) -> dict:
		return {
			"data": self.data_yaml,
			"epochs": self.epochs,
			"imgsz": self.img_size,
			"batch": self.batch,
			"workers": self.workers,
			"project": self.project,
			"model": self.model,
		}


BASE_DIR = Path(__file__).resolve().parent.parent
DATASETS_DIR = BASE_DIR / "datasets"
RUNS_DIR = BASE_DIR / "runs"
