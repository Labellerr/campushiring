import argparse
from pathlib import Path
from loguru import logger
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True, type=str)
    p.add_argument("--epochs", default=100, type=int)
    p.add_argument("--img", default=640, type=int)
    p.add_argument("--batch", default=8, type=int)
    p.add_argument("--workers", default=2, type=int)
    p.add_argument("--project", default="runs/seg_train", type=str)
    p.add_argument("--model", default="yolov8s-seg.pt", type=str)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.project)
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Starting YOLOv8-Seg training")
    model = YOLO(args.model)
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.img,
        batch=args.batch,
        workers=args.workers,
        project=args.project,
        name="exp",
        plots=True,
    )
    # Save metrics JSON
    results_path = out_dir / "results.json"
    try:
        metrics = results.results_dict
    except Exception:
        metrics = {}
    results_path.write_text(__import__("json").dumps(metrics, indent=2))
    logger.info(f"Training complete. Metrics saved to {results_path}")


if __name__ == "__main__":
    main()
