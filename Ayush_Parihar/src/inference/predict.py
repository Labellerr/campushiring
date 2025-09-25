import argparse
from pathlib import Path
from loguru import logger
from ultralytics import YOLO


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", required=True, type=str)
    p.add_argument("--source", required=True, type=str, help="images dir or file")
    p.add_argument("--save-dir", default="runs/preds", type=str)
    p.add_argument("--img", default=640, type=int)
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.save_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading model for prediction")
    model = YOLO(args.weights)
    model.predict(
        source=args.source,
        imgsz=args.img,
        save=True,
        save_txt=True,
        save_conf=True,
        save_dir=out_dir.as_posix(),
        verbose=False,
        stream=False,
    )
    logger.info(f"Predictions saved to {out_dir}")


if __name__ == "__main__":
    main()
