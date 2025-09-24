import argparse
from pathlib import Path
from loguru import logger
from ultralytics import YOLO
import json


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True, type=str)
    p.add_argument("--weights", required=True, type=str)
    p.add_argument("--project", default="runs/seg_eval", type=str)
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.project)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading model for evaluation")
    model = YOLO(args.weights)
    metrics = model.val(data=args.data, project=args.project, name="exp", plots=True)
    metrics_json = out_dir / "results.json"
    try:
        mdict = metrics.results_dict
    except Exception:
        mdict = {}
    metrics_json.write_text(json.dumps(mdict, indent=2))
    logger.info(f"Evaluation complete. Metrics saved to {metrics_json}")


if __name__ == "__main__":
    main()
