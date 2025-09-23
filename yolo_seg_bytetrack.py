import argparse
import os
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="YOLOv8 Segmentation + ByteTrack tracking runner"
    )
    p.add_argument(
        "--model",
        type=str,
        default=r"C:\\Users\\kiran\\Downloads\\Detecting cars and pedestrians.v1i.yolov8\\runs\\yolov8n-seg-fast-1ep-cli\\weights\\best.pt",
        help="Path to YOLOv8 segmentation model (.pt)",
    )
    p.add_argument(
        "--source",
        type=str,
        default=r"C:\\Users\\kiran\\Downloads\\Detecting cars and pedestrians.v1i.yolov8\\test\\images",
        help="Video file, image, directory, or webcam index (e.g., 0)",
    )
    p.add_argument("--device", type=str, default="cpu", help="cpu or cuda device id, e.g., 0")
    p.add_argument("--imgsz", type=int, default=416, help="Inference image size")
    p.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    p.add_argument("--iou", type=float, default=0.5, help="NMS IoU threshold")
    p.add_argument("--classes", type=int, nargs="*", default=None, help="Filter by class indices")
    p.add_argument("--project", type=str, default=r"C:\\Users\\kiran\\Downloads\\Detecting cars and pedestrians.v1i.yolov8\\runs", help="Project dir for outputs")
    p.add_argument("--name", type=str, default="track-bytetrack", help="Run name for outputs")
    p.add_argument("--save", action="store_true", help="Save annotated video/images")
    p.add_argument("--show", action="store_true", help="Show GUI window during tracking")
    p.add_argument("--save_txt", action="store_true", help="Save results to .txt")
    p.add_argument("--line_thickness", type=int, default=2, help="Bounding box line thickness")
    p.add_argument("--persist", action="store_true", help="Persist track IDs across streams")
    p.add_argument("--half", action="store_true", help="Use half precision if supported")
    p.add_argument("--vid_stride", type=int, default=1, help="Video frame-rate stride")
    p.add_argument("--tracker", type=str, default="bytetrack.yaml", help="Tracker config (bytetrack.yaml)")
    return p


def main():
    args = build_parser().parse_args()

    # Import here to avoid import cost on help/parse
    from ultralytics import YOLO

    model_path = args.model
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    save_dir = Path(args.project) / args.name
    save_dir.parent.mkdir(parents=True, exist_ok=True)

    model = YOLO(model_path)

    results = model.track(
        source=args.source,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        classes=args.classes,
        show=args.show,
        save=args.save,
        save_txt=args.save_txt,
        project=args.project,
        name=args.name,
        tracker=args.tracker,
        line_thickness=args.line_thickness,
        vid_stride=args.vid_stride,
        stream=False,
        verbose=True,
        persist=args.persist,
        half=args.half,
    )

    # Print a tiny summary
    out_dir = str(save_dir)
    print(f"Tracking complete. Outputs (if saved) are under: {out_dir}")


if __name__ == "__main__":
    main()
