import argparse
import json
from pathlib import Path
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--metrics", required=True, type=str)
    p.add_argument("--output", default="report.pdf", type=str)
    return p.parse_args()


def make_pdf(metrics_path: Path, output_path: Path) -> None:
    data = {}
    if metrics_path.exists():
        data = json.loads(metrics_path.read_text())
    c = canvas.Canvas(str(output_path), pagesize=A4)
    width, height = A4
    y = height - 50
    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, y, "Vehicle & Pedestrian Segmentation + Tracking Report")
    y -= 30
    c.setFont("Helvetica", 12)
    c.drawString(50, y, f"Metrics source: {metrics_path}")
    y -= 20

    for k, v in sorted(data.items()):
        if y < 80:
            c.showPage()
            y = height - 50
        c.drawString(50, y, f"{k}: {v}")
        y -= 18

    c.showPage()
    c.save()


def main():
    args = parse_args()
    make_pdf(Path(args.metrics), Path(args.output))


if __name__ == "__main__":
    main()
