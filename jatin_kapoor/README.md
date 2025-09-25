# Labeller AI – YOLOv8‑Seg + ByteTrack (v2, self-contained)
## Setup
```bash
cd labeller-ai-assignment-v2
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```
If Torch fails on Mac:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

## Train
```bash
python src/train_yolov8_seg.py --data data/dataset.yaml --epochs 5 --img 640 --batch 2
# -> runs/segment/labeller_seg_v2/weights/best.pt
```

## Copy weights (optional)
```bash
mkdir -p models
cp runs/segment/labeller_seg_v2/weights/best.pt models/
```

## Evaluate
```bash
python src/evaluate.py --weights models/best.pt --data data/dataset.yaml
```

## Track + Export JSON
```bash
python src/track_and_export.py --weights models/best.pt --source samples/sample.mp4 --out_dir outputs --conf 0.25
# outputs/sample_tracked.mp4 + outputs/sample_results.json
```
## ✅ Fixes I made

1. Fixed a problem with YOLOv8 looking in the wrong place for the dataset by converting relative paths to full (absolute) paths.
2. Fixed missing folders (`val/` and `test/`) in the dataset — YOLO needs those folders to calculate evaluation results properly.

---

## 📊 Sample Results

- Evaluation metrics: see `reports/` folder (contains PR curve and confusion matrix)
- Tracked video: `outputs/sample_tracked.mp4`
- Exported JSON: `outputs/sample_results.json`

