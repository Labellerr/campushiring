# Labellerr + YOLOv8-Seg + ByteTrack: Vehicles & Pedestrians

End-to-end workflow to collect data, annotate (via Labellerr SDK), train YOLOv8 segmentation, evaluate, run inference, and track with ByteTrack. Includes a minimal Streamlit app for video tracking.

Quick start (Windows PowerShell):

```
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Run the Streamlit app:

```
streamlit run app/streamlit_app.py
```

Notebook: see `notebooks/labellerr_yolov8_seg_workflow.ipynb` (Colab-friendly). It guides through data collection, Labellerr integration, training, evaluation, inference, and exporting.

Folders
- `data/` — raw and prepared datasets
- `runs/` — YOLO training outputs (created by Ultralytics)
- `reports/` — saved metrics, plots, and PDF summary
- `app/` — minimal Streamlit web app for ByteTrack tracking
- `src/` — helper utilities

Notes
- Labellerr SDK requires credentials. Set environment variables: `LABELLERR_API_KEY`, `LABELLERR_API_SECRET`, `LABELLERR_CLIENT_ID`.
- If you don’t have annotations yet, the notebook can fall back to COCO masks and convert them to YOLO-seg format for experimentation.
