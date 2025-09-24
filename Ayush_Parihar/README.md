## Vehicle & Pedestrian Segmentation + Tracking (YOLOv8-Seg + ByteTrack)

End-to-end pipeline to collect data, annotate with Labellerr, train YOLOv8-Seg, evaluate, and run multi-object tracking on videos via ByteTrack. Includes a Streamlit/Gradio web app, a Colab notebook, and a simple PDF report generator.

### Features
- **Data pipeline**: collect ~200 images, annotate 100 via Labellerr, export YOLO segmentation format
- **Training**: YOLOv8-Seg (~100 epochs), logs IoU/mAP, PR curve, confusion matrix
- **Evaluation**: run `val`, export metrics JSON and plots
- **Inference**: predict on test images and save outputs
- **Tracking**: ByteTrack/BOTSort via Ultralytics tracker API for uploaded videos
- **Web app**: Streamlit (and Gradio) to upload a video → track → preview → download `results.json`
- **Reproducible**: `requirements.txt`, Colab notebook, modular code

### Stack
- Python 3.10+
- Ultralytics YOLOv8-Seg
- ByteTrack/BOTSort (via Ultralytics tracker integration)
- Streamlit / Gradio
- Labellerr (annotation)

### Project Structure
```
.
├─ src/
│  ├─ config.py
│  ├─ utils/
│  │  └─ logger.py
│  ├─ data/
│  │  ├─ prepare_dataset.py
│  │  └─ labellerr_utils.py
│  ├─ training/
│  │  ├─ train.py
│  │  └─ evaluate.py
│  ├─ inference/
│  │  └─ predict.py
│  ├─ tracking/
│  │  └─ bytetrack_runner.py
│  ├─ webapp/
│  │  ├─ streamlit_app.py
│  │  └─ gradio_app.py
│  └─ report/
│     └─ generate_report.py
├─ notebooks/
│  └─ train_yolov8_seg_colab.ipynb
├─ requirements.txt
├─ sources.md
└─ README.md
```

### Quickstart
```bash
python -m venv .venv
. .venv/Scripts/activate  # Windows PowerShell
pip install -r requirements.txt
```

### Dataset Preparation
1. Collect ~200 raw images (dashcam, public datasets). Place them in a folder, e.g. `raw_images/`.
2. Create YOLO Seg structure and split train/val/test:
   ```bash
   python -m src.data.prepare_dataset --input raw_images --out datasets/your_dataset --classes person car --total 200
   ```
3. Annotate ~100 images with polygons in Labellerr. Export YOLO segmentation format and copy `labels/{train,val,test}` into `datasets/your_dataset/labels/`.
4. Ensure you have a `data.yaml` under `datasets/your_dataset/` similar to:
   ```yaml
   path: datasets/your_dataset
   train: images/train
   val: images/val
   test: images/test
   names: [person, car]
   ```

### Training
```bash
python -m src.training.train --data datasets/your_dataset/data.yaml --epochs 100 --img 640 --project runs/seg_train
```
- Outputs: weights, PR curves, confusion matrix, and `runs/seg_train/results.json`.

### Evaluation
```bash
python -m src.training.evaluate --data datasets/your_dataset/data.yaml --weights runs/seg_train/weights/best.pt --project runs/seg_eval
```
- Outputs metrics JSON and plots under `runs/seg_eval/`.

### Inference (Images)
```bash
python -m src.inference.predict --weights runs/seg_train/weights/best.pt --source datasets/your_dataset/images/test --save-dir runs/preds
```
- Saves annotated images and label files in `runs/preds/`.

### Tracking Web App (Streamlit)
```bash
python -m streamlit run src/webapp/streamlit_app.py --server.port 8501
```
- Upload a video, choose weights (default `yolov8s-seg.pt`), and run tracking.
- Preview the output video and download `results.json`.
- Artifacts are saved under `runs/webapp_results/`.

### Alternative UI (Gradio)
```bash
python -m src.webapp.gradio_app
```

### Colab Notebook
- Open `notebooks/train_yolov8_seg_colab.ipynb` in Google Colab.
- Installs dependencies, runs training, evaluation, and exports metrics JSON.

### Report
```bash
python -m src.report.generate_report --metrics runs/seg_train/results.json --output report.pdf
```

### Troubleshooting
- If Streamlit shows a spinner only:
  - First run may download weights; allow time.
  - Check `runs/webapp_results/` for a `track...` folder and open the MP4.
  - Ensure the weights path is valid (or use default `yolov8s-seg.pt`).
- Torch/CUDA:
  - This setup works on CPU; for GPU, install a CUDA-enabled `torch` build from PyTorch docs.
- Path issues on Windows:
  - If CLI scripts aren’t found, run via `python -m <module>` as shown in commands above.

### Sources and Licenses
- See `sources.md` for suggested datasets and license notes.
- Code: MIT. Dataset licenses vary by source.

### Acknowledgements
- Ultralytics YOLOv8, ByteTrack, BOTSort, Labellerr.
