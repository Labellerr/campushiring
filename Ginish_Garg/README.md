# PF Assignment — Seg + Tracking
Quickstart:
1) Open notebooks/02_train_yolov8_seg_colab.ipynb in Colab (GPU).
2) Put train/val/test images under data/yolo_data/images/* and COCO or YOLO labels under data/yolo_data/labels/*.
3) Train, run inference (saves predictions to /predictions/results.json).
4) Upload predictions to Labellerr test project using src/labellerr_upload.py.
5) Optional: run ByteTrack via src/bytetrack_integration.py for video tracking demo.
