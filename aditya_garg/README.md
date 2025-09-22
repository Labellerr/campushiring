Vehicle Tracking Project – Aditya Garg

This project demonstrates **real-time vehicle tracking** using **YOLO + ByteTrack** and exports the tracking results to CSV. Users can also view a demo video of the tracked vehicles.

Project Overview

- Detects vehicles in a video (cars, trucks, etc.)
- Tracks vehicles across frames using ByteTrack
- Exports tracking results to `vehicle_count.csv`
- Provides a video with bounding boxes and IDs

Folder Structure
aditya_garg/
├── README.md
├── export_vehicle_csv.py
├── vehicle_count.csv
├── runs/segment/track4/
│ ├── results.json
│ └── vehicle-counting-result.avi
└── report/object_tracking_rep

 How to Run

1. Clone the repository:
```bash
git clone <repo-url>
2.Activate environment:
conda activate yolo-bytetrack
3.Run tracking (YOLO + ByteTrack):
python track_vehicle.py  # Your tracking script
4.Export to CSV:
python export_vehicle_csv.py
Demo
Video demo: runs/segment/track4/vehicle-counting-result.avi
CSV output: vehicle_count.csv
Report
PDF report: report/object_tracking_report.pdf
Contains journey, challenges, resolutions, evaluation metrics, and summary
Requirements
Python 3.8+
PyTorch
OpenCV
YOLOv8 / ByteTrack




