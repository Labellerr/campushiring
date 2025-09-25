from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse, FileResponse
from ultralytics import YOLO
import os
import json
from pathlib import Path
import shutil
from fastapi.staticfiles import StaticFiles

UPLOAD_FOLDER = "uploads"
RESULT_FOLDER = "results"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

app = FastAPI()

model = YOLO("../runs/segment/train3/weights/best.pt")

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def home():
    with open("templates/index.html", "r", encoding="utf-8") as f:
        return HTMLResponse(f.read())

def extract_results_to_json(results, output_path):
    all_results = []
    for frame_idx, result in enumerate(results):
        frame_data = {"frame": frame_idx, "detections": []}
        if result.boxes is not None:
            boxes = result.boxes.xyxy.cpu().numpy()
            confidences = result.boxes.conf.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy()
            class_names = result.names
            for i in range(len(boxes)):
                detection = {
                    "bbox": boxes[i].tolist(),
                    "confidence": float(confidences[i]),
                    "class_id": int(classes[i]),
                    "class_name": class_names[int(classes[i])]
                }
                if result.masks is not None and i < len(result.masks.xy):
                    detection["mask"] = result.masks.xy[i].tolist()
                frame_data["detections"].append(detection)
        all_results.append(frame_data)
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)

@app.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    try:
        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        with open(file_path, "wb") as f:
            f.write(await file.read())

        ext = Path(file.filename).suffix.lower()
        base_filename = Path(file.filename).stem
        json_filename = f"{base_filename}_results.json"
        json_path = os.path.join(RESULT_FOLDER, json_filename)

        results = model(file_path, save=True, show=False)
        extract_results_to_json(results, json_path)

        
        output_file_path = None
        output_dir = Path("runs/segment/predict")

        if ext in [".mp4", ".avi", ".mov", ".mkv"]:  
            for f in output_dir.iterdir():
                if f.suffix.lower() in [".mp4", ".avi", ".mov", ".mkv"]:
                    output_file_path = os.path.join(RESULT_FOLDER, f"{base_filename}_processed{f.suffix}")
                    shutil.move(str(f), output_file_path)
                    break

        elif ext in [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]: 
            for f in output_dir.iterdir():
                if f.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]:
                    output_file_path = os.path.join(RESULT_FOLDER, f"{base_filename}_processed{f.suffix}")
                    shutil.move(str(f), output_file_path)
                    break

        if output_dir.exists():
            shutil.rmtree(output_dir)

        if output_file_path and os.path.exists(json_path):
            return HTMLResponse(f"""
                <html>
                <head>
                    <link rel="stylesheet" href="/static/css/style.css">
                </head>
                <body>
                    <div class="container">
                        <h1>Processing Complete</h1>
                        <a href="/results/{Path(output_file_path).name}" class="btn" target="_blank">Download Processed File</a>
                        <a href="/results/{json_filename}" class="btn" target="_blank">Download JSON Result</a>
                        <br><br>
                        <a href="/" class="btn back-btn">Process Another File</a>
                    </div>
                    <script src="/static/js/main.js"></script>
                </body>
                </html>
            """)
        else:
            return HTMLResponse("Error: Could not find processed file or JSON.", status_code=500)
    except Exception as e:
        return HTMLResponse(f"Processing Error: {str(e)}", status_code=500)

@app.get("/results/{file_name}")
def get_results(file_name: str):
    file_path = os.path.join(RESULT_FOLDER, file_name)
    if os.path.exists(file_path):
        return FileResponse(file_path)
    return HTMLResponse("<h3>File Not Found</h3>")
