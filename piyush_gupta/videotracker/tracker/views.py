import glob
import shutil
import os
import json
from django.conf import settings
from django.http import JsonResponse, HttpResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from ultralytics import YOLO

# Upload directory
UPLOAD_DIR = os.path.join(settings.BASE_DIR, "media/uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

@csrf_exempt
def video_upload_and_process(request):
    context = {}
    if request.method == "POST" and request.FILES.get("video"):
        video_file = request.FILES["video"]
        video_path = os.path.join(UPLOAD_DIR, video_file.name)

        # Save the uploaded video
        with open(video_path, "wb+") as dest:
            for chunk in video_file.chunks():
                dest.write(chunk)

        # Output JSON file
        results_path = os.path.splitext(video_path)[0] + "_results.json"

        # Run YOLO + ByteTrack
        run_yoloseg_bytetrack(video_path, results_path)

        context["results_ready"] = True
        context["json_filename"] = os.path.basename(results_path)
      
        output_dir = os.path.join(UPLOAD_DIR, "tracked_videos")
        video_files = glob.glob(os.path.join(output_dir, "*.mp4"))

        if video_files:
            saved_video_path = video_files[0]  # Get the first .avi file
            final_path = os.path.join(UPLOAD_DIR, "final_video_name.mp4")
            shutil.move(saved_video_path, final_path)
            print("Video moved to:", final_path)
            context["video_filename"] = os.path.basename(final_path) 
        else:
            print("No mp4 video file found in tracked_videos folder")

    return render(request, "video_upload.html", context)


def run_yoloseg_bytetrack(video_path, results_path):
    """
    YOLOv8 + ByteTrack tracking bypassing default.yaml
    """
    # Paths in your project
    model_path = "/Users/piyushgupta/Desktop/Projects/Labellar/campushiring/piyush_gupta/videotracker/tracker/model/best.pt"
    tracker_path = "/Users/piyushgupta/.pyenv/versions/3.10.16/lib/python3.10/site-packages/ultralytics/cfg/trackers/bytetrack.yml"

    # Load YOLO model
    model = YOLO(model_path)

    # Track video explicitly using your tracker YAML
    results = model.track(
    source=video_path,
    tracker=tracker_path,
    persist=True,
    save=True,           # Enable saving output video and results
    project=UPLOAD_DIR,  # Base directory to save results
    name="tracked_videos",  # Subfolder name under project directory
    exist_ok=True        # Overwrite if the folder already exists
    )
    
    tracked_results = []

    for frame_idx, result in enumerate(results):
        if hasattr(result, "boxes") and result.boxes is not None and len(result.boxes) > 0:
            boxes = getattr(result.boxes, 'xyxy', None)
            ids = getattr(result.boxes, 'id', None)
            classes = getattr(result.boxes, 'cls', None)

        # Convert to numpy arrays only if not None
            if boxes is not None:
                boxes = boxes.cpu().numpy()
            else:
                boxes = []

            if ids is not None:
                ids = ids.cpu().numpy()
            else:
                ids = [-1] * len(boxes)  # fill with -1 if no IDs

            if classes is not None:
                classes = classes.cpu().numpy()
            else:
                classes = [-1] * len(boxes)  # fill with -1 if no classes

            for bbox, obj_id, cls in zip(boxes, ids, classes):
                tracked_results.append({
                    "id": int(obj_id) if obj_id != -1 else None,
                    "class": int(cls) if cls != -1 else None,
                    "bbox": [float(coord) for coord in bbox],
                    "frame": frame_idx
                })
    # Save results as JSON
    with open(results_path, "w") as f:
        json.dump({"tracked_objects": tracked_results}, f)


def download_results(request, filename):
    """Download JSON results"""
    file_path = os.path.join(UPLOAD_DIR, filename)
    if not os.path.exists(file_path):
        return JsonResponse({"error": "File not found"}, status=404)

    with open(file_path, "rb") as f:
        response = HttpResponse(f.read(), content_type="application/json")
        response['Content-Disposition'] = f'attachment; filename="{filename}"'
        return response



