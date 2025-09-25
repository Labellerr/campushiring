import os
import json
from ultralytics import YOLO
import supervision  as sv


# --- 1. CONFIGURATION ---

# Folder where the script is
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Go up one level to Om_Singh
PROJECT_DIR = os.path.dirname(BASE_DIR)  # parent folder of app/

# Paths for outputs in video_and_json folder
VIDEO_DIR = os.path.join(PROJECT_DIR, "video_and_json")  # Om_Singh/video_and_json
os.makedirs(VIDEO_DIR, exist_ok=True)  # ensure folder exists

MODEL_PATH = os.path.join(BASE_DIR, "best.pt")  # model stays in app/
OUTPUT_VIDEO_PATH = os.path.join(VIDEO_DIR, "output.mp4")
OUTPUT_JSON_PATH  = os.path.join(VIDEO_DIR, "result.json")

def process_video(INPUT_VIDEO_PATH, OUTPUT_VIDEO_PATH, OUTPUT_JSON_PATH):
    print("Loading model...")
    model = YOLO(MODEL_PATH)

    print("Initializing tracker and annotators...")
    tracker = sv.ByteTrack()
    box_annotator = sv.BoxAnnotator(thickness=5)
    label_annotator = sv.LabelAnnotator(text_position=sv.Position.TOP_CENTER, text_scale = 1, text_thickness= 1)

    sv.LabelAnnotator()

    frame_generator = sv.get_video_frames_generator(source_path=INPUT_VIDEO_PATH)
    video_info = sv.VideoInfo.from_video_path(INPUT_VIDEO_PATH)

    results_list = []

    with sv.VideoSink(target_path=OUTPUT_VIDEO_PATH, video_info=video_info) as sink:
        print("Processing video frames...")
        for frame_number, frame in enumerate(frame_generator):
            # Run YOLO prediction
            results = model(frame)[0]
            detections = sv.Detections.from_ultralytics(results)

            # Update tracker
            tracked_detections = tracker.update_with_detections(detections=detections)

            # Prepare labels
            labels = [
                f"ID: {det[4]} {model.model.names[int(det[3])]}"
                for det in tracked_detections
            ]

            # Annotate frame
            annotated_frame = box_annotator.annotate(scene=frame.copy(), detections=tracked_detections)
            annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=tracked_detections, labels=labels)

            # Save tracking info
            for det in tracked_detections:
                bbox = det[0]
                conf = det[2]
                class_id = int(det[3])
                tracker_id = det[4]

                results_list.append({
                    "frame_number": frame_number,
                    "track_id": int(tracker_id),
                    "class": model.model.names[class_id],
                    "confidence": float(conf),
                    "bounding_box": [int(coord) for coord in bbox]
                })

            # Write annotated frame
            sink.write_frame(frame=annotated_frame)

            if frame_number % 30 == 0:
                print(f"Processed frame {frame_number}...")

    print("Video processing complete. Saving results...")
    with open(OUTPUT_JSON_PATH, 'w') as f:
        json.dump(results_list, f, indent=4)

    print("--- All tasks finished successfully! ---")

