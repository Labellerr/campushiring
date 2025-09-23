import json
from ultralytics import YOLO
from moviepy.editor import VideoFileClip, VideoClip
from PIL import Image, ImageDraw, ImageFont
import numpy as np

def track_video(input_path, output_path, model_weights, json_path, progress_callback=None):
    try:
        model = YOLO(model_weights)

        clip = VideoFileClip(input_path)
        total_frames = int(clip.fps * clip.duration)

        results_data = []
        frame_id = 0

        # Use default PIL font
        font = ImageFont.load_default()

        def process_frame(get_frame, t):
            nonlocal frame_id
            frame_id += 1
            frame = get_frame(t)
            pil_img = Image.fromarray(frame)
            draw = ImageDraw.Draw(pil_img)

            # Running the YOLO tracker on this frame alone is hard; Instead, we run tracking on the entire video with stream=True
            # So we will approximate here by returning the same frame if possible.
            # To fully re-implement tracking with moviepy frame processing is complicated.
            return np.array(pil_img)

        # Instead, use model.track with stream=True to get results for all frames first
        results = model.track(
            source=input_path,
            conf=0.5,
            iou=0.7,
            tracker="bytetrack.yaml",
            stream=True,
            verbose=False
        )

        # Preprocess frames with bounding boxes
        frames_with_boxes = []

        for result in results:
            frame_id += 1
            frame = Image.fromarray(result.orig_img)
            draw = ImageDraw.Draw(frame)

            frame_objects = []

            if result.boxes is not None and result.boxes.id is not None:
                boxes = result.boxes.xyxy.cpu().numpy()
                track_ids = result.boxes.id.cpu().numpy().astype(int)
                confidences = result.boxes.conf.cpu().numpy()
                classes = result.boxes.cls.cpu().numpy().astype(int)

                for box, tid, conf, cid in zip(boxes, track_ids, confidences, classes):
                    x1, y1, x2, y2 = map(int, box)
                    cls_name = model.names.get(cid, f"class_{cid}")
                    color = (0, 255, 0) if "pedestrian" in cls_name.lower() else (255, 0, 0)

                    # Draw bounding box
                    draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
                    label = f"{cls_name}#{tid}"

                    # Draw label background
                    text_size = draw.textsize(label, font=font)
                    draw.rectangle([x1, y1 - text_size[1], x1 + text_size[0], y1], fill=color)

                    # Draw label text
                    draw.text((x1, y1 - text_size[1]), label, fill=(255, 255, 255), font=font)

                    frame_objects.append({
                        "id": int(tid),
                        "class": cls_name,
                        "confidence": float(conf),
                        "bbox": [x1, y1, x2, y2]
                    })
            frames_with_boxes.append(np.array(frame))

            results_data.append({"frame_id": frame_id, "objects": frame_objects})

            if progress_callback:
                progress_callback(frame_id, total_frames)

        # Write output video with moviepy
        def make_frame(t):
            frame_index = min(int(t * clip.fps), len(frames_with_boxes) - 1)
            return frames_with_boxes[frame_index]

        tracked_clip = VideoClip(make_frame, duration=clip.duration)
        tracked_clip.write_videofile(output_path, fps=clip.fps, codec='libx264', audio=False)

        with open(json_path, "w") as jf:
            json.dump(results_data, jf, indent=2)

        return True, None

    except Exception as e:
        return False, str(e)
