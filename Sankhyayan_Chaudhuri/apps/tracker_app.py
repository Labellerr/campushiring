import os
import json
import gc
import tempfile
import streamlit as st
from pathlib import Path
from ultralytics import YOLO


def track_video(
    weights: str,
    video_path: str,
    out_dir: str = 'artifacts/tracking',
    tracker: str = 'bytetrack.yaml',
    imgsz: int = 640,
    conf: float = 0.25,
    vid_stride: int = 1,
    max_frames: int | None = None,
    classes: list[int] | None = None,
    save_video: bool = True,
):
    model = YOLO(weights)
    os.makedirs(out_dir, exist_ok=True)

    results = model.track(
        source=video_path,
        tracker=tracker,
        imgsz=imgsz,
        conf=conf,
        vid_stride=vid_stride,
        classes=classes,
        persist=True,
        save=False,
        stream=True,
        verbose=False,
    )

    # Optional video writer
    video_out = None
    writer = None
    if save_video:
        run_dir = Path(out_dir) / 'track_run'
        run_dir.mkdir(parents=True, exist_ok=True)
        video_out = str(run_dir / 'output.mp4')
        import cv2
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(video_out, fourcc, fps, (width, height))

    # Stream results JSON to disk
    out_json_path = os.path.join(out_dir, 'results.json')
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    with open(out_json_path, 'w', encoding='utf-8') as f:
        f.write('{"tracks":[')
        first = True
        for frame_idx, r in enumerate(results):
            if max_frames and frame_idx >= max_frames:
                break
            if r.boxes is not None and r.boxes.id is not None:
                ids = r.boxes.id.int().cpu().tolist()
                xyxy = r.boxes.xyxy.cpu().numpy().tolist()
                cls = r.boxes.cls.int().cpu().tolist()
                confs = r.boxes.conf.cpu().numpy().tolist()
                for tid, box, c, sc in zip(ids, xyxy, cls, confs):
                    item = {
                        'frame': int(frame_idx),
                        'track_id': int(tid),
                        'class_id': int(c),
                        'bbox_xyxy': [float(v) for v in box],
                        'score': float(sc),
                    }
                    if not first:
                        f.write(',')
                    f.write(json.dumps(item))
                    first = False
            if writer is not None:
                annotated = r.plot()
                writer.write(annotated)
            del r
            gc.collect()
        f.write(']}')
    if writer is not None:
        writer.release()

    return out_json_path, video_out


def main():
    st.title('YOLOv8 + ByteTrack Tracker')

    weights = st.text_input('YOLO Weights (detector)', 'yolov8n.pt')
    conf = st.slider('Confidence', 0.0, 1.0, 0.25, 0.05)
    imgsz = st.selectbox('Image size', [480, 640, 960, 1280], index=1)
    vid_stride = st.number_input('Frame stride (skip frames)', min_value=1, max_value=10, value=1, step=1)
    max_frames = st.number_input('Max frames (0 = all)', min_value=0, max_value=100000, value=0, step=100)
    save_video = st.checkbox('Save annotated video', value=True)
    classes = st.multiselect('Classes (COCO IDs)', options=list(range(0,81)), default=[0,2,3,5,7])

    up = st.file_uploader('Upload video', type=['mp4', 'mov', 'avi', 'mkv'])
    if up is not None and st.button('Run Tracking'):
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(up.name)[1]) as t:
            t.write(up.read())
            video_path = t.name
        with st.spinner('Tracking...'):
            out_json, out_video = track_video(
                weights=weights,
                video_path=video_path,
                imgsz=imgsz,
                conf=conf,
                vid_stride=vid_stride,
                max_frames=(None if max_frames == 0 else int(max_frames)),
                classes=classes,
                save_video=save_video,
            )
        st.success('Done!')
        st.download_button('Download results.json', data=open(out_json,'rb').read(), file_name='results.json')
        if save_video and out_video:
            st.video(out_video)


if __name__ == '__main__':
    main()
