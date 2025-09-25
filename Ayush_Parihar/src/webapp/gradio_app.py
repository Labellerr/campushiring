import gradio as gr
from pathlib import Path
from src.tracking.bytetrack_runner import track_video


def run(weights, tracker, imgsz, conf, iou, video):
    if video is None:
        return "No video uploaded."
    out = track_video(weights=weights, source_video=video, save_dir="runs/gradio_results", imgsz=int(imgsz), conf=float(conf), iou=float(iou), tracker_cfg={"tracker": tracker})
    return f"Done. Results: {out}"


def app():
    with gr.Blocks() as demo:
        gr.Markdown("# YOLOv8-Seg + ByteTrack Demo")
        weights = gr.Textbox(value="yolov8s-seg.pt", label="Weights")
        tracker = gr.Dropdown(["bytetrack.yaml", "botsort.yaml"], value="bytetrack.yaml", label="Tracker")
        imgsz = gr.Number(value=640, label="Image size")
        conf = gr.Slider(0, 1, value=0.25, step=0.05, label="Confidence")
        iou = gr.Slider(0, 1, value=0.45, step=0.05, label="IoU")
        video = gr.Video(label="Upload Video")
        btn = gr.Button("Run Tracking")
        out = gr.Textbox(label="Output")
        btn.click(fn=run, inputs=[weights, tracker, imgsz, conf, iou, video], outputs=out)
    return demo


if __name__ == "__main__":
    app().launch()
