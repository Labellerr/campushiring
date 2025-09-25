
"""
Streamlit Web Demo for YOLO-Seg + ByteTrack
Interactive web interface for video object tracking
"""

import streamlit as st
import cv2
import numpy as np
import tempfile
import json
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from PIL import Image
import sys

# Add your project path
sys.path.append('.')
# from bytetrack_integration import YOLOByteTracker

st.set_page_config(
    page_title="YOLO-Seg + ByteTrack Demo",
    page_icon="🎥",
    layout="wide"
)

def main():
    st.title("🎥 YOLO-Seg + ByteTrack Object Tracking Demo")
    st.markdown("---")

    # Sidebar
    with st.sidebar:
        st.header("Settings")

        # Model settings
        st.subheader("Model Configuration")
        confidence_threshold = st.slider("Confidence Threshold", 0.1, 1.0, 0.5, 0.05)
        track_threshold = st.slider("Track Threshold", 0.1, 1.0, 0.5, 0.05)

        # Upload model
        model_file = st.file_uploader(
            "Upload YOLO Model (.pt file)", 
            type=['pt'], 
            help="Upload your trained YOLO segmentation model"
        )

        st.markdown("---")
        st.subheader("About")
        st.info(
            "This demo showcases end-to-end object detection, segmentation, "
            "and tracking using YOLO-Seg combined with ByteTrack algorithm."
        )

    # Main content
    col1, col2 = st.columns([2, 1])

    with col1:
        st.header("Video Upload & Processing")

        # File upload
        uploaded_file = st.file_uploader(
            "Choose a video file",
            type=['mp4', 'avi', 'mov', 'mkv'],
            help="Upload a video file for object tracking analysis"
        )

        if uploaded_file is not None:
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
                tmp_file.write(uploaded_file.read())
                video_path = tmp_file.name

            # Display video info
            cap = cv2.VideoCapture(video_path)
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            duration = frame_count / fps if fps > 0 else 0
            cap.release()

            st.success(f"Video uploaded successfully!")
            st.write(f"**Resolution:** {width}x{height}")
            st.write(f"**Duration:** {duration:.1f} seconds")
            st.write(f"**FPS:** {fps}")
            st.write(f"**Total Frames:** {frame_count}")

            # Process button
            if st.button("🚀 Start Processing", type="primary"):
                if model_file is None:
                    st.error("Please upload a YOLO model file first!")
                else:
                    process_video(video_path, model_file, confidence_threshold, track_threshold)

    with col2:
        st.header("Demo Instructions")
        st.markdown("""
        ### How to use this demo:

        1. **Upload Model**: Upload your trained YOLO-Seg model (.pt file)
        2. **Upload Video**: Choose a video file for tracking
        3. **Adjust Settings**: Configure detection and tracking thresholds
        4. **Process**: Click the process button to start tracking
        5. **View Results**: Download results and visualizations

        ### Supported Features:
        - ✅ Object Detection & Segmentation
        - ✅ Multi-Object Tracking
        - ✅ Real-time Visualization
        - ✅ Results Export (JSON)
        - ✅ Performance Metrics
        """)

def process_video(video_path, model_file, conf_thresh, track_thresh):
    """Process video with YOLO + ByteTrack"""

    with st.spinner("Processing video... This may take a while."):
        # Save model file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pt') as tmp_model:
            tmp_model.write(model_file.read())
            model_path = tmp_model.name

        try:
            # Initialize tracker (simplified for demo)
            st.info("Initializing YOLO-Seg + ByteTrack...")

            # This would use the actual YOLOByteTracker class
            # tracker = YOLOByteTracker(model_path, conf_thresh=conf_thresh)

            # For demo purposes, simulate processing
            st.info("Processing frames...")
            progress_bar = st.progress(0)

            # Simulate processing progress
            for i in range(100):
                progress_bar.progress(i + 1)

            # Simulate results
            demo_results = create_demo_results()

            st.success("Processing completed!")

            # Display results
            display_results(demo_results)

        except Exception as e:
            st.error(f"Error during processing: {str(e)}")
        finally:
            # Cleanup temporary files
            Path(video_path).unlink(missing_ok=True)
            Path(model_path).unlink(missing_ok=True)

def create_demo_results():
    """Create demo results for visualization"""
    # Simulate tracking results
    results = {
        'summary': {
            'total_frames': 300,
            'total_detections': 1250,
            'unique_tracks': 15,
            'avg_confidence': 0.82,
            'processing_time': 45.2
        },
        'class_distribution': {
            'vehicle': 800,
            'pedestrian': 450
        },
        'confidence_scores': np.random.beta(8, 2, 100).tolist(),
        'tracks_per_frame': np.random.poisson(4, 300).tolist()
    }
    return results

def display_results(results):
    """Display processing results and visualizations"""
    st.header("📊 Processing Results")

    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Frames", results['summary']['total_frames'])
    with col2:
        st.metric("Total Detections", results['summary']['total_detections'])
    with col3:
        st.metric("Unique Tracks", results['summary']['unique_tracks'])
    with col4:
        st.metric("Avg Confidence", f"{results['summary']['avg_confidence']:.2f}")

    # Charts
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Class Distribution")
        fig_pie = px.pie(
            values=list(results['class_distribution'].values()),
            names=list(results['class_distribution'].keys()),
            title="Objects Detected by Class"
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    with col2:
        st.subheader("Confidence Score Distribution")
        fig_hist = px.histogram(
            x=results['confidence_scores'],
            nbins=30,
            title="Detection Confidence Scores"
        )
        fig_hist.update_layout(xaxis_title="Confidence Score", yaxis_title="Count")
        st.plotly_chart(fig_hist, use_container_width=True)

    # Timeline chart
    st.subheader("Tracks per Frame Over Time")
    fig_line = px.line(
        x=range(len(results['tracks_per_frame'])),
        y=results['tracks_per_frame'],
        title="Number of Active Tracks per Frame"
    )
    fig_line.update_layout(xaxis_title="Frame Number", yaxis_title="Active Tracks")
    st.plotly_chart(fig_line, use_container_width=True)

    # Download results
    st.subheader("📥 Download Results")

    # Create downloadable JSON
    results_json = json.dumps(results, indent=2)
    st.download_button(
        label="Download Results (JSON)",
        data=results_json,
        file_name="tracking_results.json",
        mime="application/json"
    )

    # Performance metrics
    st.subheader("⚡ Performance Metrics")
    perf_col1, perf_col2 = st.columns(2)

    with perf_col1:
        st.metric("Processing Time", f"{results['summary']['processing_time']:.1f}s")
    with perf_col2:
        fps = results['summary']['total_frames'] / results['summary']['processing_time']
        st.metric("Processing FPS", f"{fps:.1f}")

if __name__ == "__main__":
    main()
