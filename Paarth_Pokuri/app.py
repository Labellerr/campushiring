import streamlit as st
import os
import json
import tempfile
import cv2
from pathlib import Path
import traceback
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import io
import zipfile
import logging
from typing import Dict, List, Tuple, Optional
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set page config
st.set_page_config(
    page_title="AI Traffic Analytics",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .main-header {
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .status-success {
        color: #28a745;
    }
    .status-error {
        color: #dc3545;
    }
    .status-warning {
        color: #ffc107;
    }
</style>
""", unsafe_allow_html=True)

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

class TrafficAnalyzer:
    def __init__(self):
        self.model = None
        self.tracks = []
        
    def load_model(self, model_path: str) -> bool:
        """Load YOLO model with comprehensive error handling"""
        try:
            if not os.path.exists(model_path):
                st.error(f"❌ Model file '{model_path}' not found.")
                return False
            
            self.model = YOLO(model_path)
            logger.info(f"Model loaded successfully from {model_path}")
            return True
        except Exception as e:
            st.error(f"❌ Error loading model: {str(e)}")
            logger.error(f"Model loading error: {e}")
            return False
    
    def get_video_info(self, video_path: str) -> Dict:
        """Extract video metadata"""
        cap = cv2.VideoCapture(video_path)
        info = {
            "fps": int(cap.get(cv2.CAP_PROP_FPS)),
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "total_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            "duration": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) / int(cap.get(cv2.CAP_PROP_FPS))
        }
        cap.release()
        return info
    
    def process_video_advanced(self, input_path: str, output_path: str, 
                             confidence_threshold: float, progress_bar, 
                             status_text, enable_analytics: bool = True) -> List[Dict]:
        """Enhanced video processing with analytics"""
        
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise Exception("Could not open video file")
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Setup video writer with better codec
        fourcc = cv2.VideoWriter_fourcc(*'H264') if cv2.VideoWriter_fourcc(*'H264') else cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        tracks = []
        frame_count = 0
        processing_times = []
        
        # Analytics data structures
        zone_counts = {'center': 0, 'left': 0, 'right': 0, 'top': 0, 'bottom': 0}
        object_trajectories = {}
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                start_time = time.time()
                
                # Run tracking on frame
                results = self.model.track(frame, persist=True, conf=confidence_threshold, verbose=False)
                
                # Process results
                if results and len(results) > 0:
                    r = results[0]
                    
                    # Draw tracking results on frame
                    if hasattr(r, 'plot'):
                        annotated_frame = r.plot()
                    else:
                        annotated_frame = frame.copy()
                    
                    # Extract tracking data
                    if r.boxes is not None and hasattr(r.boxes, 'id') and r.boxes.id is not None:
                        track_ids = r.boxes.id.int().cpu().tolist()
                        bboxes = r.boxes.xyxy.cpu().tolist()
                        confidences = r.boxes.conf.cpu().tolist()
                        class_ids = r.boxes.cls.int().cpu().tolist()
                        
                        # Get class names
                        class_names = r.names if hasattr(r, 'names') else {}
                        
                        for tid, bbox, conf, cls_id in zip(track_ids, bboxes, confidences, class_ids):
                            label = class_names.get(cls_id, f"class_{cls_id}")
                            
                            # Calculate center point and zone
                            center_x = (bbox[0] + bbox[2]) / 2
                            center_y = (bbox[1] + bbox[3]) / 2
                            
                            # Determine zone
                            zone = self.get_zone(center_x, center_y, width, height)
                            zone_counts[zone] += 1
                            
                            # Track trajectory
                            if tid not in object_trajectories:
                                object_trajectories[tid] = []
                            object_trajectories[tid].append((center_x, center_y, frame_count))
                            
                            # Calculate speed (if trajectory exists)
                            speed = 0
                            if len(object_trajectories[tid]) > 5:  # Need at least 5 points
                                prev_point = object_trajectories[tid][-6]
                                curr_point = object_trajectories[tid][-1]
                                distance = np.sqrt((curr_point[0] - prev_point[0])**2 + (curr_point[1] - prev_point[1])**2)
                                time_diff = (curr_point[2] - prev_point[2]) / fps
                                speed = distance / time_diff if time_diff > 0 else 0
                            
                            tracks.append({
                                "frame": frame_count,
                                "timestamp": frame_count / fps,
                                "id": tid,
                                "class": label,
                                "class_id": cls_id,
                                "confidence": float(conf),
                                "bbox": [float(x) for x in bbox],
                                "center": [float(center_x), float(center_y)],
                                "zone": zone,
                                "speed": float(speed),
                                "area": float((bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))
                            })
                    
                    # Add zone overlays if analytics enabled
                    if enable_analytics:
                        annotated_frame = self.draw_zones(annotated_frame, width, height)
                        annotated_frame = self.draw_analytics_overlay(annotated_frame, zone_counts, frame_count, fps)
                
                else:
                    annotated_frame = frame
                
                # Write frame
                out.write(annotated_frame)
                frame_count += 1
                
                # Track processing time
                processing_time = time.time() - start_time
                processing_times.append(processing_time)
                
                # Update progress
                if frame_count % 5 == 0:
                    progress = min(frame_count / total_frames, 1.0)
                    avg_time = np.mean(processing_times[-50:]) if processing_times else 0
                    eta = avg_time * (total_frames - frame_count)
                    
                    progress_bar.progress(progress, f"Frame {frame_count}/{total_frames}")
                    status_text.text(f"Processing... ETA: {eta:.1f}s | FPS: {1/avg_time:.1f}")
        
        finally:
            cap.release()
            out.release()
        
        return tracks
    
    def get_zone(self, x: float, y: float, width: int, height: int) -> str:
        """Determine which zone the object is in"""
        center_x, center_y = width // 2, height // 2
        threshold = 0.3
        
        if abs(x - center_x) < width * threshold and abs(y - center_y) < height * threshold:
            return 'center'
        elif x < center_x and y < center_y:
            return 'top' if abs(y - center_y) > abs(x - center_x) else 'left'
        elif x > center_x and y < center_y:
            return 'top' if abs(y - center_y) > abs(x - center_x) else 'right'
        elif x < center_x and y > center_y:
            return 'bottom' if abs(y - center_y) > abs(x - center_x) else 'left'
        else:
            return 'bottom' if abs(y - center_y) > abs(x - center_x) else 'right'
    
    def draw_zones(self, frame: np.ndarray, width: int, height: int) -> np.ndarray:
        """Draw zone boundaries on frame"""
        # Draw zone lines
        cv2.line(frame, (width//3, 0), (width//3, height), (0, 255, 0), 2)
        cv2.line(frame, (2*width//3, 0), (2*width//3, height), (0, 255, 0), 2)
        cv2.line(frame, (0, height//3), (width, height//3), (0, 255, 0), 2)
        cv2.line(frame, (0, 2*height//3), (width, 2*height//3), (0, 255, 0), 2)
        
        return frame
    
    def draw_analytics_overlay(self, frame: np.ndarray, zone_counts: Dict, 
                             frame_count: int, fps: int) -> np.ndarray:
        """Draw analytics overlay on frame"""
        overlay = frame.copy()
        
        # Add timestamp
        timestamp = f"Time: {frame_count/fps:.1f}s"
        cv2.putText(overlay, timestamp, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Add zone counts
        y_offset = 60
        for zone, count in zone_counts.items():
            text = f"{zone.capitalize()}: {count}"
            cv2.putText(overlay, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            y_offset += 30
        
        return overlay

def create_analytics_dashboard(tracks: List[Dict]) -> None:
    """Create comprehensive analytics dashboard"""
    if not tracks:
        st.warning("No tracking data available for analytics.")
        return
    
    df = pd.DataFrame(tracks)
    
    # Basic statistics
    st.subheader("📊 Traffic Analytics Dashboard")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_detections = len(df)
        st.metric("Total Detections", total_detections)
    
    with col2:
        unique_objects = df['id'].nunique()
        st.metric("Unique Objects", unique_objects)
    
    with col3:
        avg_confidence = df['confidence'].mean()
        st.metric("Avg Confidence", f"{avg_confidence:.2f}")
    
    with col4:
        classes_detected = df['class'].nunique()
        st.metric("Classes Detected", classes_detected)
    
    # Time series analysis
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🕒 Objects Over Time")
        time_series = df.groupby('frame')['id'].nunique().reset_index()
        time_series['timestamp'] = time_series['frame'] / 30  # Assuming 30 FPS
        
        fig_time = px.line(time_series, x='timestamp', y='id', 
                          title='Active Objects Over Time',
                          labels={'timestamp': 'Time (seconds)', 'id': 'Number of Objects'})
        st.plotly_chart(fig_time, use_container_width=True)
    
    with col2:
        st.subheader("📍 Zone Distribution")
        zone_counts = df['zone'].value_counts()
        fig_zones = px.pie(values=zone_counts.values, names=zone_counts.index,
                          title='Detection Distribution by Zone')
        st.plotly_chart(fig_zones, use_container_width=True)
    
    # Class distribution
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🚗 Object Classes")
        class_counts = df['class'].value_counts()
        fig_classes = px.bar(x=class_counts.index, y=class_counts.values,
                           title='Object Class Distribution',
                           labels={'x': 'Object Class', 'y': 'Count'})
        st.plotly_chart(fig_classes, use_container_width=True)
    
    with col2:
        st.subheader("⚡ Speed Analysis")
        speed_data = df[df['speed'] > 0]['speed']
        if len(speed_data) > 0:
            fig_speed = px.histogram(speed_data, nbins=20,
                                   title='Speed Distribution (pixels/second)',
                                   labels={'value': 'Speed', 'count': 'Frequency'})
            st.plotly_chart(fig_speed, use_container_width=True)
        else:
            st.info("No speed data available")
    
    # Trajectory heatmap
    st.subheader("🗺️ Movement Heatmap")
    if 'center' in df.columns:
        centers = pd.DataFrame(df['center'].tolist(), columns=['x', 'y'])
        
        # Create heatmap
        fig_heatmap = px.density_heatmap(centers, x='x', y='y',
                                       title='Object Movement Density',
                                       labels={'x': 'X Coordinate', 'y': 'Y Coordinate'})
        fig_heatmap.update_layout(yaxis={'autorange': 'reversed'})  # Flip Y axis to match image coordinates
        st.plotly_chart(fig_heatmap, use_container_width=True)

def create_download_package(tracks: List[Dict], video_path: str, 
                          confidence_threshold: float, processing_method: str) -> bytes:
    """Create a comprehensive download package"""
    buffer = io.BytesIO()
    
    with zipfile.ZipFile(buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
        # Add JSON data
        json_data = {
            "metadata": {
                "total_detections": len(tracks),
                "unique_objects": len(set(track['id'] for track in tracks)) if tracks else 0,
                "confidence_threshold": confidence_threshold,
                "processing_method": processing_method,
                "processing_timestamp": datetime.now().isoformat(),
                "classes_detected": list(set(track['class'] for track in tracks)) if tracks else []
            },
            "tracks": tracks
        }
        
        zf.writestr("tracking_results.json", json.dumps(json_data, indent=2))
        
        # Add CSV for easy analysis
        if tracks:
            df = pd.DataFrame(tracks)
            csv_buffer = io.StringIO()
            df.to_csv(csv_buffer, index=False)
            zf.writestr("tracking_data.csv", csv_buffer.getvalue())
        
        # Add summary statistics
        if tracks:
            summary = {
                "summary_statistics": {
                    "total_frames_processed": max(track['frame'] for track in tracks) + 1,
                    "average_confidence": sum(track['confidence'] for track in tracks) / len(tracks),
                    "object_classes": {},
                    "zone_distribution": {},
                    "processing_info": {
                        "confidence_threshold": confidence_threshold,
                        "method": processing_method
                    }
                }
            }
            
            # Calculate class distribution
            class_counts = {}
            zone_counts = {}
            for track in tracks:
                class_counts[track['class']] = class_counts.get(track['class'], 0) + 1
                zone_counts[track['zone']] = zone_counts.get(track['zone'], 0) + 1
            
            summary["summary_statistics"]["object_classes"] = class_counts
            summary["summary_statistics"]["zone_distribution"] = zone_counts
            
            zf.writestr("summary_statistics.json", json.dumps(summary, indent=2))
    
    buffer.seek(0)
    return buffer.getvalue()

def main():
    # Header
    st.markdown("<h1 class='main-header'>🚦 AI Traffic Analytics Platform</h1>", unsafe_allow_html=True)
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Model selection
        st.subheader("🤖 Model Settings")
        
        # Check for available models
        model_files = [f for f in os.listdir('.') if f.endswith(('.pt', '.onnx'))]
        default_models = ['yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt', 'yolov8l.pt', 'yolov8x.pt']
        
        if not model_files:
            model_files = default_models
            st.info("💡 No local models found. You can download YOLOv8 models or upload your own.")
        
        selected_model = st.selectbox("Select Model", model_files)
        
        # Auto-download option for default models
        if selected_model in default_models and not os.path.exists(selected_model):
            if st.button(f"📥 Download {selected_model}"):
                with st.spinner(f"Downloading {selected_model}..."):
                    try:
                        model = YOLO(selected_model)  # This will auto-download
                        st.success(f"✅ {selected_model} downloaded successfully!")
                    except Exception as e:
                        st.error(f"❌ Download failed: {str(e)}")
        
        # Processing options
        st.subheader("🎛️ Processing Options")
        confidence_threshold = st.slider("Confidence Threshold", 0.1, 1.0, 0.5, 0.05)
        
        enable_analytics = st.checkbox("Enable Advanced Analytics", True)
        enable_zones = st.checkbox("Show Zone Boundaries", True)
        
        # Performance options
        st.subheader("⚡ Performance")
        frame_skip = st.slider("Frame Skip (for faster processing)", 0, 5, 0)
        
        # Export options
        st.subheader("💾 Export Options")
        export_format = st.selectbox("Video Export Format", ["MP4", "AVI"])
        include_analytics = st.checkbox("Include Analytics Overlay", True)
    
    # Main content
    if not YOLO_AVAILABLE:
        st.error("❌ ultralytics package not found. Please install it with: `pip install ultralytics`")
        st.code("pip install ultralytics", language="bash")
        st.stop()
    
    # Initialize analyzer
    analyzer = TrafficAnalyzer()
    
    # Load model
    if os.path.exists(selected_model) or selected_model in ['yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt', 'yolov8l.pt', 'yolov8x.pt']:
        if analyzer.load_model(selected_model):
            st.success(f"✅ Model '{selected_model}' loaded successfully!")
        else:
            st.stop()
    else:
        st.error(f"❌ Model file '{selected_model}' not found. Please ensure the model is available.")
        st.stop()
    
    # File upload section
    st.subheader("📁 Video Upload")
    
    # Multiple file upload support
    uploaded_files = st.file_uploader(
        "Upload video files", 
        type=["mp4", "avi", "mov", "mkv", "webm"],
        accept_multiple_files=True,
        help="Supported formats: MP4, AVI, MOV, MKV, WebM"
    )
    
    if uploaded_files:
        # Process each uploaded file
        for idx, uploaded_file in enumerate(uploaded_files):
            if len(uploaded_files) > 1:
                st.markdown(f"### Processing File {idx + 1}: {uploaded_file.name}")
            
            # Display file info
            file_size_mb = uploaded_file.size / 1024 / 1024
            st.info(f"📁 **{uploaded_file.name}** ({file_size_mb:.1f} MB)")
            
            # Create temporary directory for processing
            with tempfile.TemporaryDirectory() as temp_dir:
                # Save uploaded video
                input_video_path = os.path.join(temp_dir, uploaded_file.name)
                with open(input_video_path, "wb") as f:
                    f.write(uploaded_file.read())
                
                # Get video information
                try:
                    video_info = analyzer.get_video_info(input_video_path)
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Duration", f"{video_info['duration']:.1f}s")
                    with col2:
                        st.metric("FPS", video_info['fps'])
                    with col3:
                        st.metric("Resolution", f"{video_info['width']}x{video_info['height']}")
                    with col4:
                        st.metric("Total Frames", video_info['total_frames'])
                    
                except Exception as e:
                    st.error(f"❌ Could not read video information: {str(e)}")
                    continue
                
                # Display original video
                st.subheader("📹 Original Video")
                st.video(input_video_path)
                
                # Processing button
                if st.button(f"🚀 Process Video {idx + 1}" if len(uploaded_files) > 1 else "🚀 Process Video", 
                           type="primary", key=f"process_{idx}"):
                    try:
                        # Create output path
                        output_filename = f"tracked_{uploaded_file.name.rsplit('.', 1)[0]}.mp4"
                        output_video_path = os.path.join(temp_dir, output_filename)
                        
                        # Processing UI
                        st.info("🎬 Processing video...")
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        # Process video
                        start_time = time.time()
                        tracks = analyzer.process_video_advanced(
                            input_video_path, 
                            output_video_path, 
                            confidence_threshold,
                            progress_bar,
                            status_text,
                            enable_analytics
                        )
                        
                        processing_time = time.time() - start_time
                        
                        progress_bar.progress(1.0)
                        status_text.success(f"✅ Processing complete! ({processing_time:.1f}s)")
                        
                        # Display results
                        if os.path.exists(output_video_path):
                            unique_objects = len(set(track['id'] for track in tracks)) if tracks else 0
                            st.success(f"✅ Tracking complete! Detected {len(tracks)} instances of {unique_objects} unique objects.")
                            
                            # Display processed video
                            st.subheader("🎯 Processed Video")
                            st.video(output_video_path)
                            
                            # Analytics dashboard
                            if tracks and enable_analytics:
                                create_analytics_dashboard(tracks)
                            
                            # Download section
                            st.subheader("💾 Download Results")
                            
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                # Download video
                                with open(output_video_path, "rb") as f:
                                    st.download_button(
                                        label="📹 Download Processed Video",
                                        data=f,
                                        file_name=output_filename,
                                        mime="video/mp4"
                                    )
                            
                            with col2:
                                # Download JSON
                                if tracks:
                                    json_data = {
                                        "metadata": {
                                            "filename": uploaded_file.name,
                                            "total_detections": len(tracks),
                                            "unique_objects": unique_objects,
                                            "confidence_threshold": confidence_threshold,
                                            "processing_time": processing_time,
                                            "video_info": video_info
                                        },
                                        "tracks": tracks
                                    }
                                    
                                    st.download_button(
                                        label="📊 Download JSON Data",
                                        data=json.dumps(json_data, indent=2),
                                        file_name=f"results_{uploaded_file.name.rsplit('.', 1)[0]}.json",
                                        mime="application/json"
                                    )
                            
                            with col3:
                                # Download complete package
                                if tracks:
                                    package_data = create_download_package(
                                        tracks, output_video_path, confidence_threshold, "Advanced Processing"
                                    )
                                    
                                    st.download_button(
                                        label="📦 Download Complete Package",
                                        data=package_data,
                                        file_name=f"complete_analysis_{uploaded_file.name.rsplit('.', 1)[0]}.zip",
                                        mime="application/zip"
                                    )
                        
                        else:
                            st.error("❌ Could not process video. Please check your input.")
                            
                            # Debug information
                            with st.expander("🔍 Debug Information"):
                                st.write("Temporary directory contents:")
                                for root, dirs, files in os.walk(temp_dir):
                                    for file in files:
                                        st.text(f"📁 {os.path.relpath(os.path.join(root, file), temp_dir)}")
                    
                    except Exception as e:
                        st.error(f"❌ Error during processing: {str(e)}")
                        with st.expander("🔍 Full Error Details"):
                            st.code(traceback.format_exc())
    
    else:
        # Landing page content
        st.markdown("""
        ### Welcome to AI Traffic Analytics Platform
        
        This advanced platform provides comprehensive vehicle and pedestrian tracking using state-of-the-art YOLOv8 models with ByteTrack integration.
        
        #### 🌟 Key Features:
        - **Multi-object Tracking**: Track vehicles and pedestrians with unique IDs
        - **Real-time Analytics**: Zone analysis, speed estimation, trajectory mapping
        - **Advanced Visualizations**: Heatmaps, time series analysis, statistical dashboards  
        - **Multiple Export Formats**: JSON, CSV, complete analysis packages
        - **Cloud-Ready Architecture**: Optimized for deployment on cloud platforms
        - **Batch Processing**: Support for multiple video files
        
        #### 🚀 Getting Started:
        1. **Upload your video file(s)** using the file uploader above
        2. **Configure settings** in the sidebar (model, confidence threshold, etc.)
        3. **Click Process Video** to start the analysis
        4. **Download results** including processed video and analytics data
        
        #### 📋 Supported Formats:
        - **Input**: MP4, AVI, MOV, MKV, WebM
        - **Output**: MP4 (H.264), JSON, CSV, ZIP packages
        
        ---
        *Built with Streamlit, YOLOv8, and OpenCV*
        """)

if __name__ == "__main__":
    main()