
# Streamlit Web Application Template
import streamlit as st
import tempfile
import json
import cv2
import os
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

# Import your custom classes
# from yolo_bytetrack_integration import YOLOByteTrackIntegration

class VideoTrackingApp:
    def __init__(self):
        self.setup_page_config()
        self.model_path = self.get_model_path()

    def setup_page_config(self):
        """Configure Streamlit page"""
        st.set_page_config(
            page_title="Vehicle & Pedestrian Tracking",
            page_icon="🚗",
            layout="wide",
            initial_sidebar_state="expanded"
        )

    def get_model_path(self):
        """Get model path from user or default location"""
        return "runs/segment/vehicle_pedestrian_seg/weights/best.pt"

    def main(self):
        """Main application interface"""
        # Header
        st.title("🚗 Vehicle & Pedestrian Tracking Demo")
        st.markdown("Upload a video to track vehicles and pedestrians using YOLO-Seg + ByteTrack")

        # Sidebar
        self.render_sidebar()

        # Main content
        col1, col2 = st.columns([2, 1])

        with col1:
            self.render_upload_section()

        with col2:
            self.render_info_section()

    def render_sidebar(self):
        """Render sidebar with model configuration"""
        st.sidebar.header("⚙️ Configuration")

        # Model settings
        st.sidebar.subheader("Detection Settings")
        conf_thresh = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.5, 0.05)
        high_thresh = st.sidebar.slider("High Confidence Threshold", 0.1, 1.0, 0.6, 0.05)
        low_thresh = st.sidebar.slider("Low Confidence Threshold", 0.1, 1.0, 0.1, 0.05)

        # Tracking settings
        st.sidebar.subheader("Tracking Settings")
        track_buffer = st.sidebar.slider("Track Buffer", 10, 100, 30, 5)
        match_thresh = st.sidebar.slider("Match Threshold", 0.1, 1.0, 0.8, 0.05)

        # Visualization settings
        st.sidebar.subheader("Visualization")
        show_tracks = st.sidebar.checkbox("Show Track IDs", value=True)
        show_confidence = st.sidebar.checkbox("Show Confidence", value=True)

        return {
            'conf_thresh': conf_thresh,
            'high_thresh': high_thresh,
            'low_thresh': low_thresh,
            'track_buffer': track_buffer,
            'match_thresh': match_thresh,
            'show_tracks': show_tracks,
            'show_confidence': show_confidence
        }

    def render_upload_section(self):
        """Render video upload and processing section"""
        uploaded_file = st.file_uploader(
            "Choose a video file",
            type=['mp4', 'avi', 'mov', 'mkv'],
            help="Upload a video file to track vehicles and pedestrians"
        )

        if uploaded_file is not None:
            # Save uploaded file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
                tmp_file.write(uploaded_file.read())
                video_path = tmp_file.name

            # Display video info
            st.success(f"Video uploaded: {uploaded_file.name}")
            self.display_video_info(video_path)

            # Process button
            if st.button("🚀 Start Tracking", type="primary"):
                self.process_video(video_path, uploaded_file.name)

    def display_video_info(self, video_path):
        """Display video information"""
        cap = cv2.VideoCapture(video_path)

        if cap.isOpened():
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            duration = frame_count / fps if fps > 0 else 0

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Duration", f"{duration:.1f}s")
            col2.metric("Resolution", f"{width}x{height}")
            col3.metric("FPS", fps)
            col4.metric("Frames", frame_count)

            cap.release()

    def process_video(self, video_path, filename):
        """Process video with tracking"""
        try:
            # Initialize progress bars
            progress_bar = st.progress(0)
            status_text = st.empty()

            # Create tracker instance
            status_text.text("Initializing tracker...")
            # tracker = YOLOByteTrackIntegration(self.model_path)

            # Process video
            status_text.text("Processing video...")

            # Simulate processing (replace with actual implementation)
            import time
            total_frames = 100  # Get actual frame count

            for i in range(total_frames):
                progress = (i + 1) / total_frames
                progress_bar.progress(progress)
                status_text.text(f"Processing frame {i+1}/{total_frames}")
                time.sleep(0.01)  # Simulate processing time

            # Simulate results (replace with actual results)
            tracking_results = self.generate_sample_results()

            # Display results
            self.display_results(tracking_results, filename)

            progress_bar.progress(1.0)
            status_text.text("✅ Processing complete!")

        except Exception as e:
            st.error(f"Error processing video: {str(e)}")

    def generate_sample_results(self):
        """Generate sample tracking results for demo"""
        import random

        tracking_results = []
        for frame_idx in range(100):
            for track_id in range(random.randint(1, 5)):
                result = {
                    'frame_number': frame_idx,
                    'track_id': track_id,
                    'class_name': random.choice(['vehicle', 'pedestrian']),
                    'confidence': random.uniform(0.5, 0.95),
                    'bbox': {
                        'x1': random.randint(0, 640),
                        'y1': random.randint(0, 480),
                        'x2': random.randint(0, 640),
                        'y2': random.randint(0, 480)
                    }
                }
                tracking_results.append(result)

        return tracking_results

    def display_results(self, tracking_results, filename):
        """Display tracking results and visualizations"""
        st.success("🎉 Tracking completed successfully!")

        # Create tabs for different views
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Statistics", "📈 Timeline", "💾 Export", "🔍 Details"])

        with tab1:
            self.render_statistics(tracking_results)

        with tab2:
            self.render_timeline(tracking_results)

        with tab3:
            self.render_export_options(tracking_results, filename)

        with tab4:
            self.render_detailed_results(tracking_results)

    def render_statistics(self, tracking_results):
        """Render tracking statistics"""
        if not tracking_results:
            st.warning("No tracking results to display")
            return

        # Calculate statistics
        total_detections = len(tracking_results)
        unique_tracks = len(set(r['track_id'] for r in tracking_results))
        class_counts = {}
        for result in tracking_results:
            class_name = result['class_name']
            class_counts[class_name] = class_counts.get(class_name, 0) + 1

        # Display metrics
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Detections", total_detections)
        col2.metric("Unique Tracks", unique_tracks)
        col3.metric("Average Confidence", f"{np.mean([r['confidence'] for r in tracking_results]):.2f}")

        # Class distribution chart
        if class_counts:
            fig = px.pie(values=list(class_counts.values()), 
                        names=list(class_counts.keys()),
                        title="Detection Distribution by Class")
            st.plotly_chart(fig, use_container_width=True)

    def render_timeline(self, tracking_results):
        """Render timeline visualization"""
        if not tracking_results:
            return

        # Create timeline data
        df = pd.DataFrame(tracking_results)

        # Detections over time
        frame_counts = df.groupby(['frame_number', 'class_name']).size().reset_index(name='count')

        fig = px.line(frame_counts, x='frame_number', y='count', color='class_name',
                     title="Detections Over Time")
        st.plotly_chart(fig, use_container_width=True)

        # Track duration analysis
        track_durations = df.groupby('track_id')['frame_number'].agg(['min', 'max'])
        track_durations['duration'] = track_durations['max'] - track_durations['min']

        fig2 = px.histogram(track_durations, x='duration', 
                           title="Track Duration Distribution")
        st.plotly_chart(fig2, use_container_width=True)

    def render_export_options(self, tracking_results, filename):
        """Render export options"""
        st.subheader("📥 Export Results")

        # JSON export
        json_results = json.dumps(tracking_results, indent=2)
        st.download_button(
            label="📋 Download JSON Results",
            data=json_results,
            file_name=f"{Path(filename).stem}_tracking_results.json",
            mime="application/json"
        )

        # CSV export
        if tracking_results:
            df = pd.DataFrame(tracking_results)
            csv_data = df.to_csv(index=False)
            st.download_button(
                label="📊 Download CSV Results",
                data=csv_data,
                file_name=f"{Path(filename).stem}_tracking_results.csv",
                mime="text/csv"
            )

        # Summary report
        summary_report = self.generate_summary_report(tracking_results, filename)
        st.download_button(
            label="📄 Download Summary Report",
            data=summary_report,
            file_name=f"{Path(filename).stem}_summary_report.txt",
            mime="text/plain"
        )

    def render_detailed_results(self, tracking_results):
        """Render detailed results table"""
        if tracking_results:
            df = pd.DataFrame(tracking_results)
            st.dataframe(df, use_container_width=True)
        else:
            st.info("No detailed results available")

    def generate_summary_report(self, tracking_results, filename):
        """Generate text summary report"""
        if not tracking_results:
            return "No tracking results available"

        total_detections = len(tracking_results)
        unique_tracks = len(set(r['track_id'] for r in tracking_results))
        class_counts = {}
        for result in tracking_results:
            class_name = result['class_name']
            class_counts[class_name] = class_counts.get(class_name, 0) + 1

        report = f"""
Vehicle & Pedestrian Tracking Report
====================================

Video: {filename}
Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

Summary Statistics:
- Total Detections: {total_detections}
- Unique Tracks: {unique_tracks}
- Average Confidence: {np.mean([r['confidence'] for r in tracking_results]):.3f}

Class Distribution:
"""

        for class_name, count in class_counts.items():
            report += f"- {class_name}: {count} detections\n"

        return report

    def render_info_section(self):
        """Render information section"""
        st.markdown("### 📖 How It Works")

        with st.expander("🔍 Detection & Tracking Process"):
            st.markdown("""
            1. **Object Detection**: YOLOv8 segmentation model identifies vehicles and pedestrians
            2. **Multi-Object Tracking**: ByteTrack assigns unique IDs and tracks objects across frames
            3. **Association**: Advanced algorithms connect detections between frames
            4. **Export**: Results available in JSON, CSV, and summary formats
            """)

        with st.expander("🎯 Supported Classes"):
            st.markdown("""
            - **Vehicles**: Cars, trucks, buses, motorcycles
            - **Pedestrians**: People in various poses and scenarios
            """)

        with st.expander("⚙️ Technical Details"):
            st.markdown("""
            - **Model**: YOLOv8 Segmentation
            - **Tracker**: ByteTrack
            - **Input**: MP4, AVI, MOV, MKV videos
            - **Output**: JSON with track IDs, bounding boxes, timestamps
            """)

if __name__ == "__main__":
    app = VideoTrackingApp()
    app.main()
