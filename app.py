import streamlit as st
import os
import json
import tempfile
import cv2
from pathlib import Path
import traceback
import numpy as np

try:
    from ultralytics import YOLO
except ImportError:
    st.error("❌ ultralytics package not found. Please install it with: pip install ultralytics")
    st.stop()

def load_model(model_path):
    """Load YOLO model with error handling"""
    try:
        if not os.path.exists(model_path):
            st.error(f"❌ Model file '{model_path}' not found. Please ensure your trained weights file is in the same directory.")
            return None
        return YOLO(model_path)
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        return None

def process_video_frame_by_frame(model, input_path, output_path, confidence_threshold, progress_bar):
    """Process video frame by frame with tracking"""
    
    # Open video
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise Exception("Could not open video file")
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Setup video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    tracks = []
    frame_count = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Run tracking on frame
            results = model.track(frame, persist=True, conf=confidence_threshold, verbose=False)
            
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
                        if class_names:
                            label = class_names.get(cls_id, f"class_{cls_id}")
                        else:
                            label = "vehicle" if cls_id == 0 else "pedestrian"
                        
                        tracks.append({
                            "frame": frame_count,
                            "id": tid,
                            "class": label,
                            "class_id": cls_id,
                            "confidence": float(conf),
                            "bbox": [float(x) for x in bbox]
                        })
            else:
                annotated_frame = frame
            
            # Write frame
            out.write(annotated_frame)
            frame_count += 1
            
            # Update progress
            if frame_count % 10 == 0:  # Update every 10 frames
                progress = min(frame_count / total_frames, 1.0)
                progress_bar.progress(progress, f"Processing frame {frame_count}/{total_frames}")
    
    finally:
        cap.release()
        out.release()
    
    return tracks

def main():
    st.title("🚦 Vehicle & Pedestrian Tracking with YOLOv8 + ByteTrack")
    
    # Model loading
    model_path = "best.pt"
    model = load_model(model_path)
    
    if model is None:
        st.stop()
    
    st.success(f"✅ Model loaded successfully from '{model_path}'")
    
    # File upload
    uploaded_file = st.file_uploader(
        "Upload a video file", 
        type=["mp4", "avi", "mov", "mkv"],
        help="Supported formats: MP4, AVI, MOV, MKV"
    )
    
    if uploaded_file is not None:
        # Display file info
        st.info(f"📁 File: {uploaded_file.name} ({uploaded_file.size / 1024 / 1024:.1f} MB)")
        
        # Create temporary directory for processing
        with tempfile.TemporaryDirectory() as temp_dir:
            # Save uploaded video
            input_video_path = os.path.join(temp_dir, uploaded_file.name)
            with open(input_video_path, "wb") as f:
                f.write(uploaded_file.read())
            
            # Display original video
            st.subheader("📹 Original Video")
            st.video(input_video_path)
            
            # Tracking options
            st.subheader("⚙️ Tracking Options")
            col1, col2 = st.columns(2)
            
            with col1:
                confidence_threshold = st.slider(
                    "Confidence Threshold", 
                    min_value=0.1, 
                    max_value=1.0, 
                    value=0.5, 
                    step=0.1
                )
            
            with col2:
                processing_method = st.selectbox(
                    "Processing Method",
                    ["Frame by Frame (Recommended)", "Batch Processing"],
                    help="Frame by frame is more reliable but slower"
                )
            
            if st.button("🚀 Run Tracking", type="primary"):
                try:
                    # Create output paths
                    output_video_path = os.path.join(temp_dir, f"tracked_{uploaded_file.name}")
                    
                    if processing_method == "Frame by Frame (Recommended)":
                        st.info("🎬 Processing video frame by frame...")
                        progress_bar = st.progress(0, "Initializing...")
                        
                        # Process frame by frame
                        tracks = process_video_frame_by_frame(
                            model, 
                            input_video_path, 
                            output_video_path, 
                            confidence_threshold,
                            progress_bar
                        )
                        
                        progress_bar.progress(1.0, "✅ Processing complete!")
                        
                    else:
                        # Original batch method
                        with st.spinner("Processing video in batch mode..."):
                            results = model.track(
                                source=input_video_path,
                                save=True,
                                project=temp_dir,
                                name="tracking_output",
                                conf=confidence_threshold,
                                verbose=True
                            )
                            
                            # Find the output video
                            for root, dirs, files in os.walk(temp_dir):
                                for file in files:
                                    if file.endswith(('.mp4', '.avi', '.mov')) and 'tracking_output' in root:
                                        output_video_path = os.path.join(root, file)
                                        break
                            
                            tracks = []
                            for frame_id, r in enumerate(results):
                                if r.boxes is not None and hasattr(r.boxes, 'id') and r.boxes.id is not None:
                                    # Process tracking results...
                                    pass  # Add your existing processing logic here
                    
                    # Check if output video exists
                    if os.path.exists(output_video_path):
                        st.success(f"✅ Tracking complete! Found {len(set(track['id'] for track in tracks)) if tracks else 0} unique objects.")
                        
                        # Display results
                        st.subheader("🎯 Tracked Video")
                        st.video(output_video_path)
                        
                        # Save and display statistics
                        if tracks:
                            # Statistics
                            st.subheader("📊 Tracking Statistics")
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                st.metric("Total Detections", len(tracks))
                            
                            with col2:
                                unique_ids = len(set(track['id'] for track in tracks))
                                st.metric("Unique Objects", unique_ids)
                            
                            with col3:
                                classes = set(track['class'] for track in tracks)
                                st.metric("Classes Detected", len(classes))
                            
                            # Save JSON results
                            json_data = {
                                "metadata": {
                                    "total_detections": len(tracks),
                                    "unique_objects": len(set(track['id'] for track in tracks)),
                                    "confidence_threshold": confidence_threshold,
                                    "processing_method": processing_method
                                },
                                "tracks": tracks
                            }
                            
                            # Download buttons
                            st.subheader("💾 Downloads")
                            
                            # Download JSON
                            json_str = json.dumps(json_data, indent=2)
                            st.download_button(
                                label="📥 Download Tracking Results (JSON)",
                                data=json_str,
                                file_name="tracking_results.json",
                                mime="application/json"
                            )
                            
                            # Download video
                            with open(output_video_path, "rb") as f:
                                st.download_button(
                                    label="📥 Download Tracked Video",
                                    data=f,
                                    file_name=f"tracked_{uploaded_file.name}",
                                    mime="video/mp4"
                                )
                        else:
                            st.warning("⚠️ No objects were tracked. Try lowering the confidence threshold.")
                    
                    else:
                        st.error("❌ Could not create output video. Please check your model and input video.")
                        
                        # Debug information
                        st.subheader("🔍 Debug Information")
                        st.write("Files in temp directory:")
                        for root, dirs, files in os.walk(temp_dir):
                            for file in files:
                                st.code(f"📁 {os.path.relpath(os.path.join(root, file), temp_dir)}")
                
                except Exception as e:
                    st.error(f"❌ Error during tracking: {str(e)}")
                    st.error("Full error details:")
                    st.code(traceback.format_exc())

if __name__ == "__main__":
    main()