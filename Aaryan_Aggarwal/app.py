
import streamlit as st
import os
import tempfile
import json

try:
    from video_tracker import track_video
    TRACKER_AVAILABLE = True
except ImportError as e:
    st.error(f"Error importing video_tracker: {e}")
    TRACKER_AVAILABLE = False


st.set_page_config(
    page_title="Vehicle & Pedestrian Tracker",
    page_icon="🚦",
    layout="wide"
)

st.title("🚦 Vehicle and Pedestrian Tracking with YOLOv8 & ByteTrack")
st.markdown("Upload a video to track vehicles and pedestrians using YOLOv8 with ByteTrack tracking")

MODEL_WEIGHTS_PATH = "best.pt"

if not TRACKER_AVAILABLE:
    st.error("Tracking functionality not available. Please check video_tracker.py file.")
elif not os.path.exists(MODEL_WEIGHTS_PATH):
    st.error(f"❌ Model weights file not found at '{MODEL_WEIGHTS_PATH}'")
else:
    st.success(f"✅ Model found at: {MODEL_WEIGHTS_PATH}")
    
    uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "mov", "avi", "mkv"])

    if uploaded_file is not None:
        
        file_size = uploaded_file.size / (1024 * 1024)  # Convert to MB
        st.info(f"Uploaded video: {uploaded_file.name} ({file_size:.2f} MB)")
        
      
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tfile:
            tfile.write(uploaded_file.getbuffer())
            input_video_path = tfile.name

        st.subheader("Original Video")
        st.video(input_video_path)
        
        if st.button("🎯 Start Tracking", type="primary"):
            
            output_video_path = f"tracked_{uploaded_file.name}"
            results_json_path = "tracking_results.json"
            
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            
            with st.spinner("Processing video... This may take a few minutes depending on video length."):
                success, message = track_video(
                    input_video_path, 
                    output_video_path, 
                    MODEL_WEIGHTS_PATH,
                    results_json_path
                )
            
            if success:
                progress_bar.progress(100)
                status_text.success("✅ Processing Complete!")
                
                
                st.success("🎉 Tracking completed successfully!")
                
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Tracked Video")
                    if os.path.exists(output_video_path):
                        st.video(output_video_path)
                    else:
                        st.error("Output video file not found")
                
                with col2:
                    st.subheader("Tracking Results")
                    
                    
                    if os.path.exists(results_json_path):
                        with open(results_json_path, 'r') as f:
                            results_data = json.load(f)
                        
                        
                        if isinstance(results_data, dict):
                            
                            video_info = results_data.get('video_info', {})
                            tracking_results = results_data.get('tracking_results', [])
                        else:
                            
                            video_info = {}
                            tracking_results = results_data
                        
                        
                        total_frames = len(tracking_results) if tracking_results else 0
                        
                        
                        all_objects = []
                        for frame in tracking_results:
                            if isinstance(frame, dict) and 'objects' in frame:
                                all_objects.extend(frame['objects'])
                        
                        if all_objects:
                            unique_objects = len(set(obj['id'] for obj in all_objects))
                            total_detections = len(all_objects)
                            
                            
                            class_counts = {}
                            for obj in all_objects:
                                cls = obj['class']
                                class_counts[cls] = class_counts.get(cls, 0) + 1
                            
                            st.metric("Total Frames Processed", total_frames)
                            st.metric("Unique Objects Tracked", unique_objects)
                            st.metric("Total Detections", total_detections)
                            
                            st.subheader("Object Distribution")
                            for cls, count in class_counts.items():
                                st.metric(f"{cls.title()}", count)
                        else:
                            st.metric("Total Frames Processed", total_frames)
                            st.info("No objects detected in the video")
                
                
                st.markdown("---")
                st.subheader("📥 Download Results")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    if os.path.exists(results_json_path):
                        with open(results_json_path, "rb") as f:
                            st.download_button(
                                label="Download Tracking Results (JSON)",
                                data=f,
                                file_name="tracking_results.json",
                                mime="application/json",
                                help="Contains frame-by-frame tracking data"
                            )
                
                with col2:
                    if os.path.exists(output_video_path):
                        with open(output_video_path, "rb") as f:
                            st.download_button(
                                label="Download Tracked Video",
                                data=f,
                                file_name=output_video_path,
                                mime="video/mp4",
                                help="Video with bounding boxes and tracking IDs"
                            )
                
                
                try:
                    os.unlink(input_video_path)
                    if os.path.exists(output_video_path):
                        os.remove(output_video_path)
                    if os.path.exists(results_json_path):
                        os.remove(results_json_path)
                except Exception as e:
                    st.warning(f"Could not clean up temporary files: {e}")
            
            else:
                st.error(f"❌ Processing failed: {message}")
                
                try:
                    os.unlink(input_video_path)
                except:
                    pass


st.markdown("---")
st.subheader("📋 Instructions")
st.markdown("""
1. *Upload Video*: Select a video file (MP4, MOV, AVI, MKV)
2. *Start Tracking*: Click the 'Start Tracking' button
3. *Wait for Processing*: The app will process each frame (this may take some time)
4. *View Results*: Watch the tracked video and see statistics
5. *Download*: Save the results as JSON or the processed video

*Features*:
- Real-time object tracking with YOLOv8
- ByteTrack algorithm for consistent ID assignment
- Frame-by-frame tracking data in JSON format
- Video with bounding boxes and object IDs
- Detailed statistics and object counts
""")


with st.sidebar.expander("Model Information"):
    if os.path.exists(MODEL_WEIGHTS_PATH):
        st.write(f"*Model*: {MODEL_WEIGHTS_PATH}")
        st.write("*Tracker*: ByteTrack")
        st.write("*Classes*: Vehicles and Pedestrians")


with st.sidebar.expander("Debug Info"):
    st.write(f"Current directory: {os.getcwd()}")
    st.write("Files in directory:")
    for file in os.listdir('.'):
        st.write(f"- {file}")