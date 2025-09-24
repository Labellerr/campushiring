import streamlit as st
import tempfile
import os
from pathlib import Path
from ultralytics import YOLO

def main():
    """
    This is the main function that creates the Streamlit application.
    It handles file uploads, video processing, and displaying the results.
    """
    
    # Set the page title and a header
    st.set_page_config(page_title="Video Object Tracking", layout="wide")
    st.title("Video Object Tracking with YOLOv8 and ByteTrack")
    st.markdown("Upload a video and see it annotated with real-time object detection and tracking.")
    
    # --- File Uploader Section ---
    uploaded_file = st.file_uploader(
        "Choose a video file",
        type=['mp4', 'mov', 'avi', 'mkv'],
        help="Supported formats: .mp4, .mov, .avi, .mkv"
    )

    if uploaded_file is not None:
        # Create a temporary directory to save the uploaded file and the output
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir_path = Path(temp_dir)
            
            # Save the uploaded file to the temporary directory
            video_path = temp_dir_path / uploaded_file.name
            with open(video_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            st.success("File uploaded successfully! Processing video...")
            
            # Use a Streamlit spinner for visual feedback
            with st.spinner('Annotating video... This may take a few minutes.'):
                
                try:
                    # --- Video Processing Section ---
                    # Load a pre-trained YOLOv8 model
                    # You can change 'yolov8n.pt' to another model (e.g., 'yolov8s.pt' for better accuracy)
                    model = YOLO('yolov8n.pt')
                    
                    # Run object tracking with ByteTrack on the video
                    # The `tracker` argument specifies the tracking algorithm
                    # `save=True` saves the annotated video to the specified project directory
                    results = model.track(
                        source=str(video_path), 
                        tracker="bytetrack.yaml", 
                        save=True,
                        project=str(temp_dir_path),  # Save output to our temporary directory
                        name=f"annotated_{video_path.stem}"
                    )
                    
                    st.success("Video processing complete!")

                    # --- Display Annotated Video ---
                    # The annotated video is now saved in the temp_dir_path
                    output_video_dir = temp_dir_path / f"annotated_{video_path.stem}"
                    
                    # Find the video file in the output directory
                    output_video_path = next(output_video_dir.glob("*.mp4"))
                    
                    st.header("Annotated Video")
                    st.video(str(output_video_path))

                    # --- Add Download Button ---
                    # Read the annotated video file as bytes to prepare for download
                    with open(output_video_path, "rb") as file:
                        btn = st.download_button(
                            label="Download Annotated Video",
                            data=file,
                            file_name=f"annotated_{uploaded_file.name}",
                            mime="video/mp4"
                        )
                    
                except Exception as e:
                    st.error(f"An error occurred during video processing: {e}")
                    st.warning("Please ensure you have the required libraries installed and the video file is valid.")
                
    else:
        st.info("Please upload a video to get started.")

if __name__ == "__main__":
    main()
