import streamlit as st
import cv2
import numpy as np
import json
import tempfile
import os
import time
from datetime import datetime
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
import io
import base64

os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # or '' for CPU only
import subprocess
import shutil

def ensure_web_compatible(input_path: str) -> str:
    """Transcode input_path to a web-friendly MP4 using ffmpeg if available.
    Returns the path to the web-compatible file (may be same as input_path on failure).
    """
    try:
        # Check ffmpeg availability
        result = subprocess.run(['ffmpeg', '-version'],
                              stdout=subprocess.DEVNULL,
                              stderr=subprocess.DEVNULL,
                              timeout=10)
        if result.returncode != 0:
            print("ffmpeg not available or not working")
            return input_path
    except Exception as e:
        print(f"ffmpeg check failed: {e}")
        return input_path

    out_dir = os.path.join(os.path.dirname(input_path), 'web')
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(input_path))[0]
    web_path = os.path.join(out_dir, f"{base}_web.mp4")

    # If already exists and newer, reuse
    try:
        if os.path.exists(web_path) and os.path.getmtime(web_path) >= os.path.getmtime(input_path):
            print(f"Reusing existing web-compatible video: {web_path}")
            return web_path
    except Exception:
        pass

    # Improved ffmpeg command for better web compatibility
    cmd = [
        'ffmpeg', '-y', '-i', input_path,
        '-c:v', 'libx264',           # H.264 codec - most compatible
        '-preset', 'fast',           # Fast preset for reasonable speed/quality
        '-crf', '23',                # Constant Rate Factor for quality
        '-profile:v', 'baseline',    # Baseline profile for maximum browser compatibility
        '-level', '3.0',             # Level 3.0 widely supported
        '-pix_fmt', 'yuv420p',       # Standard pixel format
        '-movflags', '+faststart',   # Enable fast start for web streaming
        '-c:a', 'aac',               # AAC audio codec
        '-b:a', '128k',              # Audio bitrate
        web_path
    ]
   
    try:
        print(f"Converting video to web-compatible format...")
        result = subprocess.run(cmd, check=True,
                              stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE,
                              timeout=300)  # 5 minute timeout
       
        if os.path.exists(web_path) and os.path.getsize(web_path) > 0:
            print(f"Successfully created web-compatible video: {web_path}")
            return web_path
        else:
            print("Web-compatible video creation failed - empty file")
            return input_path
           
    except subprocess.TimeoutExpired:
        print("ffmpeg conversion timed out")
        return input_path
    except Exception as e:
        print(f"ffmpeg transcode failed: {e}")
        if hasattr(e, 'stderr') and e.stderr:
            print(f"ffmpeg stderr: {e.stderr.decode()}")
        return input_path

def get_media_info(path: str) -> dict:
    """Return basic stream/container info using ffprobe if available.
    Returns a dict with 'format' and 'streams' keys or an empty dict on failure.
    """
    try:
        res = subprocess.run([
            'ffprobe', '-v', 'error', '-print_format', 'json', '-show_format', '-show_streams', path
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        info = json.loads(res.stdout.decode())
        return info
    except Exception as e:
        # ffprobe not available or failed
        return {}

# Import the real YOLO + ByteTracker implementation
from collections import OrderedDict
import torch
from ultralytics import YOLO

# ByteTracker Implementation
class STrack:
    """Single object track"""
    def __init__(self, tlwh, score, cls):
        self._tlwh = np.asarray(tlwh, dtype=np.float32)
        self.is_activated = False
        self.track_id = 0
        self.state = 'new'
       
        self.score = score
        self.cls = cls
        self.tracklet_len = 0
        self.start_frame = 0
       
    @property
    def tlwh(self):
        """Get current position in format (top left x, top left y, width, height)."""
        return self._tlwh.copy()
   
    @property
    def tlbr(self):
        """Convert bounding box to format (min x, min y, max x, max y)."""
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret
   
    @property
    def xyah(self):
        """Convert bounding box to format (center x, center y, aspect ratio, height)."""
        ret = self.tlwh.copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret
   
    def predict(self):
        """Predict next position (simple linear prediction)"""
        pass
   
    def update(self, new_track, frame_id):
        """Update track with new detection"""
        self.frame_id = frame_id
        self.tracklet_len += 1
       
        self._tlwh = new_track.tlwh
        self.score = new_track.score
        self.is_activated = True
        self.state = 'tracked'
   
    def activate(self, frame_id):
        """Activate a track"""
        self.track_id = self.next_id()
        self.tracklet_len = 0
        self.state = 'tracked'
        self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id
   
    def re_activate(self, new_track, frame_id):
        """Re-activate a lost track"""
        self._tlwh = new_track.tlwh
        self.tracklet_len = 0
        self.state = 'tracked'
        self.is_activated = True
        self.frame_id = frame_id
        self.score = new_track.score
   
    def mark_lost(self):
        """Mark track as lost"""
        self.state = 'lost'
   
    def mark_removed(self):
        """Mark track as removed"""
        self.state = 'removed'
   
    @staticmethod
    def next_id():
        STrack.track_id_count += 1
        return STrack.track_id_count

# Initialize track counter
STrack.track_id_count = 0

class BYTETracker:
    """Multi-object tracker using BYTE tracking algorithm"""
   
    def __init__(self, frame_rate=30, track_thresh=0.6, track_buffer=30, match_thresh=0.8):
        self.tracked_stracks = []  # type: list[STrack]
        self.lost_stracks = []     # type: list[STrack]
        self.removed_stracks = []  # type: list[STrack]
       
        self.frame_id = 0
        self.track_thresh = track_thresh
        self.match_thresh = match_thresh
        self.track_buffer = track_buffer
        self.buffer_size = int(frame_rate / 30.0 * track_buffer)
        self.max_time_lost = self.buffer_size
       
    def update(self, output_results, img_shape):
        """Update tracks with new detections"""
        self.frame_id += 1
        activated_starcks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []
       
        # Convert detections to STrack objects
        if output_results is not None and len(output_results) > 0:
            # High confidence detections
            high_dets = []
            low_dets = []
           
            for det in output_results:
                if len(det) >= 6:  # x1, y1, x2, y2, conf, cls
                    x1, y1, x2, y2, conf, cls = det[:6]
                    tlwh = [x1, y1, x2 - x1, y2 - y1]
                   
                    if conf >= self.track_thresh:
                        high_dets.append(STrack(tlwh, conf, int(cls)))
                    else:
                        low_dets.append(STrack(tlwh, conf, int(cls)))
           
            # Predict current positions of existing tracks
            for track in self.tracked_stracks:
                track.predict()
           
            # First association with high confidence detections
            unconfirmed = []
            tracked_stracks = []
           
            for track in self.tracked_stracks:
                if track.is_activated:
                    tracked_stracks.append(track)
                else:
                    unconfirmed.append(track)
           
            # Associate with high confidence detections
            dists = self.calculate_distances(tracked_stracks, high_dets)
            matches, u_track, u_detection = self.linear_assignment(dists, thresh=self.match_thresh)
           
            for itracked, idet in matches:
                track = tracked_stracks[itracked]
                det = high_dets[idet]
                track.update(det, self.frame_id)
                activated_starcks.append(track)
           
            # Second association with low confidence detections
            r_tracked_stracks = [tracked_stracks[i] for i in u_track if tracked_stracks[i].state == 'tracked']
            dists = self.calculate_distances(r_tracked_stracks, low_dets)
            matches, u_track, u_detection_second = self.linear_assignment(dists, thresh=0.5)
           
            for itracked, idet in matches:
                track = r_tracked_stracks[itracked]
                det = low_dets[idet]
                track.update(det, self.frame_id)
                activated_starcks.append(track)
           
            # Mark unmatched tracks as lost
            for it in u_track:
                track = r_tracked_stracks[it]
                track.mark_lost()
                lost_stracks.append(track)
           
            # Deal with unconfirmed tracks
            detections = [high_dets[i] for i in u_detection]
            dists = self.calculate_distances(unconfirmed, detections)
            matches, u_unconfirmed, u_detection = self.linear_assignment(dists, thresh=0.7)
           
            for itracked, idet in matches:
                unconfirmed[itracked].update(detections[idet], self.frame_id)
                activated_starcks.append(unconfirmed[itracked])
           
            for it in u_unconfirmed:
                track = unconfirmed[it]
                track.mark_removed()
                removed_stracks.append(track)
           
            # Initialize new tracks
            for inew in u_detection:
                track = detections[inew]
                if track.score < self.track_thresh:
                    continue
                track.activate(self.frame_id)
                activated_starcks.append(track)
       
        # Update state
        for track in self.lost_stracks:
            if self.frame_id - track.frame_id > self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)
       
        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == 'tracked']
        self.tracked_stracks.extend(activated_starcks)
        self.tracked_stracks.extend(refind_stracks)
        self.lost_stracks = [t for t in self.lost_stracks if t.state == 'lost']
        self.lost_stracks.extend(lost_stracks)
        self.removed_stracks.extend(removed_stracks)
       
        # Filter out removed tracks
        self.tracked_stracks, self.lost_stracks = self.remove_duplicate_stracks(
            self.tracked_stracks, self.lost_stracks)
       
        # Return active tracks
        output_stracks = [track for track in self.tracked_stracks if track.is_activated]
        return output_stracks
   
    def calculate_distances(self, tracks, detections):
        """Calculate IoU distances between tracks and detections"""
        # Return an array shaped (len(tracks), len(detections)) even if one side is zero
        if len(tracks) == 0 or len(detections) == 0:
            return np.empty((len(tracks), len(detections)))
       
        distances = np.zeros((len(tracks), len(detections)))
       
        for i, track in enumerate(tracks):
            for j, det in enumerate(detections):
                # Calculate IoU
                iou = self.calculate_iou(track.tlbr, det.tlbr)
                distances[i, j] = 1 - iou  # Convert IoU to distance
       
        return distances
   
    @staticmethod
    def calculate_iou(box1, box2):
        """Calculate Intersection over Union (IoU) of two bounding boxes"""
        # Get intersection rectangle
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
       
        if x2 <= x1 or y2 <= y1:
            return 0.0
       
        intersection = (x2 - x1) * (y2 - y1)
       
        # Calculate union
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection
       
        return intersection / union if union > 0 else 0.0
   
    def linear_assignment(self, cost_matrix, thresh):
        """Perform linear assignment using Hungarian algorithm (simplified version)"""
        if cost_matrix.size == 0:
            return [], list(range(cost_matrix.shape[0])), list(range(cost_matrix.shape[1]))
       
        # Simple greedy assignment for demonstration
        # In production, use scipy.optimize.linear_sum_assignment or lapjv
        matches = []
        unmatched_tracks = list(range(cost_matrix.shape[0]))
        unmatched_detections = list(range(cost_matrix.shape[1]))
       
        while len(unmatched_tracks) > 0 and len(unmatched_detections) > 0:
            # Find minimum cost
            min_cost = float('inf')
            min_track_idx = -1
            min_det_idx = -1
           
            for i in unmatched_tracks:
                for j in unmatched_detections:
                    if cost_matrix[i, j] < min_cost:
                        min_cost = cost_matrix[i, j]
                        min_track_idx = i
                        min_det_idx = j
           
            if min_cost <= thresh:
                matches.append([min_track_idx, min_det_idx])
                unmatched_tracks.remove(min_track_idx)
                unmatched_detections.remove(min_det_idx)
            else:
                break
       
        return matches, unmatched_tracks, unmatched_detections
   
    @staticmethod
    def remove_duplicate_stracks(stracksa, stracksb):
        """Remove duplicate tracks"""
        pdist = np.zeros((len(stracksa), len(stracksb)))
       
        for i, tracka in enumerate(stracksa):
            for j, trackb in enumerate(stracksb):
                pdist[i, j] = 1 - BYTETracker.calculate_iou(tracka.tlbr, trackb.tlbr)
       
        pairs = np.where(pdist < 0.15)
        dupa, dupb = [], []
       
        for i, j in zip(*pairs):
            timea = stracksa[i].frame_id - stracksa[i].start_frame
            timeb = stracksb[j].frame_id - stracksb[j].start_frame
           
            if timea > timeb:
                dupb.append(j)
            else:
                dupa.append(i)
       
        resa = [t for i, t in enumerate(stracksa) if i not in dupa]
        resb = [t for i, t in enumerate(stracksb) if i not in dupb]
       
        return resa, resb

# REAL YOLO + ByteTrack implementation (REPLACES MockVideoProcessor)
class YOLOByteTracker:
    def __init__(self, model_path="best.pt", conf_threshold=0.5, track_thresh=0.6):
        """
        Initialize YOLO-Seg model with ByteTrack
       
        Args:
            model_path: Path to your trained YOLO model
            conf_threshold: Confidence threshold for detections
            track_thresh: Tracking threshold
        """
        try:
            self.model = YOLO(model_path)
            if torch.cuda.is_available():
                self.model.to('cuda')
            else:
                self.model.to('cpu')
           
            self.conf_threshold = conf_threshold
            self.tracker = BYTETracker(track_thresh=track_thresh)
           
            # Get class names from model
            self.class_names = self.model.names
            print(f"Model loaded with classes: {self.class_names}")
           
            # Test model with dummy input to verify it works
            try:
                dummy_img = np.zeros((640, 640, 3), dtype=np.uint8)
                test_results = self.model(dummy_img, conf=0.1, verbose=False)
                print(f"Model test successful, detected {len(test_results)} results")
            except Exception as e:
                print(f"Model test failed: {e}")
                raise e
               
        except Exception as e:
            print(f"Error loading model: {e}")
            raise e
       
    def process_video(self, video_path, progress_callback=None, max_frames=None):
        """
        Process video with YOLO detection and ByteTrack tracking.
        This version uses a reliable intermediate AVI file before converting to a web-compatible MP4.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video file: {video_path}")

        # Get video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Limit frames if specified
        if max_frames and max_frames < total_frames:
            total_frames = max_frames

        # Setup output directory
        output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output')
        os.makedirs(output_dir, exist_ok=True)

        # Use a timestamp for unique filenames
        timestamp = int(time.time())
        
        # STEP 1: Create a reliable intermediate AVI file with XVID codec
        intermediate_avi_path = os.path.join(output_dir, f"tracked_intermediate_{timestamp}.avi")
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(intermediate_avi_path, fourcc, fps, (width, height))

        if not out.isOpened():
            cap.release()
            raise IOError("Cannot create intermediate video writer. Check if 'XVID' codec is available.")

        results = []
        frame_count = 0
        
        # Define colors (same as your original code)
        colors = {
            0: (0, 255, 0),    # Green for person
            2: (0, 0, 255),    # Blue for car
            # ... add other classes as needed
        }
        
        print(f"Processing video: {total_frames} frames at {fps:.2f} FPS")
        print(f"Intermediate output will be: {intermediate_avi_path}")

        try:
            while cap.isOpened() and frame_count < total_frames:
                ret, frame = cap.read()
                if not ret:
                    print(f"Could not read frame {frame_count}")
                    break
                
                output_frame = frame.copy()
                
                # --- YOUR YOLO INFERENCE AND TRACKING LOGIC ---
                # This part of your code is fine, no changes needed here.
                # (YOLO detections, tracker.update, drawing boxes, etc.)
                # ...
                # (Code from your original function for processing and drawing)
                # ...
                detections = self.model(frame, conf=self.conf_threshold, verbose=False)
                det_boxes = []
                if detections:
                    for detection in detections:
                        if hasattr(detection, 'boxes') and detection.boxes is not None:
                            boxes = detection.boxes
                            for i in range(len(boxes.xyxy)):
                                x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy()
                                conf = float(boxes.conf[i].cpu().numpy())
                                cls = int(boxes.cls[i].cpu().numpy())
                                if conf >= self.conf_threshold:
                                    det_boxes.append([x1, y1, x2, y2, conf, cls])
                
                online_targets = self.tracker.update(det_boxes, frame.shape[:2])
                
                for track in online_targets:
                    tlbr = track.tlbr
                    track_id = track.track_id
                    cls = track.cls
                    conf = track.score
                    x1, y1, x2, y2 = map(int, tlbr)
                    
                    if x1 >= 0 and y1 >= 0 and x2 < width and y2 < height:
                        w, h = x2 - x1, y2 - y1
                        result = {
                            'track_id': int(track_id),
                            'class': self.class_names.get(cls, f'class_{cls}'),
                            'frame_number': frame_count,
                            'confidence': float(conf),
                            'bbox': [x1, y1, w, h],
                            'center_x': int(x1 + w/2),
                            'center_y': int(y1 + h/2),
                            'area': int(w * h),
                            'timestamp': frame_count / fps
                        }
                        results.append(result)

                        color = colors.get(cls, (128, 128, 128))
                        cv2.rectangle(output_frame, (x1, y1), (x2, y2), color, 2)
                        label = f'ID:{track_id} {self.class_names.get(cls, "?")}'
                        cv2.putText(output_frame, label, (x1, y1 - 10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                # --- END OF YOUR LOGIC ---

                if output_frame is not None:
                    out.write(output_frame)
                
                frame_count += 1
                
                if progress_callback and frame_count % 10 == 0:
                    progress = frame_count / total_frames
                    progress_callback(progress, f"Processing frame {frame_count}/{total_frames}")

        finally:
            cap.release()
            out.release()
            cv2.destroyAllWindows()
            print("Intermediate AVI file writing complete.")

        # STEP 2: Verify intermediate file and convert to web-compatible MP4
        if not os.path.exists(intermediate_avi_path) or os.path.getsize(intermediate_avi_path) < 1000:
            raise RuntimeError("Failed to create a valid intermediate AVI file.")

        print(f"Intermediate file size: {os.path.getsize(intermediate_avi_path) / 1e6:.2f} MB")
        
        try:
            web_compatible_path = ensure_web_compatible(intermediate_avi_path)
            
            if web_compatible_path != intermediate_avi_path:
                print(f"Successfully created web-compatible video: {web_compatible_path}")
                # Clean up the intermediate file
                os.remove(intermediate_avi_path)
                return results, web_compatible_path
            else:
                # ffmpeg conversion failed, return the AVI path with a warning
                print("Warning: ffmpeg conversion failed. Returning the intermediate AVI file.")
                return results, intermediate_avi_path

        except Exception as e:
            print(f"An error occurred during final video conversion: {e}")
            # Return the intermediate file as a last resort
            return results, intermediate_avi_path

def main():
    # Page configuration
    st.set_page_config(
        page_title="YOLO-Seg + ByteTrack Video Tracker",
        page_icon="🎯",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Custom CSS for styling
    st.markdown("""
    <style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
    }
    .stats-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        color: white;
        margin-bottom: 1rem;
    }
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
        margin-bottom: 1rem;
    }
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 20px;
        padding: 0.5rem 2rem;
        font-weight: bold;
    }
    .upload-section {
        border: 2px dashed #667eea;
        border-radius: 10px;
        padding: 2rem;
        text-align: center;
        margin-bottom: 2rem;
    }
    </style>
    """, unsafe_allow_html=True)

    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🎯 YOLO-Seg + ByteTrack Video Tracker</h1>
        <p>Advanced Object Detection, Segmentation & Multi-Object Tracking</p>
    </div>
    """, unsafe_allow_html=True)

    # Initialize session state variables
    if 'processing_complete' not in st.session_state:
        st.session_state.processing_complete = False
    if 'tracking_results' not in st.session_state:
        st.session_state.tracking_results = []
    if 'video_stats' not in st.session_state:
        st.session_state.video_stats = {}

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
       
        # Model settings
        st.subheader("Model Configuration")
        model_path = st.text_input("Model Path", value="best.pt", help="Path to your trained YOLO model")
        confidence_threshold = st.slider("Confidence Threshold", 0.1, 1.0, 0.3, 0.05)  # Lowered default
        track_threshold = st.slider("Tracking Threshold", 0.1, 1.0, 0.5, 0.05)  # Lowered default
       
        # Processing settings
        st.subheader("Processing Settings")
        max_frames = st.number_input("Max Frames to Process", min_value=50, max_value=10000, value=1000, help="Limit processing for large videos")
       
        # Export settings
        st.subheader("Export Settings")
        include_metadata = st.checkbox("Include Metadata in JSON", value=True)
        include_timestamps = st.checkbox("Include Frame Timestamps", value=True)
       
        if st.session_state.processing_complete:
            st.success("✅ Processing Complete!")
           
            # Quick stats in sidebar
            if st.session_state.tracking_results:
                st.metric("Total Detections", len(st.session_state.tracking_results))
                unique_tracks = len(set(r['track_id'] for r in st.session_state.tracking_results))
                st.metric("Unique Tracks", unique_tracks)

    # Main content area
    col1, col2 = st.columns([1, 1])

    with col1:
        st.header("📹 Video Upload")
       
        # Video upload
        uploaded_file = st.file_uploader(
            "Choose a video file",
            type=['mp4', 'avi', 'mov', 'mkv'],
            help="Upload your video file for object tracking analysis"
        )
       
        if uploaded_file is not None:
            # Save uploaded file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
                tmp_file.write(uploaded_file.read())
                video_path = tmp_file.name
           
            # Display video info
            st.success(f"✅ Video uploaded: {uploaded_file.name}")
           
            # Video details
            cap = cv2.VideoCapture(video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            duration = total_frames / fps if fps > 0 else 0
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
           
            # Video stats
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.metric("Duration", f"{duration:.1f}s")
            with col_b:
                st.metric("Frames", total_frames)
            with col_c:
                st.metric("FPS", f"{fps:.1f}")
           
            # Show original video
            st.video(uploaded_file)
           
            # Process button
            if st.button("🚀 Start Tracking", type="primary"):
                # Clear previous results
                st.session_state.processing_complete = False
                st.session_state.tracking_results = []
                st.session_state.video_stats = {}
               
                with st.spinner("Initializing YOLO-Seg and ByteTrack..."):
                    try:
                        # Initialize processor with REAL YOLO + ByteTrack
                        processor = YOLOByteTracker(
                            model_path=model_path,
                            conf_threshold=confidence_threshold,
                            track_thresh=track_threshold
                        )
                       
                        # Progress tracking
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                       
                        start_time = time.time()
                       
                        def progress_callback(progress, message):
                            progress_bar.progress(progress)
                            status_text.text(message)
                       
                        # Process video with REAL YOLO + ByteTrack
                        results, tracked_video_path = processor.process_video(
                            video_path,
                            progress_callback,
                            max_frames
                        )
                       
                        processing_time = time.time() - start_time
                       
                        # Verify results
                        if not os.path.exists(tracked_video_path):
                            st.error(f"❌ Tracked video file not found at: {tracked_video_path}")
                            return
                       
                        # Store results in session state
                        st.session_state.tracking_results = results
                        st.session_state.processing_complete = True
                        # Try to ensure web-friendly path for display
                        try:
                            display_path = ensure_web_compatible(tracked_video_path)
                        except Exception:
                            display_path = tracked_video_path

                        st.session_state.video_stats = {
                            'total_frames': total_frames,
                            'fps': fps,
                            'duration': duration,
                            'resolution': f"{width}x{height}",
                            'processing_time': processing_time,
                            'total_detections': len(results),
                            'unique_tracks': len(set(r['track_id'] for r in results)) if results else 0,
                            'tracked_video_path': os.path.abspath(display_path)
                        }
                       
                        progress_bar.progress(1.0)
                        status_text.text("✅ Processing complete!")
                       
                        st.success(f"🎉 Video processed successfully in {processing_time:.2f} seconds!")
                        st.balloons()
                    except Exception as e:
                        st.error(f"❌ Error processing video: {str(e)}")
                        if "best.pt" in str(e):
                            st.error("Please check if your model file 'best.pt' exists and is valid.")
                        progress_bar.empty()
                        status_text.empty()

    with col2:
        st.header("📊 Results & Analytics")
    
        if st.session_state.processing_complete:
            # Display tracked video if available
            if 'tracked_video_path' in st.session_state.video_stats:
                # This subheader can be changed or removed
                st.subheader("🎯 Download Tracked Video") 
                
                video_path = st.session_state.video_stats['tracked_video_path']

                if os.path.exists(video_path):
                    try:
                        # Read the video file into bytes for the download button
                        with open(video_path, "rb") as video_file:
                            video_bytes = video_file.read()
                        
                        # REMOVED: The st.video() player is now gone.
                        
                        # The download button remains
                        st.download_button(
                            label="📥 Download Tracked Video",
                            data=video_bytes,
                            file_name=f"tracked_video_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4",
                            mime="video/mp4"
                        )

                        # UPDATED: Changed the success message to reflect the change.
                        st.success("✅ Tracked video is ready for download!")

                    except Exception as e:
                        st.error(f"Error preparing video for download: {e}")
                else:
                    st.error(f"❌ Video file not found at path: {video_path}")
            
            # Key metrics (This part remains the same)
            st.subheader("📈 Key Metrics")
            stats = st.session_state.video_stats
            col_x, col_y, col_z = st.columns(3)
            
            with col_x:
                st.metric(
                    "Total Detections", 
                    stats['total_detections'],
                    delta=f"{stats['total_detections']/max(stats['total_frames'], 1):.1f} per frame"
                )
            
            with col_y:
                st.metric(
                    "Unique Tracks", 
                    stats['unique_tracks'],
                    delta=f"Avg {stats['total_detections']/max(stats['unique_tracks'], 1):.1f} det/track"
                )
            
            with col_z:
                st.metric(
                    "Processing Speed", 
                    f"{stats['processing_time']:.2f}s",
                    delta=f"{stats['total_frames']/max(stats['processing_time'], 0.01):.1f} FPS"
                )
        
        else:
            st.info("👆 Upload a video and click 'Start Tracking' to see results here")
            st.markdown("""
            ### 🔧 Troubleshooting Tips:
            - Make sure your `best.pt` model file exists in the same directory.
            - Check that your video file is in a supported format (MP4, AVI, MOV, MKV).
            - Lower the confidence threshold if you're not getting any detections.
            - Reduce max frames for faster processing during testing.
            """)

    # Results section (full width)
    if st.session_state.processing_complete and st.session_state.tracking_results:
       
        st.markdown("---")
        st.header("🔍 Detailed Analysis")
       
        # Tabs for different views
        tab1, tab2, tab3, tab4 = st.tabs(["📋 Detection Table", "📊 Visualizations", "🎯 Track Analysis", "💾 Export"])
       
        with tab1:
            st.subheader("📋 Detection Table")
           
            # Convert results to DataFrame
            df = pd.DataFrame(st.session_state.tracking_results)
           
            # Display summary statistics
            col_summary1, col_summary2, col_summary3, col_summary4 = st.columns(4)
            with col_summary1:
                st.metric("Total Rows", len(df))
            with col_summary2:
                st.metric("Unique Classes", df['class'].nunique())
            with col_summary3:
                st.metric("Frame Range", f"{df['frame_number'].min()}-{df['frame_number'].max()}")
            with col_summary4:
                avg_conf = df['confidence'].mean()
                st.metric("Avg Confidence", f"{avg_conf:.3f}")
           
            # Filter options
            st.subheader("🔍 Filters")
            col_filter1, col_filter2, col_filter3 = st.columns(3)
           
            with col_filter1:
                selected_classes = st.multiselect(
                    "Select Classes",
                    options=df['class'].unique(),
                    default=df['class'].unique()
                )
           
            with col_filter2:
                conf_range = st.slider(
                    "Confidence Range",
                    float(df['confidence'].min()),
                    float(df['confidence'].max()),
                    (float(df['confidence'].min()), float(df['confidence'].max())),
                    step=0.01
                )
           
            with col_filter3:
                frame_range = st.slider(
                    "Frame Range",
                    int(df['frame_number'].min()),
                    int(df['frame_number'].max()),
                    (int(df['frame_number'].min()), int(df['frame_number'].max()))
                )
           
            # Apply filters
            filtered_df = df[
                (df['class'].isin(selected_classes)) &
                (df['confidence'] >= conf_range[0]) &
                (df['confidence'] <= conf_range[1]) &
                (df['frame_number'] >= frame_range[0]) &
                (df['frame_number'] <= frame_range[1])
            ]
           
            st.write(f"Showing {len(filtered_df)} of {len(df)} detections")
           
            # Display filtered results
            st.dataframe(filtered_df, use_container_width=True)
           
            # Download CSV button
            csv_download = filtered_df.to_csv(index=False)
            st.download_button(
                "📥 Download Filtered CSV",
                csv_download,
                f"tracking_results_filtered_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                "text/csv",
                key="download-filtered-csv"
            )
       
        with tab2:
            st.subheader("📊 Visualizations")
           
            if st.session_state.tracking_results:
                df = pd.DataFrame(st.session_state.tracking_results)
               
                # 1. Detections per frame
                st.subheader("📈 Detections Over Time")
                detections_per_frame = df.groupby('frame_number').size().reset_index(name='detections')
               
                fig1 = px.line(
                    detections_per_frame,
                    x='frame_number',
                    y='detections',
                    title='Number of Detections per Frame',
                    labels={'frame_number': 'Frame Number', 'detections': 'Number of Detections'}
                )
                fig1.update_traces(line_color='#667eea', line_width=2)
                st.plotly_chart(fig1, use_container_width=True)
               
                # 2. Class distribution
                st.subheader("🏷️ Class Distribution")
                class_counts = df['class'].value_counts().reset_index()
                class_counts.columns = ['class', 'count']
               
                fig2 = px.pie(
                    class_counts,
                    values='count',
                    names='class',
                    title='Distribution of Detected Classes'
                )
                st.plotly_chart(fig2, use_container_width=True)
               
                # 3. Confidence distribution
                st.subheader("🎯 Confidence Distribution")
                fig3 = px.histogram(
                    df,
                    x='confidence',
                    nbins=30,
                    title='Distribution of Detection Confidence Scores',
                    labels={'confidence': 'Confidence Score', 'count': 'Frequency'}
                )
                fig3.update_traces(marker_color='#f093fb')
                st.plotly_chart(fig3, use_container_width=True)
               
                # 4. Track duration analysis
                st.subheader("⏱️ Track Duration Analysis")
                track_duration = df.groupby('track_id')['frame_number'].agg(['min', 'max']).reset_index()
                track_duration['duration'] = track_duration['max'] - track_duration['min'] + 1
               
                fig4 = px.histogram(
                    track_duration,
                    x='duration',
                    nbins=20,
                    title='Distribution of Track Durations (frames)',
                    labels={'duration': 'Track Duration (frames)', 'count': 'Number of Tracks'}
                )
                fig4.update_traces(marker_color='#764ba2')
                st.plotly_chart(fig4, use_container_width=True)
               
            else:
                st.info("No tracking results found. Please process a video first.")
       
        with tab3:
            st.subheader("🎯 Track Analysis")
           
            if st.session_state.tracking_results:
                df = pd.DataFrame(st.session_state.tracking_results)
               
                # Track ID selection
                track_ids = sorted(df['track_id'].unique())
               
                col_track1, col_track2 = st.columns(2)
                with col_track1:
                    selected_track_id = st.selectbox(
                        "Select Track ID",
                        options=track_ids,
                        index=0
                    )
               
                with col_track2:
                    # Show track summary
                    track_data = df[df['track_id'] == selected_track_id]
                    st.metric("Track Length", f"{len(track_data)} detections")
               
                # Filter results for selected track ID
                track_df = df[df['track_id'] == selected_track_id].sort_values('frame_number')
               
                if not track_df.empty:
                    # Track statistics
                    st.subheader(f"📊 Statistics for Track ID {selected_track_id}")
                   
                    col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
                    with col_stat1:
                        st.metric("First Frame", int(track_df['frame_number'].min()))
                    with col_stat2:
                        st.metric("Last Frame", int(track_df['frame_number'].max()))
                    with col_stat3:
                        st.metric("Avg Confidence", f"{track_df['confidence'].mean():.3f}")
                    with col_stat4:
                        st.metric("Class", track_df['class'].iloc[0])
                   
                    # Visualizations for selected track
                    col_vis1, col_vis2 = st.columns(2)
                   
                    with col_vis1:
                        # Track trajectory (center coordinates)
                        if 'center_x' in track_df.columns and 'center_y' in track_df.columns:
                            fig_traj = px.line(
                                track_df,
                                x='center_x',
                                y='center_y',
                                title=f'Track Trajectory (ID: {selected_track_id})',
                                labels={'center_x': 'X Coordinate', 'center_y': 'Y Coordinate'}
                            )
                            fig_traj.update_traces(line_color='#667eea', line_width=3)
                            fig_traj.update_yaxes(autorange="reversed")  # Flip Y axis for image coordinates
                            st.plotly_chart(fig_traj, use_container_width=True)
                   
                    with col_vis2:
                        # Confidence over time
                        fig_conf = px.line(
                            track_df,
                            x='frame_number',
                            y='confidence',
                            title=f'Confidence Over Time (ID: {selected_track_id})',
                            labels={'frame_number': 'Frame Number', 'confidence': 'Confidence Score'}
                        )
                        fig_conf.update_traces(line_color='#f093fb', line_width=3)
                        st.plotly_chart(fig_conf, use_container_width=True)
                   
                    # Detailed track data table
                    st.subheader("📋 Track Details")
                    st.dataframe(track_df, use_container_width=True)
                   
                    # Export track data
                    track_csv = track_df.to_csv(index=False)
                    st.download_button(
                        f"📥 Download Track {selected_track_id} CSV",
                        track_csv,
                        f"track_{selected_track_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        "text/csv",
                        key=f"download-track-csv-{selected_track_id}"
                    )
               
                else:
                    st.info(f"No data found for Track ID {selected_track_id}.")
           
            else:
                st.info("No tracking results available for analysis.")
       
        with tab4:
            st.subheader("💾 Export Results")
           
            # Configuration summary
            st.subheader("⚙️ Processing Configuration")
            if st.session_state.video_stats:
                config_data = {
                    "Model Path": st.session_state.video_stats.get('model_path', 'N/A'),
                    "Confidence Threshold": st.session_state.video_stats.get('conf_threshold', 'N/A'),
                    "Tracking Threshold": st.session_state.video_stats.get('track_thresh', 'N/A'),
                    "Max Frames Processed": st.session_state.video_stats.get('max_frames', 'N/A'),
                    "Total Processing Time": f"{st.session_state.video_stats.get('processing_time', 0):.2f}s"
                }
               
                config_df = pd.DataFrame(list(config_data.items()), columns=['Parameter', 'Value'])
                st.table(config_df)
           
            # JSON export with metadata
            st.subheader("📄 JSON Export")
            export_data = {
                'metadata': {
                    'export_timestamp': datetime.now().isoformat(),
                    'processing_config': st.session_state.video_stats,
                    'total_detections': len(st.session_state.tracking_results),
                    'unique_tracks': len(set(r['track_id'] for r in st.session_state.tracking_results)) if st.session_state.tracking_results else 0
                },
                'detections': st.session_state.tracking_results
            }
           
            json_str = json.dumps(export_data, indent=2)
           
            st.download_button(
                label="📥 Download Complete Results (JSON)",
                data=json_str,
                file_name=f"tracking_results_complete_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
           
            # Statistics export
            st.subheader("📊 Statistics Export")
            if st.session_state.tracking_results:
                df = pd.DataFrame(st.session_state.tracking_results)
               
                # Generate summary statistics
                summary_stats = {
                    'total_detections': len(df),
                    'unique_tracks': df['track_id'].nunique(),
                    'unique_classes': df['class'].nunique(),
                    'frame_range': f"{df['frame_number'].min()}-{df['frame_number'].max()}",
                    'confidence_range': f"{df['confidence'].min():.3f}-{df['confidence'].max():.3f}",
                    'average_confidence': df['confidence'].mean(),
                    'detections_per_frame_avg': len(df) / df['frame_number'].nunique(),
                    'class_distribution': df['class'].value_counts().to_dict(),
                    'track_lengths': df.groupby('track_id').size().describe().to_dict()
                }
               
                summary_json = json.dumps(summary_stats, indent=2)
               
                st.download_button(
                    label="📥 Download Summary Statistics (JSON)",
                    data=summary_json,
                    file_name=f"tracking_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )

    # Reset button
    if st.session_state.processing_complete:
        st.markdown("---")
        col_reset1, col_reset2, col_reset3 = st.columns([1, 1, 1])
        with col_reset2:
            if st.button("🔄 Reset Application", type="secondary", use_container_width=True):
                # Clean up temporary files
                try:
                    if 'tracked_video_path' in st.session_state.video_stats:
                        tracked_path = st.session_state.video_stats['tracked_video_path']
                        if os.path.exists(tracked_path):
                            os.unlink(tracked_path)
                except:
                    pass
               
                # Clear session state
                for key in list(st.session_state.keys()):
                    del st.session_state[key]
                st.rerun()

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 20px;">
    <p>Built with ❤️ using Streamlit • YOLO-Seg • ByteTrack</p>
    <p>For assignment: End-to-End Image Segmentation & Object Tracking Pipeline</p>
</div>
""", unsafe_allow_html=True)

if __name__ == "__main__":
    main()