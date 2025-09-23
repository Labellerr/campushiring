import os
import cv2
import numpy as np
from flask import Flask, request, redirect, url_for, send_from_directory, render_template_string
from werkzeug.utils import secure_filename
from ultralytics import YOLO
import yaml

# --- Configuration ---
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'processed_videos'
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv'} # Define allowed video extensions

# Ensure upload and output folders exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# --- Model Loading ---
# Define the path to the trained YOLO-Seg model weights
model_path = 'yolov8n-seg.pt' # Based on previous steps, this file exists

# Load the trained YOLO-Seg model once when the application starts
try:
    yolo_model = YOLO(model_path)
    print(f"YOLO-Seg model loaded successfully from '{model_path}'.")
except Exception as e:
    print(f"Error loading YOLO-Seg model: {e}")
    yolo_model = None # Set to None if loading fails

# --- Flask App Initialization ---
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # Limit file size to 16MB (example)
# Secret key for session (used by flash). Prefer setting via environment in production.
app.secret_key = os.environ.get('FLASK_SECRET_KEY', 'dev_secret_key_change_me')

# --- Helper Functions ---
def allowed_file(filename):
    """Checks if the uploaded file extension is allowed."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# --- HTML Content (for simplicity, keep inline or load from templates) ---
# Using render_template_string for simplicity in this notebook context.
# In a real application, use render_template with separate HTML files.
index_html_content = """
<!doctype html>
<html>
<head>
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <title>Video Processor</title>
        <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/modern-normalize/modern-normalize.min.css">
        <style>
                :root{--accent:#2b8cff;--muted:#666;--bg:#f7f9fb}
                body{font-family:Inter, system-ui, -apple-system, 'Segoe UI', Roboto, 'Helvetica Neue', Arial;margin:0;background:var(--bg);color:#222}
                .container{max-width:980px;margin:28px auto;padding:18px;background:#fff;border-radius:10px;box-shadow:0 6px 18px rgba(20,30,50,.06)}
                header{display:flex;align-items:center;justify-content:space-between;margin-bottom:12px}
                h1{font-size:20px;margin:0}
                form{display:flex;flex-wrap:wrap;gap:12px;align-items:center}
                .file-input{flex:1;display:flex;gap:8px;align-items:center}
                input[type=file]{display:none}
                .btn{background:var(--accent);color:#fff;border:none;padding:10px 14px;border-radius:8px;cursor:pointer}
                .btn:disabled{opacity:.6}
                .file-label{padding:10px 12px;border:1px dashed #ddd;border-radius:8px;background:#fafafa;flex:1}
                .meta{color:var(--muted);font-size:13px}
                .flash{padding:10px;border-radius:8px;margin:10px 0}
                .flash.error{background:#ffecec;color:#900}
                .flash.success{background:#e6ffed;color:#006400}
                .grid{display:grid;grid-template-columns:1fr 320px;gap:16px;margin-top:18px}
                .panel{padding:12px;border-radius:8px;border:1px solid #eef2f7;background:#fff}
                .files-list{list-style:none;padding:0;margin:0}
                .files-list li{display:flex;justify-content:space-between;padding:8px 6px;border-bottom:1px solid #f1f4f7}
                .spinner{display:none;align-items:center;gap:8px}
                .spinner.show{display:inline-flex}
        </style>
</head>
<body>
<div class="container">
    <header>
        <h1>Video Segmentation & Labeling</h1>
        <div class="meta">Drop a video or select to process — outputs appear on the right</div>
    </header>

    {% with messages = get_flashed_messages(with_categories=true) %}
        {% if messages %}
            {% for category, message in messages %}
                <div class="flash {{ category }}">{{ message }}</div>
            {% endfor %}
        {% endif %}
    {% endwith %}

    <form id="uploadForm" action="{{ url_for('upload_video') }}" method="post" enctype="multipart/form-data">
        <div class="file-input">
            <label class="file-label" id="fileLabel">Choose a video or drop it here</label>
            <input id="videoFileInput" type="file" name="videoFile" accept="video/*" required>
            <button type="button" id="chooseBtn" class="btn">Choose</button>
            <button id="submitBtn" class="btn" type="submit">Upload & Process</button>
            <div class="spinner" id="spinner"><svg width="18" height="18" viewBox="0 0 50 50"><circle cx="25" cy="25" r="20" stroke="#2b8cff" stroke-width="4" fill="none" stroke-linecap="round"/></svg><span class="meta">Processing...</span></div>
        </div>
    </form>

        <div style="margin-top:18px">
            <div class="panel">
                <h3 style="margin-top:0">How it works</h3>
                <p class="meta">Upload a video (mp4, mov, avi, mkv). The server will run segmentation/tracking and return a processed video with bounding boxes and labels.</p>
                {% if download_link %}
                    <div style="margin-top:12px"><strong>Download:</strong> <a href="{{ download_link }}">Download processed file</a></div>
                {% else %}
                    <div style="margin-top:12px" class="meta">After processing completes, a download link will appear here.</div>
                {% endif %}
            </div>
        </div>
</div>

<script>
    const chooseBtn = document.getElementById('chooseBtn');
    const fileInput = document.getElementById('videoFileInput');
    const fileLabel = document.getElementById('fileLabel');
    const spinner = document.getElementById('spinner');
    const submitBtn = document.getElementById('submitBtn');
    const form = document.getElementById('uploadForm');

    chooseBtn.addEventListener('click', ()=> fileInput.click());
    fileInput.addEventListener('change', ()=>{
        const f = fileInput.files[0];
        fileLabel.textContent = f ? f.name : 'Choose a video or drop it here';
    });

    form.addEventListener('submit', ()=>{
        submitBtn.disabled = true;
        spinner.classList.add('show');
    });
</script>

</body>
</html>
"""
# Add flash messages for user feedback
from flask import flash, get_flashed_messages

# --- Routes ---
@app.route('/')
def index():
    return render_template_string(index_html_content, download_link=None)

@app.route('/upload')
def upload_form():
    return redirect(url_for('index'))

@app.route('/upload_video', methods=['POST'])
def upload_video():
    global yolo_model # Declare yolo_model as global

    if 'videoFile' not in request.files:
        flash('No video file part in the request', 'error')
        return redirect(url_for('index'))

    file = request.files['videoFile']

    if file.filename == '':
        flash('No selected file', 'error')
        return redirect(url_for('index'))

    if not allowed_file(file.filename):
        flash('Invalid file type. Allowed types are: mp4, avi, mov, mkv', 'error')
        return redirect(url_for('index'))

    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

        try:
            file.save(filepath)
            print(f"File saved to: {filepath}")
        except IOError as e:
            flash(f"Error saving file: {e}", 'error')
            return redirect(url_for('index'))


        # Check if model loaded successfully
        if yolo_model is None:
            flash("Segmentation model not loaded. Cannot process video.", 'error')
            # Attempt to load model again as a fallback, though ideally loaded at startup
            try:
                yolo_model = YOLO(model_path)
                print("Attempted to reload YOLO-Seg model.")
                if yolo_model is None:
                     return redirect(url_for('index')) # Still failed to load
            except Exception as e:
                 flash(f"Failed to reload model: {e}", 'error')
                 return redirect(url_for('index'))


        # Initialize video capture and writer
        cap = cv2.VideoCapture(filepath)
        if not cap.isOpened():
            flash(f"Error: Could not open video file {filename} for processing.", 'error')
            os.remove(filepath) # Clean up uploaded file
            return redirect(url_for('index'))

        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        output_filename = f"processed_{filename}"
        output_filepath = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)

        # Try different codecs for compatibility
        fourcc_options = [cv2.VideoWriter_fourcc(*'mp4v'), cv2.VideoWriter_fourcc(*'XVID')]
        out = None
        for fourcc_code in fourcc_options:
             try:
                 out = cv2.VideoWriter(output_filepath, fourcc_code, fps, (frame_width, frame_height))
                 if out.isOpened():
                      print(f"VideoWriter opened successfully with codec {fourcc_code}")
                      break
                 else:
                      print(f"VideoWriter failed to open with codec {fourcc_code}")
             except Exception as e:
                  print(f"Error with codec {fourcc_code}: {e}")
                  out = None # Ensure out is None if it fails

        if out is None or not out.isOpened():
            flash("Error: Could not initialize video writer.", 'error')
            cap.release()
            os.remove(filepath) # Clean up uploaded file
            return redirect(url_for('index'))


        print("Starting video processing...")
        frame_count = 0
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frame_count += 1
                # print(f"Processing frame {frame_count}") # Optional: log frame processing

                # Perform YOLO-Seg prediction
                results = yolo_model(frame, stream=False, verbose=False)

                # Process results and overlay bounding boxes and labels (prefer rectangular bounding boxes)
                if results and len(results) > 0:
                    r = results[0]
                    # Choose color palette
                    def _color_for_id(id_val):
                        np.random.seed(int(id_val) & 0xFFFFFFFF)
                        return tuple(int(x) for x in np.random.randint(0, 255, size=3).tolist())

                    # If boxes are present, draw them
                    if hasattr(r, 'boxes') and r.boxes is not None and len(r.boxes) > 0:
                        for box in r.boxes:
                            # box.xyxy, box.conf, box.cls
                            try:
                                xyxy = box.xyxy[0].cpu().numpy() if hasattr(box.xyxy, 'cpu') else np.array(box.xyxy)
                                x1, y1, x2, y2 = map(int, xyxy)
                            except Exception:
                                # Fallback if format differs
                                coords = np.array(box.xyxy).flatten().astype(int)
                                x1, y1, x2, y2 = coords[:4]

                            conf = float(box.conf[0]) if hasattr(box, 'conf') and len(box.conf) > 0 else (float(box.conf) if hasattr(box, 'conf') else 0.0)
                            cls_id = int(box.cls[0]) if hasattr(box, 'cls') and len(box.cls) > 0 else (int(box.cls) if hasattr(box, 'cls') else 0)

                            color = _color_for_id(cls_id)

                            # Resolve original class name safely
                            orig_name = None
                            if hasattr(r, 'names') and r.names is not None:
                                try:
                                    orig_name = r.names[int(cls_id)]
                                except Exception:
                                    try:
                                        orig_name = r.names.get(int(cls_id), str(cls_id))
                                    except Exception:
                                        orig_name = str(cls_id)
                            else:
                                orig_name = str(cls_id)

                            # Map to simplified labels: pedestrian -> human, common vehicles -> vehicle
                            name_lower = str(orig_name).lower()
                            vehicle_keywords = {'car', 'truck', 'bus', 'motorbike', 'motorcycle', 'bicycle', 'van', 'train', 'boat'}
                            human_keywords = {'person', 'pedestrian', 'human'}

                            if any(k in name_lower for k in human_keywords):
                                mapped_name = 'human'
                            elif any(k in name_lower for k in vehicle_keywords):
                                mapped_name = 'vehicle'
                            else:
                                mapped_name = orig_name

                            label = f"{mapped_name}: {conf:.2f}"

                            # Draw rectangle
                            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                            # Draw label background
                            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                            cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 6, y1), color, -1)
                            cv2.putText(frame, label, (x1 + 3, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

                    # If no boxes but masks exist, use mask-based overlay (legacy behavior)
                    elif hasattr(r, 'masks') and r.masks is not None:
                        masks = r.masks.data.cpu().numpy()
                        overlay = frame.copy()
                        alpha = 0.5

                        for mask in masks:
                            contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                            cv2.drawContours(overlay, contours, -1, (0, 255, 0), -1)

                        frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

                # Write the processed frame
                out.write(frame)

            flash(f"Video processing completed. Processed {frame_count} frames.", 'success')

        except Exception as e:
             flash(f"An error occurred during video processing: {e}", 'error')
             print(f"Error during processing: {e}")
             # Clean up partially written output file if it exists
             if os.path.exists(output_filepath):
                 os.remove(output_filepath)

        finally:
            # Release video capture and writer
            # Safely release video capture
            try:
                if 'cap' in locals() and hasattr(cap, 'isOpened') and cap.isOpened():
                    cap.release()
            except Exception as e:
                print(f"Error releasing cap: {e}")

            # Safely release video writer if it was initialized
            try:
                if 'out' in locals() and out is not None and hasattr(out, 'isOpened') and out.isOpened():
                    out.release()
            except Exception as e:
                print(f"Error releasing out: {e}")
            # Keep the uploaded file for potential debugging or remove later
            # os.remove(filepath) # Uncomment to remove original uploaded file


        # Redirect to index page with download link
        download_link = url_for('download_file', filename=output_filename)
        return render_template_string(index_html_content, download_link=download_link)

    return "Error processing file", 500 # Should be caught by specific error handlers above

@app.route('/download/<filename>')
def download_file(filename):
    """Provides the processed video file for download."""
    try:
        # Ensure filename is safe to prevent directory traversal
        safe_filename = secure_filename(filename)
        return send_from_directory(app.config['OUTPUT_FOLDER'], safe_filename, as_attachment=True)
    except FileNotFoundError:
        flash("Processed file not found.", 'error')
        return redirect(url_for('index'))
    except Exception as e:
        flash(f"An error occurred during download: {e}", 'error')
        return redirect(url_for('index'))

# --- Requirements.txt generation ---
# This part is for documenting dependencies for deployment.
# In a real project, you'd typically use pip freeze > requirements.txt
# but for this notebook, we'll just list the main ones.
requirements_content = """
Flask
ultralytics
opencv-python
PyYAML
numpy
gunicorn # For production server
"""
with open('requirements.txt', 'w') as f:
    f.write(requirements_content.strip())

print("\nCreated 'requirements.txt' with basic dependencies.")


# --- Gunicorn Instructions ---
gunicorn_instructions = """
To run this Flask application using Gunicorn for production:

1. Make sure you have Gunicorn installed:
   pip install gunicorn

2. Save the Flask application code (including routes, model loading, etc.) into a Python file, e.g., 'app.py'.
   Ensure your Flask app instance is named 'app'.

3. Run the Gunicorn server from your terminal in the same directory as 'app.py':
   gunicorn -w 4 'app:app'

   -w 4: Specifies the number of worker processes (adjust based on your server's CPU cores).
   'app:app': Specifies the module (app.py) and the Flask application instance (app).

4. Access the application in your web browser at the server's address and port (default is http://127.0.0.1:8000).

Remember to handle environment variables and configuration securely in a production environment.
"""
print(gunicorn_instructions)

# Note: Running the Flask app directly with app.run() is not recommended for production.
# The code above defines the app and routes, and the requirements.txt and instructions.