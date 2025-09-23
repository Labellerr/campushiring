from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, send_file
import os
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from datetime import datetime
try:
    from flask_bootstrap5 import Bootstrap5 as _Bootstrap
except Exception:
    _Bootstrap = None

import markdown
from config import Config
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__)
app.config.from_object(Config)

# Initialize extensions
db = SQLAlchemy(app)
login_manager = LoginManager(app)

# Configure login
login_manager.login_view = 'login'
login_manager.login_message_category = 'info'

# Models
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password_hash = db.Column(db.String(120), nullable=False)
    entries = db.relationship('JournalEntry', backref='author', lazy=True)

class JournalEntry(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    content = db.Column(db.Text, nullable=False)
    created_at = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    tags = db.relationship('Tag', secondary='entry_tags')

class Tag(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(50), nullable=False)

# Association table for Entry-Tag relationship
entry_tags = db.Table('entry_tags',
    db.Column('entry_id', db.Integer, db.ForeignKey('journal_entry.id')),
    db.Column('tag_id', db.Integer, db.ForeignKey('tag.id'))
)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Routes
@app.route('/')
def index():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    return render_template('index.html')

@app.route('/dashboard')
@login_required
def dashboard():
    entries = JournalEntry.query.filter_by(user_id=current_user.id).order_by(JournalEntry.created_at.desc()).all()
    
    # Calculate statistics
    total_entries = len(entries)
    total_tags = len(set([tag.name for entry in entries for tag in entry.tags]))
    recent_entries = entries[:5] if entries else []
    
    return render_template('dashboard.html', 
                           entries=recent_entries,
                           total_entries=total_entries,
                           total_tags=total_tags,
                           user=current_user)

@app.route('/new_entry', methods=['GET', 'POST'])
@login_required
def new_entry():
    if request.method == 'POST':
        content = request.form.get('content')
        tags = request.form.get('tags', '').split(',')
        
        entry = JournalEntry(content=content, user_id=current_user.id)
        
        # Handle tags
        for tag_name in tags:
            tag_name = tag_name.strip()
            if tag_name:
                tag = Tag.query.filter_by(name=tag_name).first()
                if not tag:
                    tag = Tag(name=tag_name)
                    db.session.add(tag)
                entry.tags.append(tag)
        
        db.session.add(entry)
        db.session.commit()
        flash('Entry saved successfully!', 'success')
        return redirect(url_for('dashboard'))
    
    return render_template('new_entry.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        user = User.query.filter_by(username=username).first()
        
        if user and check_password_hash(user.password_hash, password):
            login_user(user)
            return redirect(url_for('dashboard'))
        flash('Invalid username or password', 'error')
    
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        if User.query.filter_by(username=username).first():
            flash('Username already exists', 'error')
            return redirect(url_for('register'))
        
        user = User(username=username, password_hash=generate_password_hash(password))
        db.session.add(user)
        db.session.commit()
        
        flash('Registration successful! Please login.', 'success')
        return redirect(url_for('login'))
    
    return render_template('register.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('index'))

# --- Demo page ---
@app.route('/video')
def video_demo():
    return render_template('video_demo.html')

# --- YOLO Segmentation + ByteTrack API ---
@app.route('/api/segment', methods=['POST'])
def api_segment():
    # Accept file upload or path
    file = request.files.get('file')
    image_path = request.form.get('image_path')
    conf = float(request.form.get('conf', 0.15))
    iou = float(request.form.get('iou', 0.5))
    imgsz = int(request.form.get('imgsz', 416))
    device = request.form.get('device', 'cpu')

    if file:
        upload_dir = app.config.get('UPLOAD_DIR', 'uploads')
        os.makedirs(upload_dir, exist_ok=True)
        save_path = os.path.join(upload_dir, file.filename)
        file.save(save_path)
        image_path = save_path

    if not image_path:
        return jsonify({"error": "No image provided"}), 400

    from yolo_service import run_segment
    out = run_segment(image_path=image_path, conf=conf, iou=iou, imgsz=imgsz, device=device, save=True, project='runs', name='segment-api')
    return jsonify(out)


@app.route('/api/track', methods=['POST'])
def api_track():
    # Accept video/image/dir path or uploaded file
    file = request.files.get('file')
    source = request.form.get('source')
    conf = float(request.form.get('conf', 0.15))
    iou = float(request.form.get('iou', 0.5))
    imgsz = int(request.form.get('imgsz', 416))
    device = request.form.get('device', 'cpu')
    tracker = request.form.get('tracker', 'bytetrack.yaml')

    if file:
        upload_dir = app.config.get('UPLOAD_DIR', 'uploads')
        os.makedirs(upload_dir, exist_ok=True)
        save_path = os.path.join(upload_dir, file.filename)
        file.save(save_path)
        source = save_path

    if not source:
        return jsonify({"error": "No source provided"}), 400

    from yolo_service import run_track
    out = run_track(source=source, conf=conf, iou=iou, imgsz=imgsz, device=device, tracker=tracker, save=True, project='runs', name='track-api')
    return jsonify(out)


@app.route('/api/track_video', methods=['POST'])
def api_track_video():
    file = request.files.get('file')
    conf = float(request.form.get('conf', 0.15))
    iou = float(request.form.get('iou', 0.5))
    imgsz = int(request.form.get('imgsz', 416))
    device = request.form.get('device', 'cpu')
    tracker = request.form.get('tracker', 'bytetrack.yaml')
    persist = request.form.get('persist', 'true').lower() in ('1','true','yes','on')

    if not file:
        return jsonify({"error": "No video provided"}), 400

    upload_dir = app.config.get('UPLOAD_DIR', 'uploads')
    os.makedirs(upload_dir, exist_ok=True)
    in_path = os.path.join(upload_dir, file.filename)
    file.save(in_path)

    from yolo_service import track_video_collect
    out = track_video_collect(
        source=in_path,
        conf=conf,
        iou=iou,
        imgsz=imgsz,
        device=device,
        tracker=tracker,
        save=True,
        project='runs',
        name='track-video',
        persist=persist,
    )
    return jsonify(out)


@app.route('/download')
def download():
    path = request.args.get('path')
    if not path or not os.path.exists(path):
        return 'Not found', 404
    if os.path.isdir(path):
        return 'Directory listing not implemented. Please open on disk.', 400
    return send_file(path, as_attachment=True)


@app.route('/preview')
def preview():
    path = request.args.get('path')
    if not path or not os.path.exists(path):
        return 'Not found', 404
    # For inline display
    return send_file(path)


@app.route('/api/progress')
def api_progress():
    path = request.args.get('path')
    if not path or not os.path.exists(path):
        return jsonify({"processed": 0, "total": None})
    try:
        import json
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception:
        data = {"processed": 0, "total": None}
    return jsonify(data)

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(host='0.0.0.0', port=5000, debug=True)
