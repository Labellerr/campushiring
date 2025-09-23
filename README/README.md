# YOLO Segmentation + ByteTrack Flask Application 🚀

A powerful web application that combines **YOLO segmentation** with **ByteTrack** for advanced object detection and tracking in videos. Built with Flask, this application provides a user-friendly interface for uploading videos, running AI analysis, and downloading detailed results.

## ✨ Features

### 🤖 AI-Powered Analysis
- **YOLO Segmentation**: State-of-the-art object detection and segmentation
- **ByteTrack**: Multi-object tracking with persistent IDs
- **Real-time Processing**: Efficient video analysis with progress tracking
- **Customizable Parameters**: Adjustable confidence, IoU, and image size settings

### 🌐 Web Interface
- **Modern UI**: Clean, responsive design with dark theme
- **Video Preview**: Instant video preview upon upload
- **Live Progress**: Real-time processing status updates
- **Results Download**: Export detailed JSON results with tracking data

### 🔐 User Management
- **Authentication**: User registration and login system
- **Personal Dashboard**: Individual user workspaces
- **Journal Entries**: Built-in journaling functionality
- **Session Management**: Secure user sessions

## 📋 Requirements

- Python 3.8+
- Flask
- OpenCV
- PyTorch
- YOLOv8
- ByteTrack
- Additional dependencies (see `requirements.txt`)

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/entity079/kiranjot_kaur.git
cd kiranjot_kaur
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Set Up Environment
```bash
# Configure your settings in config.py
# Add your model paths and API keys as needed
```

### 4. Run the Application
```bash
python app.py
```

### 5. Access the Application
Open your browser and go to: `http://localhost:5000`

## 📖 Usage Guide

### Video Analysis
1. **Upload Video**: Select a video file (MP4, AVI, MOV supported)
2. **Configure Settings**:
   - **Detection Sensitivity**: Low (0.05), Medium (0.25), High (0.50)
   - **NMS Strictness**: Low (0.40), Medium (0.50), High (0.70)
   - **Image Size**: Low (320), Medium (416), High (640)
   - **Persist Track IDs**: Enable for consistent object tracking
3. **Run Analysis**: Click "Run Analysis" to start processing
4. **Download Results**: Get detailed JSON with detection data and tracking IDs

### User Features
- **Register/Login**: Create an account or sign in
- **Dashboard**: View your analysis history
- **Journal**: Keep notes about your projects
- **Profile Management**: Update your information

## 🔧 Configuration

### Model Configuration
Update `yolo_service.py` to point to your trained models:
```python
# Example model paths
model_path = "path/to/your/yolo/model.pt"
tracker_config = "bytetrack.yaml"
```

### Application Settings
Modify `config.py` for your environment:
```python
# Database configuration
SQLALCHEMY_DATABASE_URI = 'sqlite:///app.db'

# Upload directory
UPLOAD_DIR = 'uploads'

# Model settings
MODEL_CONFIDENCE = 0.25
```

## 📁 Project Structure

```
kiranjot_kaur/
├── app.py                 # Main Flask application
├── config.py             # Configuration settings
├── yolo_service.py       # YOLO model integration
├── labellerr.py          # Labeling utilities
├── templates/            # HTML templates
│   ├── base.html        # Base template
│   ├── video_demo.html  # Video analysis interface
│   ├── dashboard.html   # User dashboard
│   └── ...              # Other templates
├── runs/                # Model outputs and results
├── uploads/             # User uploaded files
└── requirements.txt     # Python dependencies
```

## 🤝 API Endpoints

### Video Processing
- `POST /api/track_video` - Process video with tracking
- `POST /api/segment` - Image segmentation
- `POST /api/track` - Object tracking
- `GET /api/progress` - Get processing progress

### User Management
- `GET/POST /login` - User authentication
- `GET/POST /register` - User registration
- `GET /dashboard` - User dashboard
- `POST /new_entry` - Create journal entry

### File Operations
- `GET /download` - Download processed files
- `GET /preview` - Preview uploaded files

## 📊 Output Format

The application generates detailed JSON results with:
```json
{
  "summary": {
    "person": {
      "detections": 1500,
      "unique_ids": 25
    }
  },
  "results_json": "/path/to/results.json"
}
```

## 🔍 Advanced Features

### Model Customization
- Support for custom YOLO models
- Configurable tracking parameters
- Multiple detection classes
- Real-time video processing

### Performance Optimization
- GPU acceleration support
- Batch processing capabilities
- Memory-efficient streaming
- Asynchronous processing

## 🐛 Troubleshooting

### Common Issues
1. **Model Loading Errors**: Ensure correct model paths in `yolo_service.py`
2. **Memory Issues**: Reduce batch size or image resolution
3. **Upload Failures**: Check file size limits and supported formats

### Debug Mode
Run with debug logging:
```bash
export FLASK_DEBUG=1
python app.py
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Commit your changes: `git commit -am 'Add feature'`
4. Push to the branch: `git push origin feature-name`
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **Ultralytics** for YOLOv8
- **ByteTrack** for multi-object tracking
- **Flask** for the web framework
- **OpenCV** for computer vision utilities

## 📞 Support

For support and questions:
- Create an issue on GitHub
- Check the troubleshooting section
- Review the API documentation

---

**Made with ❤️ for AI-powered video analysis**
