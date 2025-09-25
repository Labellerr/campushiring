from flask import Flask, request, jsonify
from ultralytics import YOLO
import json, os

app = Flask(__name__)
model = YOLO('yolov8n.pt')

@app.route('/')
def index():
    return '''
    <h1>🚗 Labellerr Vehicle-Pedestrian Tracker</h1>
    <p>✅ YOLO Detection Ready | ✅ Web Interface Live</p>
    <form method="post" enctype="multipart/form-data" action="/predict">
        <input type="file" name="image" accept=".jpg,.png">
        <button>Detect Objects</button>
    </form>
    '''

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files.get('image')
    if not file: return jsonify({'error': 'No image'})
    
    filepath = f"temp_{file.filename}"
    file.save(filepath)
    results = model(filepath)
    
    detections = []
    for r in results:
        for box in r.boxes:
            detections.append({
                'class': model.names[int(box.cls[0])],
                'confidence': float(box.conf[0])
            })
    
    os.remove(filepath)
    return jsonify({'detections': detections, 'count': len(detections)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
