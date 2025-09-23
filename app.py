from flask import Flask, request, jsonify
app = Flask(__name__)

@app.route("/upload", methods=["POST"])
def upload():
    # forgot to handle file saving properly
    file = request.files["video"]
    return jsonify({"status":"ok", "file": file.filename})

@app.route("/run", methods=["POST"])
def run_tracking():
    # references undefined function
    result = perform_tracking("uploads/video.mp4")
    return jsonify({"result": result})

if __name__ == "__main__":
    app.run()
