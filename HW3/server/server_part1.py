import time
from flask import Flask, request, jsonify, Response, stream_with_context
import numpy as np
from ultralytics import YOLO
import cv2

app = Flask(__name__)

# Load Model
print("Loading model...")
model = YOLO("yolov8n.pt")

# Global state for streaming
current_state = {
    "people": 0,
    "last_updated": 0.0
}

@app.route("/infer", methods=["POST"])
def infer():
    # 1. Decode Image
    if "image" not in request.files:
        return jsonify({"error": "No image"}), 400
        
    file_bytes = np.frombuffer(request.files["image"].read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    # 2. Run Inference
    results = model(img, classes=[0], verbose=False) # class 0 = person
    count = len(results[0].boxes) if results else 0

    # 3. Update Stats
    server_ts = time.time()
    
    # Calculate latency if timestamp provided
    capture_ts = request.form.get("capture_ts")
    latency = 0.0
    if capture_ts:
        latency = server_ts - float(capture_ts)

    # Update global state
    current_state["people"] = count
    current_state["last_updated"] = server_ts
    return jsonify({
        "people": count,
        "latency": latency,
        "server_ts": server_ts
    })

@app.route("/stream", methods=["GET"])
def get_status():
    return jsonify({
        "people": current_state["people"],
        "timestamp": current_state["last_updated"]
    })

@app.route("/time", methods=["GET"])
def get_time():
    # Return server time so the device can sync
    return jsonify({"server_ts": time.time()})

# ... existing code ...

if __name__ == "__main__":
    # Host 0.0.0.0 is required for VM/Docker
    app.run(host="0.0.0.0", port=8080, threaded=True)