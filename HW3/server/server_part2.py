# server.py
import io
import time
import threading
from datetime import datetime
from queue import Queue, Empty

from flask import Flask, request, jsonify, Response, stream_with_context
import numpy as np
from PIL import Image
from ultralytics import YOLO  # Ultralytics YOLOv8
import cv2

app = Flask(__name__)

# Load model once (yolov8n = lightweight)
MODEL_NAME = "yolov8n.pt"
print("Loading YOLO model:", MODEL_NAME)
model = YOLO(MODEL_NAME)  # Downloads if necessary

# Stats and streaming
latest_count = 0
latest_inference_ts = None
latest_latency = None
total_requests = 0
latency_sum = 0.0
last_inference_time = None
# For simple max_fps estimate we keep timestamps of inferences in a short buffer
fps_timestamps = []

# Queue to push events to SSE clients
event_queue = Queue(maxsize=100)

def push_event(data: dict):
    """Push data to SSE queue (non-blocking)."""
    try:
        event_queue.put_nowait(data)
    except:
        # drop if queue full
        pass


@app.route("/infer", methods=["POST"]) # POST method
def infer():
    """Accepts an image via multipart/form-data with key 'image' OR raw bytes in body.
    Optional form field 'capture_ts' = epoch float or iso timestamp sent from the device for latency calc.
    Returns JSON: {'people': int, 'server_ts': epoch, 'inference_time': seconds, 'processing_latency': seconds}"""
    global latest_count, latest_inference_ts, latest_latency, total_requests, latency_sum, last_inference_time, fps_timestamps

    # Parse capture timestamp if provided
    capture_ts = None
    if "capture_ts" in request.form:
        try:
            capture_ts = float(request.form["capture_ts"])
        except:
            try:
                capture_ts = float(request.form["capture_ts"])
            except:
                capture_ts = None
    elif request.args.get("capture_ts"):
        try:
            capture_ts = float(request.args.get("capture_ts"))
        except:
            capture_ts = None

    # Image bytes
    img_bytes = None
    if "image" in request.files:
        f = request.files["image"]
        img_bytes = f.read()
    else:
        # Maybe raw body bytes
        img_bytes = request.data if request.data else None

    if not img_bytes:
        return jsonify({"error": "no image provided"}), 400

    # Decode image from bytes to numpy array (BGR for cv2)
    try:
        img_array = np.frombuffer(img_bytes, dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        if img is None:
            # Fallback via PIL
            pil = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            img = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    except Exception as e:
        return jsonify({"Error": "could not decode image", "exception": str(e)}), 400

    # Run detection restricted to class 0 (person) for speed
    start = time.time()
    # Specify classes=0 to only detect persons
    results = model(img, conf=0.25, classes=[0], imgsz=640)  # Imgsz adjustable
    inference_time = time.time() - start

    # Count people
    people_count = 0
    if len(results) > 0:
        # results[0].boxes is ultralytics structure
        boxes = results[0].boxes
        if boxes is not None:
            people_count = len(boxes)

    server_ts = time.time()

    # Update stats
    total_requests += 1
    latest_count = int(people_count)
    latest_inference_ts = server_ts
    latest_latency = None
    if capture_ts:
        # capture_ts expected as epoch float seconds
        latest_latency = server_ts - capture_ts
        latency_sum += latest_latency

    # Update FPS timestamps
    fps_timestamps.append(server_ts)
    # Keep last N timestamps (e.g. last 100)
    if len(fps_timestamps) > 200:
        fps_timestamps = fps_timestamps[-200:]

    # Compute avg latency
    avg_latency = (latency_sum / total_requests) if (total_requests > 0 and latency_sum > 0) else None

    # Compute simple max_fps estimate (Inverse of min delta between inference timestamps)
    max_fps = None
    if len(fps_timestamps) >= 2:
        diffs = [t2 - t1 for t1, t2 in zip(fps_timestamps[:-1], fps_timestamps[1:]) if (t2 - t1) > 0]
        if diffs:
            min_diff = min(diffs)
            if min_diff > 0:
                max_fps = 1.0 / min_diff

    # Push event to SSE clients
    push_event({
        "ts": server_ts,
        "people": latest_count,
        "inference_time": inference_time,
        "latency": latest_latency,
        "avg_latency": avg_latency,
        "max_fps_est": max_fps,
    })

    return jsonify({
        "people": latest_count,
        "server_ts": server_ts,
        "inference_time": inference_time,
        "latency": latest_latency,
        "avg_latency": avg_latency,
        "max_fps_est": max_fps,
    })


@app.route("/stream") # GET method
def stream():
    """SSE stream that yields events whenever a new inference happens.
    Clients can connect to /stream and will receive lines like:
    event: update
    data: {...json...}"""
    def event_stream():
        # A simple consumer of the shared queue. We block on get with timeout.
        while True:
            try:
                item = event_queue.get(timeout=10)  # Block up to 10s
                yield f"event: update\n"
                yield f"data: {item}\n\n"
            except Empty:
                # Keep connection alive with a ping
                yield "event: ping\n"
                yield f"data: {{'ts': {time.time()}}}\n\n"
    headers = {"Content-Type": "text/event-stream", "Cache-Control": "no-cache", "Connection": "keep-alive"}
    return Response(stream_with_context(event_stream()), headers=headers)
   

if __name__ == "__main__":
    # Run on port 80 (requires root) — for testing you can change.
    # When running as non-root use port 8080 and map host port 80 to VM port 8080 via port forwarding.
    app.run(host="0.0.0.0", port=8080, threaded=True)

# I reused Nisa's code above
@app.route("/upload_count", methods=["POST"])

def upload_count():
    """Receive a JSON {people, capture_ts} from the device (Part 2).
    Stores latest_count and pushes an SSE event."""
    global latest_count, latest_inference_ts, latest_latency, total_requests, latency_sum

    data = request.get_json(force=True, silent=True)
    if not data:
        return jsonify({"error": "invalid json"}), 400

    people = data.get("people")
    capture_ts = data.get("capture_ts")  # optional

    try:
        people = int(people)
    except:
        return jsonify({"error": "invalid people count"}), 400

    server_ts = time.time()
    latest_count = people
    latest_inference_ts = server_ts

    latest_latency = None
    if capture_ts:
        try:
            capture_ts_f = float(capture_ts)
            latest_latency = server_ts - capture_ts_f
            latency_sum += latest_latency
        except:
            latest_latency = None

    total_requests += 1

    # push to SSE clients
    push_event({
        "ts": server_ts,
        "people": latest_count,
        "latency": latest_latency,
    })

    return jsonify({"status": "ok", "server_ts": server_ts})
