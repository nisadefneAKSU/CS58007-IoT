import time
from flask import Flask, request, jsonify, Response, stream_with_context

app = Flask(__name__)

# Global state for streaming
current_state = {
    "people": 0,
    "last_updated": 0.0
}

@app.route("/infer", methods=["POST"])
def infer():
    # 1. Parse JSON data (No image decoding needed!)
    data = request.json
    if not data:
        return jsonify({"error": "No JSON data"}), 400

    people_count = data.get("people", 0)
    capture_ts = float(data.get("capture_ts", 0))
    
    server_ts = time.time()

    # 3. Update Global State
    current_state["people"] = people_count
    current_state["last_updated"] = server_ts

    return jsonify({
        "status": "ok",
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
    return jsonify({"server_ts": time.time()})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, threaded=True)