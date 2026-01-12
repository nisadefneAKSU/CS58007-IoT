from flask import Flask, request, jsonify
import joblib
import json
import pandas as pd
from datetime import datetime
from collections import deque
from threading import Lock

"""
Occupancy Detection Inference Service
-------------------------------------
- Loads trained Random Forest model
- Converts raw sensor readings into model features
- Predicts occupancy (0 = Empty, 1 = Occupied)
- Exposes REST API for system integration
"""

MODEL_PATH = "trained_models/random_forest.pkl" # Path to the trained Random Forest model
FEATURES_PATH = "trained_models/feature_names.json" # Path to the saved list of feature names used during training
model = joblib.load(MODEL_PATH) # Load the trained Random Forest model into memory
with open(FEATURES_PATH, "r") as f: # Load the exact feature order expected by the model
    FEATURE_NAMES = json.load(f)

# ==================== THREAD-SAFE BUFFERS ====================

BUFFER_SIZE = 5  # Match training rolling window
buffers = {
    "Temperature": deque(maxlen=BUFFER_SIZE),
    "Humidity": deque(maxlen=BUFFER_SIZE),
    "Light": deque(maxlen=BUFFER_SIZE),
    "CO2": deque(maxlen=BUFFER_SIZE),
    "Microphone": deque(maxlen=BUFFER_SIZE),
    "PIR": deque(maxlen=BUFFER_SIZE)
}
buffer_lock = Lock()

def compute_stats(buffer):
    """Compute mean, variance, std, min, max, delta for a buffer."""
    if not buffer:
        return None, None, None, None, None, None
    values = list(buffer)
    mean = sum(values)/len(values)
    variance = sum((x - mean)**2 for x in values)/len(values)
    std = variance ** 0.5
    min_val = min(values)
    max_val = max(values)
    delta = values[-1] - values[-2] if len(values) > 1 else 0.0
    return mean, variance, std, min_val, max_val, delta

def build_feature_vector(raw):
    """Convert raw sensor readings into feature vector consistent with training."""

    now = datetime.now()  # For temporal features

    with buffer_lock:  # Thread-safe update
        for sensor in buffers:
            buffers[sensor].append(raw[sensor])

        # Skip prediction if buffers not full yet
        if any(len(buffers[sensor]) < BUFFER_SIZE for sensor in buffers):
            return None  # Not enough data yet for rolling statistics

        # Compute rolling stats
        temp_mean, temp_var, temp_std, temp_min, temp_max, temp_delta = compute_stats(buffers["Temperature"])
        humidity_mean, humidity_var, humidity_std, humidity_min, humidity_max, humidity_delta = compute_stats(buffers["Humidity"])
        light_mean, light_var, light_std, light_min, light_max, light_delta = compute_stats(buffers["Light"])
        co2_mean, co2_var, co2_std, co2_min, co2_max, co2_delta = compute_stats(buffers["CO2"])
        noise_mean, noise_var, noise_std, noise_min, noise_max, noise_delta = compute_stats(buffers["Microphone"])
        pir_sum = sum(buffers["PIR"])
        pir_max = max(buffers["PIR"])

        # Compute CO2 rolling delta mean
        co2_values = list(buffers["CO2"])
        co2_deltas = [j-i for i, j in zip(co2_values[:-1], co2_values[1:])]
        co2_delta_mean = sum(co2_deltas)/len(co2_deltas) if co2_deltas else 0.0

    # Build feature dictionary
    features = {
        # Raw sensors
        "Temperature": raw["Temperature"],
        "Humidity": raw["Humidity"],
        "Light": raw["Light"],
        "CO2": raw["CO2"],
        "PIR": raw["PIR"],
        "Microphone": raw["Microphone"],

        # Noise features
        "noise_mean": noise_mean,
        "noise_min": noise_min,

        # CO2 features
        "co2_mean": co2_mean,
        "co2_delta": co2_delta,
        "co2_delta_mean": co2_delta_mean,
        "co2_variance": co2_var,

        # Light features
        "light_mean": light_mean,
        "light_variance": light_var,
        "light_std": light_std,
        "light_delta": light_delta,

        # Temp/Humidity
        "temp_mean": temp_mean,
        "humidity_mean": humidity_mean,
        "temp_delta": temp_delta,

        # PIR motion
        "pir_sum": pir_sum,
        "pir_max": pir_max,
        "pir_rolling_mean": pir_sum / BUFFER_SIZE,  # rolling mean = sum / window size

        # Temporal features
        "hour": now.hour
    }

    # Reorder features exactly as training
    ordered_features = [features[f] for f in FEATURE_NAMES]
    return pd.DataFrame([ordered_features], columns=FEATURE_NAMES)

# ==================== FLASK SERVER ====================
    
app = Flask(__name__)# Create Flask application instance

# Root/health check endpoint: Allows to verify that the server is running by visiting http://<server-ip>:5000/
# Responds with a simple message. Method is GET.
@app.route("/", methods=["GET"])
def health():
    """Returns a simple message to verify server is running."""
    return "Occupancy Inference Server is running"

@app.route("/predict", methods=["POST"])
def predict():
    """Expects JSON payload with raw sensor values.
    Returns prediction 0 (Empty) or 1 (Occupied), or status if not enough data."""

    # Extract JSON payload from the POST request
    raw_data = request.json

    # Check if all expected sensor readings are present
    missing_sensors = [s for s in buffers if s not in raw_data]
    if missing_sensors:
        # Return 400 if any sensor is missing
        return jsonify({
            "status": "error",
            "message": f"Missing sensor data: {missing_sensors}"
        }), 400
    
    # Validate sensor data types
    invalid_sensors = [s for s in buffers if not isinstance(raw_data[s], (int, float))]
    if invalid_sensors:
        return jsonify({
            "status": "error",
            "message": f"Invalid value type for sensors: {invalid_sensors}"
        }), 400

    try:
        # Convert raw sensor data into a DataFrame of features using rolling stats
        X = build_feature_vector(raw_data)

        # If the rolling buffer does not yet contain enough samples, skip prediction
        if X is None:
            buffer_lengths = {s: len(buffers[s]) for s in buffers}
            return jsonify({
                "status": "waiting",
                "message": f"Need {BUFFER_SIZE} samples per sensor before predicting.",
                "buffer_lengths": buffer_lengths
            }), 200

        # Perform inference using the pre-loaded Random Forest model
        prediction = int(model.predict(X)[0])

        # Log the input and prediction (optional but recommended for production)
        app.logger.info(f"Raw input: {raw_data}")
        app.logger.info(f"Prediction: {prediction}")

        # Return successful prediction
        return jsonify({
            "prediction": prediction,
            "status": "success"
        })

    except Exception as e:
        # Catch unexpected errors (e.g., type errors, computation issues)
        app.logger.error(f"Prediction error: {str(e)}", exc_info=True)
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

# ==================== SERVICE ENTRY POINT ====================

# Service entry endpoint
if __name__ == "__main__": # This block runs only when the script is executed directly
    app.run(
        host="0.0.0.0",  # Makes server accessible on local network
        port=5000,       # Port number for the API
        debug=False      # Debug disabled for production-like usage
    )

# Example client-side usage (commented out)
'''
import requests
response = requests.post(
    "http://<raspberry-pi-ip>:5000/predict",
    json=sensor_data
)
occupancy = response.json()["prediction"]
'''