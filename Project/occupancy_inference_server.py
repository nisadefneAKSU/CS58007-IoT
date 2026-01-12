"""
Occupancy Detection Inference Service
-------------------------------------
- Loads trained Random Forest model
- Converts raw sensor readings into model features
- Predicts occupancy (0 = Empty, 1 = Occupied)
- Exposes REST API for system integration
"""

from flask import Flask, request, jsonify
import joblib
import json
import pandas as pd
from datetime import datetime

MODEL_PATH = "trained_models/random_forest.pkl" # Path to the trained Random Forest model
FEATURES_PATH = "trained_models/feature_names.json" # Path to the saved list of feature names used during training
model = joblib.load(MODEL_PATH) # Load the trained Random Forest model into memory
with open(FEATURES_PATH, "r") as f: # Load the exact feature order expected by the model
    FEATURE_NAMES = json.load(f)

def build_feature_vector(raw): # This function maps raw sensor input to the features expected by the trained model
    """Converts raw sensor readings into a feature vector
    consistent with the training pipeline."""

    now = datetime.now() # Get the current timestamp to compute temporal features

    features = { # Dictionary containing all features expected by the model
        # Raw sensor values (from Pi)
        "Temperature": raw["Temperature"],
        "Humidity": raw["Humidity"],
        "Light": raw["Light"],
        "CO2": raw["CO2"],
        "PIR": raw["PIR"],
        "Microphone": raw["Microphone"],

        # Rolling/statistical feature proxies (approximated in real-time)
        "temp_mean": raw["Temperature"],
        "humidity_mean": raw["Humidity"],
        "light_mean": raw["Light"],
        "co2_mean": raw["CO2"],
        "noise_mean": raw["Microphone"],

        # Noise statistics (no rolling history available)
        "noise_min": raw["Microphone"],
        "noise_max": raw["Microphone"],
        "noise_std": 0.0,
        "noise_variance": 0.0,

        # CO2 change features (initialized safely)
        "co2_delta": 0.0,
        "co2_delta_mean": 0.0,
        "co2_variance": 0.0,

        # Light change features
        "light_variance": 0.0,
        "light_std": 0.0,
        "light_delta": 0.0,

        # Temperature change feature
        "temp_delta": 0.0,

        # PIR aggregation features
        "pir_sum": raw["PIR"],
        "pir_max": raw["PIR"],

        # Time-based features
        "hour": now.hour,
        "day_of_week": now.weekday()
    }

    # Reorder features exactly as they were during training
    ordered_features = [features[f] for f in FEATURE_NAMES]
    # Return a single-row DataFrame ready for model.predict()
    return pd.DataFrame([ordered_features], columns=FEATURE_NAMES)
    
app = Flask(__name__)# Create Flask application instance

# Root/health check endpoint: Allows to verify that the server is running by visiting http://<server-ip>:5000/
# Responds with a simple message. Method is GET.
@app.route("/", methods=["GET"])
def health():
    return "Occupancy Inference Server is running"

@app.route("/predict", methods=["POST"])
# Define an HTTP endpoint at /predict that accepts POST requests
def predict():
    """Expects JSON payload with raw sensor values.
    Returns: { "prediction": 0 }  or  { "prediction": 1 }"""

    raw_data = request.json # Extract JSON body from incoming request
    try:
        X = build_feature_vector(raw_data) # Convert raw sensor data into model features
        prediction = int(model.predict(X)[0]) # Run model inference and convert output to int (0 or 1)
        return jsonify({  # Return prediction as JSON response
            "prediction": prediction,
            "status": "success"
        })
    except Exception as e: # Catch any error (missing feature, wrong format, etc.)
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 400 # Return error message with HTTP 400 status
        
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
