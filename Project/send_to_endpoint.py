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

# ==========================================================
# 1. LOAD MODEL & FEATURE DEFINITIONS
# ==========================================================

MODEL_PATH = "trained_models/random_forest.pkl"
FEATURES_PATH = "trained_models/feature_names.json"

model = joblib.load(MODEL_PATH)

with open(FEATURES_PATH, "r") as f:
    FEATURE_NAMES = json.load(f)

# ==========================================================
# 2. FEATURE CONSTRUCTION
#    (Matches training-time features)
# ==========================================================

def build_feature_vector(raw):
    """
    Converts raw sensor readings into a feature vector
    consistent with the training pipeline.
    """

    now = datetime.now()

    features = {
        # Raw sensors
        "Temperature": raw["Temperature"],
        "Humidity": raw["Humidity"],
        "Light": raw["Light"],
        "CO2": raw["CO2"],
        "PIR": raw["PIR"],
        "Microphone": raw["Microphone"],

        # Rolling/statistical approximations (real-time)
        "temp_mean": raw["Temperature"],
        "humidity_mean": raw["Humidity"],
        "light_mean": raw["Light"],
        "co2_mean": raw["CO2"],
        "noise_mean": raw["Microphone"],

        "noise_min": raw["Microphone"],
        "noise_max": raw["Microphone"],
        "noise_std": 0.0,
        "noise_variance": 0.0,

        "co2_delta": 0.0,
        "co2_delta_mean": 0.0,
        "co2_variance": 0.0,

        "light_variance": 0.0,
        "light_std": 0.0,
        "light_delta": 0.0,

        "temp_delta": 0.0,

        "pir_sum": raw["PIR"],
        "pir_max": raw["PIR"],

        # Temporal features
        "hour": now.hour,
        "day_of_week": now.weekday()
    }

    # Enforce correct feature order
    ordered_features = [features[f] for f in FEATURE_NAMES]

    return pd.DataFrame([ordered_features], columns=FEATURE_NAMES)

# ==========================================================
# 3. FLASK APPLICATION
# ==========================================================

app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    """
    Expects JSON payload with raw sensor values.
    Returns:
        { "prediction": 0 }  or  { "prediction": 1 }
    """

    raw_data = request.json

    try:
        X = build_feature_vector(raw_data)
        prediction = int(model.predict(X)[0])

        return jsonify({
            "prediction": prediction,
            "status": "success"
        })

    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 400

# ==========================================================
# 4. SERVICE ENTRY POINT
# ==========================================================

if __name__ == "__main__":
    app.run(
        host="0.0.0.0",  # Accessible from Raspberry Pi / network
        port=5000,
        debug=False
    )


'''
import requests

response = requests.post(
    "http://<raspberry-pi-ip>:5000/predict",
    json=sensor_data
)

occupancy = response.json()["prediction"] -> use like this

'''