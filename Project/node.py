import serial, joblib, requests, time, random, numpy as np, pandas as pd
from datetime import datetime
from collections import deque

# --- CONFIG ---
SERVER_IP = "192.168.137.1" 
SERIAL_PORT = "/dev/ttyUSB0" 
MODEL_PATH = "trained_models/random_forest.pkl"
ROOM_ID = "Room_101"

FEATURE_NAMES = [
    "Temperature", "Humidity", "Light", "CO2", "PIR", "Microphone",
    "noise_mean", "noise_min", "co2_mean", "co2_delta", "co2_delta_mean", 
    "co2_variance", "light_mean", "light_variance", "light_std", "light_delta", 
    "temp_mean", "humidity_mean", "temp_delta", "pir_sum", "pir_max", 
    "pir_rolling_mean", "hour"
]

print("Initializing Node with 23-feature model...")
model = joblib.load(MODEL_PATH)
buffers = {s: deque(maxlen=5) for s in ["Temperature", "Humidity", "Light", "CO2", "Microphone", "PIR"]}

def get_features(raw):
    # Calculate stats for the 5-sample window
    stats = {n: {"mean": np.mean(b), "max": np.max(b), "min": np.min(b), "var": np.var(b), "std": np.std(b), "delta": b[-1]-b[-2] if len(b)>1 else 0} for n, b in buffers.items()}
    c_list = list(buffers["CO2"])
    
    f = {
        "Temperature": raw["Temperature"], 
        "Humidity": raw["Humidity"], 
        "Light": raw["Light"], 
        "CO2": raw["CO2"], 
        "PIR": raw["PIR"], 
        "Microphone": raw["Microphone"],
        "noise_mean": stats["Microphone"]["mean"], 
        "noise_min": stats["Microphone"]["min"],
        "co2_mean": stats["CO2"]["mean"], 
        "co2_delta": stats["CO2"]["delta"], 
        "co2_delta_mean": np.mean(np.diff(c_list)) if len(c_list)>1 else 0, 
        "co2_variance": stats["CO2"]["var"],
        "light_mean": stats["Light"]["mean"], 
        "light_variance": stats["Light"]["var"], 
        "light_std": stats["Light"]["std"], 
        "light_delta": stats["Light"]["delta"],
        "temp_mean": stats["Temperature"]["mean"], 
        "humidity_mean": stats["Humidity"]["mean"], 
        "temp_delta": stats["Temperature"]["delta"],
        "pir_sum": sum(buffers["PIR"]), 
        "pir_max": max(buffers["PIR"]),
        "pir_rolling_mean": stats["PIR"]["mean"], # New feature added
        "hour": datetime.now().hour
    }
    return pd.DataFrame([[f[n] for n in FEATURE_NAMES]], columns=FEATURE_NAMES)

try:
    ser = serial.Serial(SERIAL_PORT, 9600, timeout=1)
    time.sleep(2)
    ser.reset_input_buffer()
except Exception as e:
    print(f"Serial Error: {e}")
    exit()

while True:
    line = ser.readline().decode('utf-8', errors='ignore').strip()
    if line and ',' in line:
        ser.reset_input_buffer()
        try:
            p = [float(x) for x in line.split(',')]
            if len(p) == 5:

                raw = {
                    "Temperature": p[0], "Humidity": p[1], "Light": p[2], 
                    "CO2": p[3], "PIR": p[4], "Microphone": 0.0 
                }
                
                for k,v in raw.items(): buffers[k].append(v)
                
                if all(len(b) >= 5 for b in buffers.values()):
                    X = get_features(raw)
                    pred_bool = bool(model.predict(X)[0])
                    
                    #Manual safety
                    if pred_bool and p[4] == 0 and p[3] < 150:
                        pred_bool = False
                    
                    payload = {
                        "room_id": ROOM_ID, 
                        "is_occupied": pred_bool, 
                        "temperature": raw["Temperature"], 
                        "light": raw["Light"], 
                        "humidity": raw["Humidity"], 
                        "pir": int(raw["PIR"])
                    }
                    
                    requests.post(f"http://{SERVER_IP}:8000/update", json=payload, timeout=0.5)
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] Occupied: {pred_bool} | PIR: {int(p[4])}")
        except: continue
    time.sleep(0.5)