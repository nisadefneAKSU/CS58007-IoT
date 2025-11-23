# device.py
import time
import csv
import argparse
import requests
import cv2
import numpy as np
from datetime import datetime

# Argument parsing
parser = argparse.ArgumentParser()

# Server URL must include /infer
parser.add_argument("--server", default="http://127.0.0.1/infer", help="Server infer URL (include /infer)")

# Which camera index to use (0 = default webcam)
parser.add_argument("--camera", type=int, default=0, help="Camera index")

# Delay between captured frames (0.2 → approx. 5 FPS)
parser.add_argument("--interval", type=float, default=0.2, help="Seconds between frames")

# Output CSV file for latency logging
parser.add_argument("--out", default="latency_log.csv", help="CSV log file")

# How many frames to capture before stopping (0 = infinite)
parser.add_argument("--max_frames", type=int, default=0, help="Stop after this many frames (0 = infinite)")

args = parser.parse_args()
SERVER_URL = args.server

# Initialize webcam
cap = cv2.VideoCapture(args.camera)
if not cap.isOpened():
    raise SystemExit("Cannot open camera index {}".format(args.camera))

# Set resolution for consistency
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Prepare the CSV log file
fieldnames = ["frame_idx", "capture_ts", "client_send_ts", "server_ts", "inference_time", "latency_server_calc", "rtt_seconds", "people"]
csvfile = open(args.out, "w", newline="")
writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
writer.writeheader()

# Main capture loop
frame_idx = 0
try:
    while True:
        # Stop if max frame count reached
        if args.max_frames and frame_idx >= args.max_frames:
            break
         # Capture one frame from webcam
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame")
            break
        # Timestamp right after frame acquisition
        capture_ts = time.time()
        # JPEG encode the frame before sending
        ok, jpg = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        if not ok:
            print("JPEG encode failed")
            continue
        img_bytes = jpg.tobytes()

        # Prepare for network request
        client_send_ts = time.time()
        # Send frame to server and measure RTT
        try:
            files = {"image": ("frame.jpg", img_bytes, "image/jpeg")}
            data = {"capture_ts": str(capture_ts)}
            t0 = time.time() # Start time of POST request
            resp = requests.post(SERVER_URL, files=files, data=data, timeout=5.0)
            t1 = time.time() # End time of POST request
        except Exception as e:
            print("Request failed:", e)
            time.sleep(args.interval)
            continue

        # Round-trip time (client → server → client)
        rtt = t1 - t0

        # Process server response
        if resp.status_code == 200:
            j = resp.json()
            server_ts = j.get("server_ts") # Timestamp inside server before running YOLO
            inference_time = j.get("inference_time") # Server-side model inference duration
            latency_server_calc = j.get("latency") # server_ts - capture_ts
            people = j.get("people") # People detected in frame
        else:
            print("Server returned", resp.status_code, resp.text)
            time.sleep(args.interval)
            continue
        
         # Write log row to CSV
        writer.writerow({
            "frame_idx": frame_idx,
            "capture_ts": capture_ts,
            "client_send_ts": client_send_ts,
            "server_ts": server_ts,
            "inference_time": inference_time,
            "latency_server_calc": latency_server_calc,
            "rtt_seconds": rtt,
            "people": people
        })
        csvfile.flush()

        # Display readable output
        print(f"[{frame_idx}] people={people} inference={inference_time:.3f}s rtt={rtt:.3f}s server_latency={latency_server_calc}")
        frame_idx += 1
        # Control frame rate
        time.sleep(args.interval)

# Ctrl+C handling       
except KeyboardInterrupt:
    print("Stopping capture")
finally:
    # Cleanup resources
    cap.release()
    csvfile.close()