# device_part2.py
import time
import csv
import argparse
import requests
import cv2
from ultralytics import YOLO

parser = argparse.ArgumentParser()
parser.add_argument("--server", default="http://127.0.0.1/upload_count",
                    help="Server upload_count URL (include /upload_count)")
parser.add_argument("--camera", type=int, default=0, help="Camera index")
parser.add_argument("--interval", type=float, default=0.0,
                    help="Seconds between frames (0.0 -> run as fast as possible)")
parser.add_argument("--out", default="latency_part2.csv", help="CSV log file")
parser.add_argument("--max_frames", type=int, default=0,
                    help="Stop after this many frames (0 = infinite)")
parser.add_argument("--conf", type=float, default=0.25, help="YOLO confidence threshold")
args = parser.parse_args()

SERVER_URL = args.server

# Load Tiny YOLO (lightweight)
print("Loading YOLO model (device) ...")
model = YOLO("yolov8n.pt")  # yolov8n is small; replace if you have different tiny model

# Initialize webcam

# Bende kaspersky yüzünden kamera çalışmıyordu, başka bilgisayar da yoktu
# bu yüzden video kullandım:

cap = cv2.VideoCapture(r"C:\Users\suuser\Desktop\IoT_HW3\test_video_fixed.mp4")
if not cap.isOpened():
    raise SystemExit("Cannot open video file")

# Kamera için şununla test edin pls:
# cap = cv2.VideoCapture(args.camera)

# Şu resolution olabilir:

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# CSV setup
fieldnames = [
    "frame_idx",
    "capture_ts",
    "inference_start_ts",
    "inference_end_ts",
    "inference_time",
    "client_send_ts",
    "server_response_ts",
    "rtt_seconds",
    "people"
]
csvfile = open(args.out, "w", newline="")
writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
writer.writeheader()

frame_idx = 0
try:
    while True:
        if args.max_frames and frame_idx >= args.max_frames:
            break

        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame")
            break

        # capture timestamp
        capture_ts = time.time()

        # run local Tiny YOLO inference (only persons)
        inf_start = time.time()
        # request only class 0 (person) for speed
        results = model(frame, conf=args.conf, classes=[0], imgsz=640)
        inf_end = time.time()
        inference_time = inf_end - inf_start

        # count people (results[0].boxes may be None)
        people_count = 0
        if len(results) > 0:
            boxes = results[0].boxes
            if boxes is not None:
                # boxes is an array, number of boxes = people
                try:
                    people_count = len(boxes)
                except:
                    # fallback: use .xyxy or .boxes.xyxy
                    people_count = 0

        # 3) POST the small JSON with the count
        client_send_ts = time.time()
        json_payload = {"people": int(people_count), "capture_ts": capture_ts}
        t0 = time.time()
        try:
            resp = requests.post(SERVER_URL, json=json_payload, timeout=5.0)
            t1 = time.time()
            rtt = t1 - t0
            server_response_ts = t1
        except Exception as e:
            print(f"POST failed: {e}")
            rtt = None
            server_response_ts = None
            # still log locally
            resp = None

        # 4) write CSV row
        writer.writerow({
            "frame_idx": frame_idx,
            "capture_ts": capture_ts,
            "inference_start_ts": inf_start,
            "inference_end_ts": inf_end,
            "inference_time": inference_time,
            "client_send_ts": client_send_ts,
            "server_response_ts": server_response_ts,
            "rtt_seconds": rtt,
            "people": int(people_count)
        })
        csvfile.flush()

        # 5) print readable status
        rtt_str = f"{rtt:.3f}s" if rtt is not None else "N/A"
        print(f"[{frame_idx}] people={people_count} inf={inference_time:.3f}s rtt={rtt_str}")

        frame_idx += 1
        if args.interval > 0:
            time.sleep(args.interval)

except KeyboardInterrupt:
    print("Stopping capture")
finally:
    cap.release()
    csvfile.close()
