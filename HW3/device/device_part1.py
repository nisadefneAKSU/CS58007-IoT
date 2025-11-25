import time
import csv
import requests
import cv2
import numpy as np
import threading

# Configuration variables. Change as necessary
SERVER_IP = "http://10.165.66.45:8080"
SERVER_URL = SERVER_IP + "/infer"
CAMERA_ID = 0

class CameraStream:
    def __init__(self, src=0):
        # Initialize with V4L2
        self.stream = cv2.VideoCapture(src, cv2.CAP_V4L2)
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        if not self.stream.isOpened():
            print("Error: Camera failed to open.")
            exit()
            
        # Read first frame
        (self.grabbed, self.frame) = self.stream.read()
        self.stopped = False

    def start(self):
        # Start the thread
        threading.Thread(target=self.update, args=(), daemon=True).start()
        return self

    def update(self):
        while not self.stopped:
            # Try to read a frame
            (grabbed, frame) = self.stream.read()
            
            # If successful, update the shared frame
            if grabbed:
                self.frame = frame
                self.grabbed = True
            else:
                # If failed, DO NOT STOP. Just wait a tiny bit and try again.
                self.grabbed = False
                time.sleep(0.01) 

    def read(self):
        # Return the latest frame we have
        return self.frame

    def stop(self):
        self.stopped = True
        self.stream.release()

# csv logging
csv_file = open("latency_log_part1.csv", "w", newline="")
writer = csv.writer(csv_file)
# Added columns for FPS data
writer.writerow(["frame", "capture_ts", "rtt", "server_latency", "people", "current_fps"])

# --- Clock Sync Function ---
def sync_clock(server_ip):
    print("Synchronizing clocks...")
    try:
        t0 = time.time()
        response = requests.get(f"{server_ip}/time", timeout=5)
        t1 = time.time()
        if response.status_code == 200:
            server_ts = response.json().get("server_ts", 0)
            rtt = t1 - t0
            clock_offset = server_ts - (t1 - (rtt / 2))
            print(f"  > Time Offset: {clock_offset:.4f}s")
            return clock_offset
    except Exception as e:
        print(f"  > Sync failed: {e}")
        return 0.0
    return 0.0

# --- Main Execution ---
print("Starting Camera Thread...")
cam = CameraStream(CAMERA_ID).start()
time.sleep(2.0)

time_offset = sync_clock(SERVER_IP)
frame_count = 0

# FPS Calculation Variables
fps_start_time = time.time()
fps_frame_counter = 0
current_fps = 0.0

try:
    while True:
        frame = cam.read()
        if frame is None:
            time.sleep(0.1)
            continue

        capture_ts = time.time() + time_offset

        # Compress
        _, jpg_data = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        
        t_start = time.time()
        try:
            response = requests.post(
                SERVER_URL,
                files={"image": ("frame.jpg", jpg_data.tobytes(), "image/jpeg")},
                data={"capture_ts": str(capture_ts)},
                timeout=3.0
            )
            
            rtt = time.time() - t_start

            if response.status_code == 200:
                data = response.json()
                people = data.get("people", 0)
                lat = data.get("latency", 0)

                fps_frame_counter += 1
                if (time.time() - fps_start_time) > 1.0:
                    current_fps = fps_frame_counter / (time.time() - fps_start_time)
                    fps_frame_counter = 0
                    fps_start_time = time.time()

                print(f"FPS: {current_fps:.1f} | Latency: {lat:.3f}s | People: {people}")
                
                writer.writerow([frame_count, capture_ts, rtt, lat, people, current_fps])
                csv_file.flush()
            else:
                print(f"Server error: {response.status_code}")

        except Exception as e:
            print(f"Drop: {e}")

        frame_count += 1

except KeyboardInterrupt:
    print("Stopping...")
    cam.stop()
finally:
    csv_file.close()