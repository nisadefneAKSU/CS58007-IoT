import csv
import time
import requests
import cv2
import numpy as np
import threading

# Import TFLite
try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    try:
        import tensorflow.lite as tflite
    except ImportError:
        print("Error: Install tflite-runtime or tensorflow")
        exit()

### Configuration (change as necessary) ###
SERVER_IP = "http://10.165.66.45:8080"  # change this as necessary. maybe actually allow this to be user input?
SERVER_URL = SERVER_IP + "/infer"
MODEL_PATH = "yolov8n_float32.tflite"
CONF_THRESHOLD = 0.45 # confidence threshold
INTERVAL = 0.0 # set this variable as necessary to make the device sleep for a certain time in seconds (Seconds between frames) (Can also use for debugging purposes)
LOG_FILE = "latency_log_part2.csv"

class YOLO_TFLite:
    def __init__(self, model_path):
        print(f"Loading Model: {model_path}")
        self.interpreter = tflite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
        self.input_shape = self.input_details[0]['shape']
        self.h = self.input_shape[1]
        self.w = self.input_shape[2]
        print(f"Model Expects: {self.w}x{self.h} | Type: Float32")

    def detect(self, image):
        img_resized = cv2.resize(image, (self.w, self.h))
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        input_data = (img_rgb / 255.0).astype(np.float32)
        input_data = np.expand_dims(input_data, axis=0)
        
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        self.interpreter.invoke()
        
        output_data = self.interpreter.get_tensor(self.output_details[0]['index'])[0]
        
        if output_data.shape[0] < output_data.shape[1]:
            output_data = output_data.T

        scores = output_data[:, 4] 
        max_score = np.max(scores)
        count = np.sum(scores > CONF_THRESHOLD)
        
        return int(count), float(max_score)

class CameraStream:
    def __init__(self, src=0):
        self.stream = cv2.VideoCapture(src, cv2.CAP_V4L2)
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        if not self.stream.isOpened():
            print("Error: Camera failed to open.")
            exit()
            
        (self.grabbed, self.frame) = self.stream.read()
        self.stopped = False

    def start(self):
        threading.Thread(target=self.update, args=(), daemon=True).start()
        return self

    def update(self):
        while not self.stopped:
            (grabbed, frame) = self.stream.read()
            if grabbed:
                self.frame = frame
                self.grabbed = True
            else:
                self.grabbed = False
                time.sleep(0.01)

    def read(self):
        return self.frame

    def stop(self):
        self.stopped = True
        self.stream.release()

session = requests.Session()
def sync_clock(server_ip):
    print("Synchronizing clocks...")
    try:
        t0 = time.time()
        response = session.get(f"{server_ip}/time", timeout=5)
        t1 = time.time()
        if response.status_code == 200:
            server_ts = response.json().get("server_ts", 0)
            rtt = t1 - t0
            offset = server_ts - (t1 - (rtt / 2))
            print(f"  > Time Offset: {offset:.4f}s")
            return offset
    except Exception as e:
        print(f"  > Sync failed: {e}")
    return 0.0

detector = YOLO_TFLite(MODEL_PATH)

print("Starting Camera...")
cam = CameraStream(0).start()
time.sleep(2.0)

time_offset = sync_clock(SERVER_IP)
frame_idx = 0

# fps calculation
fps_start_time = time.time()
fps_frame_counter = 0
current_fps = 0.0

# Prepare the CSV log file
fieldnames = ["frame_idx", "capture_ts", "proc_latency", "current_fps", "rtt_seconds", "people"]
csvfile = open(LOG_FILE, "w", newline="")
writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
writer.writeheader()


try:
    while True:
        # Capture
        frame = cam.read()
        if frame is None:
            time.sleep(0.01)
            continue
            
        capture_ts = time.time() + time_offset
        t_proc_start = time.time()
        people_count, max_conf = detector.detect(frame)
        t_proc_end = time.time()
        proc_latency = t_proc_end - t_proc_start # latency = time taken to resize+preprocess+infer

        fps_frame_counter += 1
        rtt=time.time() - fps_start_time
        if (rtt) > 1.0:
            current_fps = fps_frame_counter / (rtt)
            fps_frame_counter = 0
            fps_start_time = time.time()

        # Try Send Data
        try:
            payload = {
                "people": people_count,
                "capture_ts": capture_ts,
                "max_conf": max_conf,
                "proc_latency": proc_latency  # Sending latency to server too
            }
            
            session.post(SERVER_URL, json=payload, timeout=2.0)
            
            writer.writerow({
            "frame_idx": frame_idx,
            "capture_ts": capture_ts,
            "proc_latency": proc_latency,
            "current_fps": current_fps
            "rtt_seconds": rtt,
            "people": people_count
            })
            csvfile.flush()
            
            print(f"FPS: {current_fps:.1f} | Latency: {proc_latency:.3f}s | People: {people_count}")
            
        except Exception as e:
            print(f"Net Error: {e}")
            
        frame_idx += 1
        if INTERVAL > 0: time.sleep(INTERVAL)

except KeyboardInterrupt:
    print("\nStopping...")
    cam.stop()
    session.close()
    csvfile.close()