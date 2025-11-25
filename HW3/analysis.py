import pandas as pd
import numpy as np
import os.path

df = pd.read_csv("latency_log.csv")
avg_latency = df['latency_server_calc'].dropna().mean()
print(f"Average latency (s): {avg_latency:.2f}") # This gives the average latency per frame

df['fps'] = 1 / df['inference_time']
max_fps = df['fps'].max()
print(f"Maximum inference rate (FPS): {max_fps:.2f}")

timestamps = df['capture_ts'].to_numpy()
frame_deltas = np.diff(timestamps)  # Time between consecutive frames
fps_estimate = 1 / np.mean(frame_deltas)  # Average FPS end-to-end
max_fps_end2end = 1 / np.min(frame_deltas)  # Maximum FPS observed
print(f"Average end-to-end FPS: {fps_estimate:.2f}")
print(f"Maximum end-to-end FPS: {max_fps_end2end:.2f}")
