import pandas as pd
import numpy as np
import os.path

def get_file_path():
    path=input("Please enter the relative or absolute path of the csv file for analysis: ").strip('"')
    while not os.path.exists(path):
        print(f"The file {path} does not exist. Please try again!")
        path=input("Please enter the relative or absolute path of the csv file for analysis: ")
    return path

df = pd.read_csv(get_file_path())
# file_path="C:/Users/suuser/Desktop/IoT_HW3/latency_part2.csv"
# df = pd.read_csv(file_path)

df['total_latency'] = df['inference_time'] + df['rtt_seconds']
avg_latency = df['total_latency'].dropna().mean()
print(f"Average latency (s): {avg_latency:.3f}")

df['fps'] = 1 / df['inference_time']
max_fps = df['fps'].max()
print(f"Maximum inference rate (FPS): {max_fps:.2f}")

timestamps = df['capture_ts'].to_numpy()
frame_deltas = np.diff(timestamps)  # Time between consecutive frames
fps_estimate = 1 / np.mean(frame_deltas)  # Average FPS end-to-end
max_fps_end2end = 1 / np.min(frame_deltas)  # Maximum FPS observed
print(f"Average end-to-end FPS: {fps_estimate:.2f}")
print(f"Maximum end-to-end FPS: {max_fps_end2end:.2f}")