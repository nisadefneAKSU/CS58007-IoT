#!/usr/bin/env python3
# ================================================================
# Activity Recognition with Accelerometer Data
# Group 1 Members: Nisa Defne Aksu, Barkın Var, Pelin Karadal, Şahd Şerif
# ================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# Function to load Pyphox data CSVs
def load_data(filename):
    if not os.path.exists(filename):
        print(f"File not found: {filename}")
        return None
    df = pd.read_csv(filename)
    df.columns = [c.strip() for c in df.columns]
    return df

# ================================================================
# PART 1: Activity Visualization & Feature Inspection
# ================================================================
def plot_activities():
    activities = ["standing", "sitting", "walking", "running"]
    plt.figure(figsize=(10, 6))

    for act in activities:
        fname = f"{act}.csv"
        df = load_data(fname)
        if df is None:
            continue

        # Compute absolute acceleration
        ax = df["Acceleration x (m/s^2)"].values
        ay = df["Acceleration y (m/s^2)"].values
        az = df["Acceleration z (m/s^2)"].values
        acc_abs = np.sqrt(ax**2 + ay**2 + az**2)
        df["Absolute acceleration (m/s^2)"] = acc_abs

        plt.plot(df["Time (s)"], acc_abs, label=act)

        # Print feature observations
        print(f"{act}: mean={np.mean(acc_abs):.2f} m/s², std={np.std(acc_abs):.2f} m/s², min={np.min(acc_abs):.2f}, max={np.max(acc_abs):.2f}")

    plt.title("Accelerometer Data by Activity")
    plt.xlabel("Time (s)")
    plt.ylabel("Absolute Acceleration (m/s²)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# ================================================================
# PART 2: Step Counting (from Walking Data) with Manual Sliding Window
# ================================================================
def count_steps(filename="walking.csv"):
    df = load_data(filename)
    if df is None:
        return

    t = df["Time (s)"].values
    ax = df["Acceleration x (m/s^2)"].values
    ay = df["Acceleration y (m/s^2)"].values
    az = df["Acceleration z (m/s^2)"].values

    # Compute absolute acceleration
    acc_abs = np.sqrt(ax**2 + ay**2 + az**2)

    # Remove gravity offset
    acc_no_g = acc_abs - np.mean(acc_abs)

    # Manual sliding window for smoothing
    window_size = 5
    smooth = []
    for i in range(len(acc_no_g)):
        start = max(0, i - window_size//2)
        end = min(len(acc_no_g), i + window_size//2 + 1)
        smooth.append(np.mean(acc_no_g[start:end]))
    smooth = np.array(smooth)

    # Threshold-based peak detection (manual implementation)
    threshold = smooth.std() * 1.0
    peaks = []
    for i in range(1, len(smooth)-1):
        if smooth[i] > threshold and smooth[i] > smooth[i-1] and smooth[i] > smooth[i+1]:
            peaks.append(i)

    step_count = len(peaks)
    print(f"Estimated step count from {filename}: {step_count}")

    # Plot with detected steps
    plt.figure(figsize=(10, 5))
    plt.plot(t, smooth, label="Filtered Acceleration")
    plt.plot(t[peaks], smooth[peaks], "ro", label="Detected Steps")
    plt.title(f"Step Detection from {filename}")
    plt.xlabel("Time (s)")
    plt.ylabel("Acceleration (m/s²)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# ================================================================
# PART 3: Pose Estimation for All 4 Activities (Accelerometer + Gyroscope)
# ================================================================
def estimate_pose_all():
    folders = ["sittingwithgyroscope", "standingwithgyroscope",
               "walkingwithgyroscope", "runningwithgyroscope"]

    alpha = 0.98  # Complementary filter coefficient

    for folder in folders:
        acc_file = os.path.join(folder, "accelerometer.csv")
        gyro_file = os.path.join(folder, "gyroscope.csv")

        df_acc = load_data(acc_file)
        df_gyro = load_data(gyro_file)

        if df_acc is None or df_gyro is None:
            continue

        ax = df_acc["Acceleration x (m/s^2)"].values
        ay = df_acc["Acceleration y (m/s^2)"].values
        az = df_acc["Acceleration z (m/s^2)"].values

        gx = df_gyro["Gyroscope x (rad/s)"].values
        gy = df_gyro["Gyroscope y (rad/s)"].values
        gz = df_gyro["Gyroscope z (rad/s)"].values

        t = df_acc["Time (s)"].values

        # Synchronize accelerometer and gyroscope lengths
        length = min(len(t), len(gx))
        t = t[:length]
        ax = ax[:length]
        ay = ay[:length]
        az = az[:length]
        gx = gx[:length]
        gy = gy[:length]
        gz = gz[:length]

        roll = 0.0
        pitch = 0.0
        yaw = 0.0

        roll_list = []
        pitch_list = []
        yaw_list = []

        for i in range(1, length):
            dt = t[i] - t[i-1]

            # Accelerometer angles
            roll_acc = np.arctan2(ay[i], az[i]) * 180 / np.pi
            pitch_acc = np.arctan2(-ax[i], np.sqrt(ay[i]**2 + az[i]**2)) * 180 / np.pi

            # Gyroscope integration
            roll_gyro = roll + gx[i] * dt * 180 / np.pi
            pitch_gyro = pitch + gy[i] * dt * 180 / np.pi
            yaw += gz[i] * dt * 180 / np.pi

            # Complementary filter
            roll = alpha * roll_gyro + (1 - alpha) * roll_acc
            pitch = alpha * pitch_gyro + (1 - alpha) * pitch_acc

            roll_list.append(roll)
            pitch_list.append(pitch)
            yaw_list.append(yaw)

        # Plot results for this activity
        plt.figure(figsize=(10, 5))
        plt.plot(t[1:], roll_list, label="Roll (°)")
        plt.plot(t[1:], pitch_list, label="Pitch (°)")
        plt.plot(t[1:], yaw_list, label="Yaw (°)")
        plt.title(f"Pose Estimation: {folder}")
        plt.xlabel("Time (s)")
        plt.ylabel("Angle (°)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        # Save results to CSV
        out_df = pd.DataFrame({
            "Time (s)": t[1:],
            "Roll (°)": roll_list,
            "Pitch (°)": pitch_list,
            "Yaw (°)": yaw_list
        })
        out_csv = os.path.join(folder, "pose_estimation.csv")
        out_df.to_csv(out_csv, index=False)
        print(f"Pose estimation saved for {folder} -> {out_csv}\n")

# ================================================================
# MAIN EXECUTION
# ================================================================
if __name__ == "__main__":

    print("=== Part 1: Activity Visualization & Feature Inspection ===")
    plot_activities()

    print("\n=== Part 2: Step Counting ===")
    filename = input("Enter the walking data filename (e.g., walking.csv): ")
    count_steps(filename)

    print("\n=== Part 3: Pose Estimation for All Activities ===")
    estimate_pose_all()


