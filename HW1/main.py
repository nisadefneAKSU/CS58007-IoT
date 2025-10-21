#!/usr/bin/env python3
# ================================================================
# CS 58007 IoT Homework 1
# Group 1 Members: Nisa Defne Aksu, Barkın Var, Pelin Karadal, Shahd Şerif
# Please look at following files:
#   Homework 1 Report.pdf (Contains plots, figures and explanations, and serves as an alternative to README.md)
#   .gitignore
#   requirements.txt
# ================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# Function to load Pyphox app data CSVs
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

    # Load files named in activities
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
def count_steps(filename, alpha=0.995, threshold_factor=0.6, window_size=30, min_step_interval=0.25):
    '''
    Parameters:
    -----------
    filename: CSV file containing accelerometer data.
    alpha: Low-pass filter coefficient for estimating gravity.
    threshold_factor: Multiplier for dynamic threshold based on signal standard deviation.
    window_size: Size of the sliding window for smoothing.
    min_step_interval: Minimum time between two steps to prevent double counting.

    Returns:
    -----------
    step_count: Estimated number of steps detected in the data.
    '''
    
    # Load accelerometer data
    df = pd.read_csv(filename)
    ax, ay, az = df["Acceleration x (m/s^2)"], df["Acceleration y (m/s^2)"], df["Acceleration z (m/s^2)"]
    t = df["Time (s)"].values

    # Raw acceleration magnitude
    acc_mag = np.sqrt(ax**2 + ay**2 + az**2)

    # Low-pass filter to estimate gravity
    gravity = np.zeros_like(acc_mag)
    gravity[0] = acc_mag[0]
    for i in range(1, len(acc_mag)):
        gravity[i] = alpha * gravity[i-1] + (1 - alpha) * acc_mag[i]

    # High-pass filter to remove gravity (motion only)
    acc = acc_mag - gravity

    # Manual smoothing (moving average) using a sliding window
    smooth = []
    for i in range(len(acc)):
        start = max(0, i - window_size // 2)
        end = min(len(acc), i + window_size // 2 + 1)
        smooth.append(np.mean(acc[start:end]))
    smooth = np.array(smooth)

    # Manual peak detection for step counting
    # A peak = Local maximum above dynamic threshold, separated by at least 'min_step_interval'
    peaks = []
    a0 = threshold_factor * np.std(smooth) # Adaptive threshold
    last_peak_idx = -int(min_step_interval / np.mean(np.diff(t)))  # Enforce min interval
    for i in range(1, len(smooth) - 1):
        if smooth[i] > a0 and smooth[i] > smooth[i - 1] and smooth[i] > smooth[i + 1]:
            if i - last_peak_idx >= int(min_step_interval / np.mean(np.diff(t))):
                peaks.append(i)
                last_peak_idx = i
    # Count detected peaks (steps)
    step_count = len(peaks)
    print(f"Estimated steps in {filename}: {step_count}")

    # Plot results for visualization
    plt.figure(figsize=(12, 6))
    plt.plot(t, acc_mag, label="Raw Acceleration Magnitude", color='gray', alpha=0.4)
    plt.plot(t, smooth, label="Smoothed (Motion Only)", color='blue')
    plt.axhline(y=a0, color='red', linestyle='--', label=f"Threshold = {a0:.2f}")
    plt.plot(t[peaks], smooth[peaks], 'go', label="Detected Steps")
    plt.xlabel("Time (s)")
    plt.ylabel("Acceleration (m/s²)")
    plt.title("Step Detection with Gravity Removal and Smoothing")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return step_count

# ================================================================
# PART 3: Pose Estimation for All 4 Activities (Accelerometer + Gyroscope)
# ================================================================
def estimate_pose_all():

    # Define the folders for each activity (each contains accelerometer.csv and gyroscope.csv)
    folders = ["sittingwithgyroscope", "standingwithgyroscope",
               "walkingwithgyroscope", "runningwithgyroscope"]

    alpha = 0.98  # Complementary filter coefficient (balance between gyro and accel data)

    # Loop through each activity folder
    for folder in folders:
        acc_file = os.path.join(folder, "accelerometer.csv")
        gyro_file = os.path.join(folder, "gyroscope.csv")

        # Load sensor data using helper function
        df_acc = load_data(acc_file)
        df_gyro = load_data(gyro_file)

        # Skip if either file is missing
        if df_acc is None or df_gyro is None:
            continue

        # Extract accelerometer data
        ax = df_acc["Acceleration x (m/s^2)"].values
        ay = df_acc["Acceleration y (m/s^2)"].values
        az = df_acc["Acceleration z (m/s^2)"].values

        # Extract gyroscope data (angular velocity in rad/s)
        gx = df_gyro["Gyroscope x (rad/s)"].values
        gy = df_gyro["Gyroscope y (rad/s)"].values
        gz = df_gyro["Gyroscope z (rad/s)"].values

        # Extract time (in seconds)
        t = df_acc["Time (s)"].values

        # Ensure both datasets have equal length
        length = min(len(t), len(gx))
        t = t[:length]
        ax = ax[:length]
        ay = ay[:length]
        az = az[:length]
        gx = gx[:length]
        gy = gy[:length]
        gz = gz[:length]

        # Initialize orientation angles
        roll = 0.0 # Rotation around X-axis
        pitch = 0.0 # Rotation around Y-axis
        yaw = 0.0 # Rotation around Z-axis

        # Lists to store estimated angles
        roll_list = []
        pitch_list = []
        yaw_list = []

        # Loop through all samples to integrate and fuse data
        for i in range(1, length):
            dt = t[i] - t[i-1] # Time difference between samples

            # Compute roll and pitch from accelerometer (static estimate based on gravity vector)
            roll_acc = np.arctan2(ay[i], az[i]) * 180 / np.pi
            pitch_acc = np.arctan2(-ax[i], np.sqrt(ay[i]**2 + az[i]**2)) * 180 / np.pi

            # Integrate gyroscope angular velocity to get orientation
            roll_gyro = roll + gx[i] * dt * 180 / np.pi
            pitch_gyro = pitch + gy[i] * dt * 180 / np.pi
            yaw += gz[i] * dt * 180 / np.pi # Yaw accumulated directly (no accel correction)

            # Complementary filter fusion
            # Combine fast gyro data and stable accelerometer data
            roll = alpha * roll_gyro + (1 - alpha) * roll_acc
            pitch = alpha * pitch_gyro + (1 - alpha) * pitch_acc

            # Store computed values
            roll_list.append(roll)
            pitch_list.append(pitch)
            yaw_list.append(yaw)

        # Plot results for the corresponding activity
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
    
