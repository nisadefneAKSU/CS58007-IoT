import os
import time
import platform
import subprocess
import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

### Here we do OS specific WiFi scan so that user can do it both with Windows and Linux
def scan_wifis():
    system = platform.system().lower() # Make platform name lowercase to match the cases
    results = []

    if system == "windows":
        # Windows command to list WiFi networks + BSSIDs + signal
        cmd = ["netsh", "wlan", "show", "networks", "mode=bssid"]
        # Run the Windows WiFi scan command, decode the text output, and split it into individual lines for parsing.
        output = subprocess.check_output(cmd).decode(errors="ignore").split("\n")

        ssid = None
        for line in output:
            line = line.strip() # Ensure consistent line-processing, remove trailing hidden \r if any, and prevent fingerprinting failures.
            # Extract SSID
            m = re.match(r"SSID\s+\d+\s*:\s*(.*)", line)
            if m:
                ssid = m.group(1).strip()
                continue
             # Extract BSSID (MAC address)
            m = re.match(r"BSSID\s+\d+\s*:\s*(.*)", line)
            if m:
                bssid = m.group(1).strip()
                continue
            # Extract RSSI (Signal strength)
            m = re.match(r"Signal\s*:\s*(\d+)%", line)
            if m:
                rssi = int(m.group(1))
                # Save one AP per row
                results.append({
                    "ssid": ssid or "",
                    "bssid": bssid or "",
                    "rssi": rssi
                })

    elif system == "linux":
        # Linux WiFi scan command -> requests SSID, BSSID, and SIGNAL columns
        cmd = ["nmcli", "-f", "SSID,BSSID,SIGNAL", "device", "wifi", "list"]
        # Run the command, decode the output, and split it into lines
        output = subprocess.check_output(cmd).decode(errors="ignore").split("\n")

        # Skip the first line (header) and iterate through each row of scan results
        for line in output[1:]: 
             # Split the row by 2+ spaces to separate SSID / BSSID / SIGNAL columns
            parts = re.split(r'\s{2,}', line.strip())
            # Ensure the row contains at least 3 valid fields
            if len(parts) >= 3:  
                ssid, bssid, rssi = parts[:3]
                # Ignore entries where BSSID is unavailable ("--")
                if bssid != "--": 
                    # Save one AP per row
                    results.append({ #
                        "ssid": ssid.strip(),
                        "bssid": bssid.strip(),
                        "rssi": int(rssi)
                    })

    else: # If the OS is neither Windows nor Linux print error and abort
        print("Unsupported OS!")
        return []

    return results


### Here we colect the fingerprint data (one sample means one scan)
def collect_data_for_each_room(room_name, scans_per_room=20, delay=1.0, outfile="wifi_dataset.csv"):
    rows = [] # List to store all collected rows from all scans

    print(f"\nCollecting {scans_per_room} full fingerprints for room: {room_name}")

    for i in range(scans_per_room):
        print(f"[{i+1}/{scans_per_room}] Scanning...") # Scan number for user to see in the terminal like [1/10], [2/10]...

        # Perform one full WiFi scan and return list of APs for that scan
        scan = scan_wifis()
        # Use current time as the group ID for this scan (1 scan = 1 sample)
        timestamp = int(time.time()) 

        # Each AP seen in this scan becomes one row in dataset
        for ap in scan:
            # We must store only rows with valid BSSID and RSSI
            if ap["bssid"] and ap["rssi"] is not None:
                rows.append({
                    "timestamp": timestamp, # Same timestamp is used to group all APs from this scan
                    "room": room_name, # Which room this scan belongs to that we got as user input
                    "bssid": ap["bssid"], # Unique AP ID
                    "ssid": ap.get("ssid", ""), # AP name (if hidden then we see in the dataset as "")
                    "rssi": ap["rssi"] # Signal strength
                })

        time.sleep(delay)   # Wait before doing the next scan (to avoid duplicate timestamps)

    # Convert all collected rows into a DataFrame
    df = pd.DataFrame(rows)
    # Check if output CSV already exists and set column names header only once
    file_exists = os.path.isfile(outfile)
    df.to_csv(outfile, mode="a", header=not file_exists, index=False)

    print(f"Saved {len(df)} rows for room {room_name}.\n")
    return df


### We have to clean the dataset before training and evaluation
def clean_dataset(df):
    # Count how many rows the dataset had before cleaning
    before = len(df)
    # Remove rows where BSSID or RSSI is missing (NaN values)
    df = df.dropna(subset=["bssid", "rssi"])
    # Remove rows where BSSID is an empty string ("")
    df = df[df["bssid"].str.len() > 0]
    # Remove duplicate WiFi readings: 
    # If room, BSSID, SSID, and RSSI are exactly the same, the row is redundant and provides no new information
    df = df.drop_duplicates(subset=["room", "bssid", "ssid", "rssi"])
    # Count rows after cleaning
    after = len(df)

    print(f"Cleaned the dataset: Removed {before - after} rows -> Now {after} rows.\n")
    return df


### We have to convert raw WiFi logs into a structured ML-ready matrix where each timestamp becomes one complete fingerprint (one sample) containing all BSSID signal values of a room.
def build_fingerprint_matrix(df):
    matrix = df.pivot_table(
        index="timestamp", # Row index is timestamp.
        columns="bssid", # Each BSSID becomes a column.
        values="rssi", # The cell value is the RSSI signal strength.
        aggfunc="mean", # If the same BSSID appears multiple times in a timestamp we take the mean to get a single RSSI value.
        fill_value=-100 # Missing BSSIDs are filled with -100 (weak/no signal).
    )

    # Add room label as a column (same timestamp means same room)
    room_map = df.groupby("timestamp")["room"].first()
    matrix["room"] = room_map

    return matrix


### For training, we use RSSI + known room labels to teach the model. For testing, we use only RSSI (no room info) -> predict room -> compare predictions to ground truth.
def train_and_evaluate(csv_file="wifi_dataset.csv"):
    # Load the Wi-Fi dataset from a CSV file
    df = pd.read_csv(csv_file)
    # Print the shape (rows, columns) of the dataset
    print("\nDataset loaded:", df.shape)

    # Clean the dataset
    df = clean_dataset(df)
    # Convert the dataset into a fingerprint matrix: rows = timestamps/scans, columns = BSSIDs, values = RSSI
    pivot = build_fingerprint_matrix(df)
    # Print the shape of the fingerprint matrix
    print("Fingerprint matrix shape:", pivot.shape)

    # Encode room labels into numerical values for classification
    label_encoder = LabelEncoder()
    pivot["room_encoded"] = label_encoder.fit_transform(pivot["room"])
    # Features: RSSI values for all BSSIDs
    X = pivot.drop(["room", "room_encoded"], axis=1)
    # Labels: encoded room numbers
    y = pivot["room_encoded"]

    # Count how many samples exist per room
    counts = pivot["room"].value_counts()
    print("\nSamples per room:\n", counts)

    # Split data into training and test sets (80/20), stratified by room label to maintain class proportions
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=0.2,
            stratify=y,
            random_state=42
        )
    except ValueError as e:
        print("Error:", e)
        print("\nTip: You need to collect more samples per room so that each class is represented in the test set.\n")

    # Initialize K-Nearest Neighbors classifier with 3 neighbors, using distance-weighted voting
    model = KNeighborsClassifier(n_neighbors=3,  weights="distance")
    # Train the model on the training data
    model.fit(X_train, y_train)
    # Predict room labels on the test set
    y_pred = model.predict(X_test)
    # Print overall accuracy of the classifier
    print("\nAccuracy:", accuracy_score(y_test, y_pred))

    # Get the unique room labels present in the test set
    unique_labels = sorted(set(y_test))
    # Print detailed classification report (precision, recall, f1-score) only for rooms in the test set
    print("\nClassification Report:\n")
    print(classification_report(
        y_test,
        y_pred,
        labels=unique_labels,
        target_names=label_encoder.inverse_transform(unique_labels),
        zero_division=0 # Avoids warnings
    ))

    # Compute confusion matrix for test predictions
    confusion_m = confusion_matrix(y_test, y_pred, labels=unique_labels)

    # Plot the confusion matrix as a heatmap for better visualization
    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion_m, annot=True, fmt="d", cmap="Blues",
                xticklabels=label_encoder.inverse_transform(unique_labels),
                yticklabels=label_encoder.inverse_transform(unique_labels))
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()


### Selection menu where user selects options from terminal and corresponding function is executed
def user_selection():
    print("\nOptions:")
    print("1 - Collect data")
    print("2 - Train and evaluate\n")

    choice = input("Please select an option (write 1 or 2): ").strip()

    if choice == "1":
        room = input("Enter room name: ").strip()
        collect_data_for_each_room(room)
    elif choice == "2":
        train_and_evaluate()
    else:
        print("Please enter a valid choice.")

if __name__ == "__main__":
    user_selection()