import numpy as np
import pandas as pd
import os
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from collections import defaultdict
from pathlib import Path

'''
Part 1:
Implement KNN classifiers both for aggregate data and individual data.
Train the classifier using the training feature set. Test your classifier with the test data.
Compute precision, recall, and F1 manually (no built-in metrics).
'''

### Manually implemented KNN without sklearn
def manual_knn_classifier(X_train, y_train, X_test, k):
    # Convert all data to NumPy arrays and ensure float dtype
    X_train = np.asarray(X_train, dtype=float)
    X_test = np.asarray(X_test, dtype=float)
    y_train = np.asarray(y_train)
    predictions = []

    for x in X_test: # Loop through each test sample
        # Compute Euclidean distance between this test sample and all train samples
        distances = np.sqrt(np.sum((X_train - x) ** 2, axis=1))

        # Get indices of k nearest neighbors (smallest distances)
        nearest_indices = np.argsort(distances)[:k]

        # Get the corresponding labels
        nearest_labels = y_train[nearest_indices]

        # Majority vote: choose the most common label
        values, counts = np.unique(nearest_labels, return_counts=True)
        predicted_label = values[np.argmax(counts)]

        # Append prediction
        predictions.append(predicted_label)

    return np.array(predictions)

### Find the optimal k for manual KNN using a held-out validation split
def find_best_k(X_train, y_train, k_values=range(1, 11)):
    X_train = np.asarray(X_train, dtype=float)
    y_train = np.asarray(y_train)

    # Split once into train/val
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.2, stratify=y_train, random_state=42
    )

    best_k = None
    best_f1 = -1.0

    print("Finding best k using validation fold:")
    for k in k_values:
        # Manual KNN predictions
        y_val_pred = manual_knn_classifier(X_tr, y_tr, X_val, k=k)
        # Manual metrics
        precision, recall, f1, _ = compute_weighted_metrics(y_val, y_val_pred)
        print(f"k = {k:2d} → validation F1 = {f1:.4f}")
        if f1 > best_f1:
            best_f1 = f1
            best_k = k

    print(f"\nPicked best k = {best_k} (Validation F1 = {best_f1:.4f})")
    return best_k

### Calculate precision, recall, and F1 manually for each class, then compute their weighted average based on class support
def compute_weighted_metrics(y_true, y_pred):
    labels = sorted(set(y_true))  # All unique activity labels
    metrics = defaultdict(dict)   # For storing
    total_support = 0             # Total number of samples
    weighted_precision = 0
    weighted_recall = 0
    weighted_f1 = 0

    for label in labels:
        # True positives, false positives, false negatives for each activity label
        TP = sum((yt == label and yp == label) for yt, yp in zip(y_true, y_pred))
        FP = sum((yt != label and yp == label) for yt, yp in zip(y_true, y_pred))
        FN = sum((yt == label and yp != label) for yt, yp in zip(y_true, y_pred))
        support = sum(yt == label for yt in y_true)

        # Manual formulas (also handling division by zero)
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        # Save the calculated metrics
        metrics[label] = {
            "support": support,
            "precision": precision,
            "recall": recall,
            "f1": f1
        }

        # Weighted totals (weighted by number of samples per class)
        total_support += support
        weighted_precision += support * precision
        weighted_recall += support * recall
        weighted_f1 += support * f1

    # Normalize by total number of samples
    weighted_precision /= total_support
    weighted_recall /= total_support
    weighted_f1 /= total_support

    return weighted_precision, weighted_recall, weighted_f1, metrics

### Load the UCI HAR dataset for aggregate data
def load_har_aggregate_dataset(base_path):
    # Load training and test sets
    X_train = np.loadtxt(base_path / "train" / "X_train.txt")
    y_train = np.loadtxt(base_path / "train" / "y_train.txt").astype(int)
    X_test = np.loadtxt(base_path / "test" / "X_test.txt")
    y_test = np.loadtxt(base_path / "test" / "y_test.txt").astype(int)
    
    return X_train, X_test, y_train, y_test

### Train and evaluate KNN for aggregate data and compute the evaluation metrics
def knn_for_aggregate_data(base_path, k=5):
    print("\n=== Manual KNN Classifier For Aggregate Data ===")
    X_train, X_test, y_train, y_test = load_har_aggregate_dataset(base_path)
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")

    print(f"\nTraining manual KNN with k = {k}...")
    y_pred = manual_knn_classifier(X_train, y_train, X_test, k)

    precision, recall, f1, metrics = compute_weighted_metrics(y_test, y_pred)
    print("\n=== Manual Evaluation Metrics (Aggregate) ===")
    print(f"Precision (weighted): {precision:.3f}")
    print(f"Recall (weighted):    {recall:.3f}")
    print(f"F1-score (weighted):  {f1:.3f}")

    # Print the confusion matrix
    labels = sorted(set(y_test))
    cm = np.zeros((len(labels), len(labels)), dtype=int)
    for yt, yp in zip(y_test, y_pred):
        cm[labels.index(yt)][labels.index(yp)] += 1
    print("\nConfusion Matrix:")
    print(pd.DataFrame(cm, index=labels, columns=labels))

    return

### Load user-specific HAR data
def load_har_user_dataset(base_path):
    # Combine train and test data into a single dataset and attache subject (user) IDs so that user-specific models can be trained
    X_train = np.loadtxt(os.path.join(base_path, "train", "X_train.txt"))
    y_train = np.loadtxt(os.path.join(base_path, "train", "y_train.txt")).astype(int)
    subject_train = np.loadtxt(os.path.join(base_path, "train", "subject_train.txt")).astype(int)

    X_test = np.loadtxt(os.path.join(base_path, "test", "X_test.txt"))
    y_test = np.loadtxt(os.path.join(base_path, "test", "y_test.txt")).astype(int)
    subject_test = np.loadtxt(os.path.join(base_path, "test", "subject_test.txt")).astype(int)

    # Merge train and test splits together
    X_all = np.concatenate((X_train, X_test), axis=0)
    y_all = np.concatenate((y_train, y_test), axis=0)
    subject_all = np.concatenate((subject_train, subject_test), axis=0)

    # Convert to DataFrame for easier filtering by subject
    X_all = pd.DataFrame(X_all, columns=[f"feature_{i}" for i in range(X_all.shape[1])])
    df = X_all.copy()
    df["label"] = y_all
    df["subject"] = subject_all

    return df

### Train and evaluate user-specific KNNs
def knn_per_user(df, k=5):
    # Train one KNN model per user and evaluates it manually
    users = sorted(df["subject"].unique())
    print(f"\n=== Running Individual User-Specific KNN Classifiers ===")
    print(f"\nTotal users: {len(users)}\n")

    user_metrics = []
    print("Printing both sklearn built-in metrics and manual calculation metrics for comparison. \n")

    for user in users:
        print(f"--- User {user} ---")
        df_user = df[df["subject"] == user]  # Filter only this user's data
        
        # Skip users with too few samples to split
        if len(df_user) < 10:
            print(f"Skipping user {user} (too few samples).")
            continue

        # Split into features and labels
        X = df_user.drop(columns=["label", "subject"])
        y = df_user["label"]

        # 80/20 train-test split, stratified by label distribution
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

        # Train and evaluate KNN
        y_pred = manual_knn_classifier(X_train, y_train, X_test, k)

        # Compare manual vs sklearn metrics
        precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
        recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
        f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

        manual_precision, manual_recall, manual_f1, _ = compute_weighted_metrics(y_test, y_pred)
        print(f"Sklearn -> Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}")
        print(f"Manual  -> Precision: {manual_precision:.3f}, Recall: {manual_recall:.3f}, F1: {manual_f1:.3f}\n")

        # Store user-level results
        user_metrics.append((user, manual_precision, manual_recall, manual_f1))

    # Summarize results across users
    metrics_df = pd.DataFrame(user_metrics, columns=["User", "Precision", "Recall", "F1"])
    avg_f1 = metrics_df["F1"].mean() if not metrics_df.empty else float("nan")

    print("=== Summary Per User ===")
    print(metrics_df)
    print(f"\nAverage F1 across users: {avg_f1:.3f}")

    return metrics_df

'''Part 2:
Train a deep neural network using the raw data. Train both an aggregate model and user-specific models
Hint: Segmentation now refers to how you build the frames that you feed into your model
Compute precision, recall, and F1'''




'''Part 3 (Extra credit):
Implement inference and training of 3 Perceptrons: OR, NAND and AND.
Combine them together to implement XOR'''

### Step function
def step(x):
    return 1 if x >= 0 else 0

### Train a perceptron
def train_perceptron(X, y, lr=0.1, epochs=10):
    w = np.zeros(X.shape[1])
    b = 0
    for _ in range(epochs):
        for xi, yi in zip(X, y):
            y_pred = step(np.dot(xi, w) + b)
            w += lr * (yi - y_pred) * xi
            b += lr * (yi - y_pred)
    return w, b


### XOR function
def xor(x):
    or_out = step(np.dot(x, w_or) + b_or)
    nand_out = step(np.dot(x, w_nand) + b_nand)
    return step(np.dot([or_out, nand_out], w_and) + b_and)

### Main execution
if __name__ == "__main__":
    # Path to UCI HAR Dataset folder
    base_path = Path("./UCI HAR Dataset")

    print("\n===Part 1===\n")

    X_train, X_test, y_train, y_test = load_har_aggregate_dataset(base_path)
    best_k = find_best_k(X_train, y_train) # We found the best k using validation fold
    knn_for_aggregate_data(base_path, k=best_k) 

    user_df = load_har_user_dataset(base_path)     
    knn_per_user(user_df, k=5) # We chose the best k by trying k manually for user-specific (individual) data

    print("\n===Part 2===\n")




    print("\n===Part 3===\n")

    # Training data for XOR
    X = np.array([[0,0],[0,1],[1,0],[1,1]])

    # Train OR perceptron
    y_or = np.array([0,1,1,1])
    w_or, b_or = train_perceptron(X, y_or)

    # Train NAND perceptron
    y_nand = np.array([1,1,1,0])
    w_nand, b_nand = train_perceptron(X, y_nand)

    # Train AND perceptron (used to combine OR and NAND outputs)
    y_and = np.array([0,0,0,1])
    w_and, b_and = train_perceptron(np.array([[0,0],[0,1],[1,0],[1,1]]), y_and)

    # Display results in a table
    print("Input A | Input B | XOR Output")
    print("-----------------------------")
    for xi in X:
        print(f"   {xi[0]}    |    {xi[1]}    |     {xor(xi)}")


