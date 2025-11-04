import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
import os
from sklearn.preprocessing import StandardScaler
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

# ---------------------------------------------------------------
# Find the best k using cross-validation (For aggregate data)
# ---------------------------------------------------------------
def find_best_k(X_train, y_train, k_values=range(1, 21)):
    best_k = None
    best_f1 = 0  # Track best F1-score found

    # Try each k value and evaluate using 5-fold cross-validation
    for k in k_values:
        knn = KNeighborsClassifier(n_neighbors=k)
        f1_scores = cross_val_score(knn, X_train, y_train, cv=5, scoring='f1_weighted')
        mean_f1 = f1_scores.mean()
        print(f"k = {k:2d} → Mean F1-score = {mean_f1:.4f}")
        
        # Update best k if a better F1-score is found
        if mean_f1 > best_f1:
            best_f1 = mean_f1
            best_k = k

    print(f"\nBest k = {best_k} (Mean F1 = {best_f1:.4f})")
    return best_k

# ---------------------------------------------------------------
# Manual implementation of weighted precision, recall, and F1 metrics
# ---------------------------------------------------------------
def compute_weighted_metrics(y_true, y_pred):
    # Calculates precision, recall, and F1 manually for each class,then computes their weighted averages based on class support.
    labels = sorted(set(y_true))  # All unique activity labels
    metrics = defaultdict(dict)   # Store metrics for each class
    support_total = 0             # Total number of samples
    weighted_precision = 0
    weighted_recall = 0
    weighted_f1 = 0

    for label in labels:
        # True positives, false positives, false negatives
        TP = sum((yt == label and yp == label) for yt, yp in zip(y_true, y_pred))
        FP = sum((yt != label and yp == label) for yt, yp in zip(y_true, y_pred))
        FN = sum((yt == label and yp != label) for yt, yp in zip(y_true, y_pred))
        support = sum(yt == label for yt in y_true)

        # Manual formulas (handling division by zero)
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        # Save class-level metrics
        metrics[label] = {
            "support": support,
            "precision": precision,
            "recall": recall,
            "f1": f1
        }

        # Weighted totals (weighted by number of samples per class)
        support_total += support
        weighted_precision += support * precision
        weighted_recall += support * recall
        weighted_f1 += support * f1

    # Normalize by total number of samples
    weighted_precision /= support_total
    weighted_recall /= support_total
    weighted_f1 /= support_total

    return weighted_precision, weighted_recall, weighted_f1, metrics

# ---------------------------------------------------------------
# Load the UCI HAR dataset (For aggregate data)
# ---------------------------------------------------------------
def load_har_dataset(base_path):
    # Loads training and test sets from the dataset. Each row corresponds to a pre-computed feature vector for one time window.
    X_train = np.loadtxt(base_path / "train" / "X_train.txt")
    y_train = np.loadtxt(base_path / "train" / "y_train.txt").astype(int)
    X_test = np.loadtxt(base_path / "test" / "X_test.txt")
    y_test = np.loadtxt(base_path / "test" / "y_test.txt").astype(int)
    
    return X_train, X_test, y_train, y_test

# ---------------------------------------------------------------
# Train and evaluate KNNs (For aggregate data using manual metric calculation)
# ---------------------------------------------------------------
def run_knn_aggregate(base_path, k=5):
    print("\n=== Aggregate KNN Classifier ===")

    # Load full dataset
    X_train, X_test, y_train, y_test = load_har_dataset(base_path)
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")

    # Train KNN with best k
    print(f"\nTraining KNN with k = {k} ...")
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)

    # Predict on test data
    print("Predicting test data...")
    y_pred = knn.predict(X_test)

    # Compute precision, recall, and F1 manually
    precision, recall, f1, metrics = compute_weighted_metrics(y_test, y_pred)

    print("\n=== Manual Evaluation Metrics (Aggregate) ===")
    print(f"Precision (weighted): {precision:.3f}")
    print(f"Recall (weighted):    {recall:.3f}")
    print(f"F1-score (weighted):  {f1:.3f}")

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    return knn

# ---------------------------------------------------------------
# Load user-specific HAR data (For individual data)
# ---------------------------------------------------------------
def load_har_user_dataset(base_path):
    # Combines train and test data into a single dataset and attaches subject (user) IDs so that user-specific models can be trained.
    X_train = np.loadtxt(os.path.join(base_path, "train", "X_train.txt"))
    y_train = np.loadtxt(os.path.join(base_path, "train", "y_train.txt")).astype(int)
    subject_train = np.loadtxt(os.path.join(base_path, "train", "subject_train.txt")).astype(int)

    X_test = np.loadtxt(os.path.join(base_path, "test", "X_test.txt"))
    y_test = np.loadtxt(os.path.join(base_path, "test", "y_test.txt")).astype(int)
    subject_test = np.loadtxt(os.path.join(base_path, "test", "subject_test.txt")).astype(int)

    # Merge train and test splits together
    X_all = np.concatenate((X_train, X_test), axis=0)
    y_all = np.concatenate((y_train, y_test), axis=0)
    subj_all = np.concatenate((subject_train, subject_test), axis=0)

    # Convert to DataFrame for easier filtering by subject
    X_all = pd.DataFrame(X_all, columns=[f"feature_{i}" for i in range(X_all.shape[1])])
    df = X_all.copy()
    df["label"] = y_all
    df["subject"] = subj_all

    return df

# ---------------------------------------------------------------
# Train and evaluate user-specific KNNs (For individual data using manual metric calculation)
# ---------------------------------------------------------------
def run_knn_per_user(df, k=5):
    # Trains one KNN model per user and evaluates it manually. This demonstrates personalization vs. aggregate modeling.
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

        # Train KNN
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)

        # Predict and evaluate
        y_pred = knn.predict(X_test)

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

# ---------------------------------------------------------------
# Step function
# ---------------------------------------------------------------
def step(x):
    return 1 if x >= 0 else 0

# ---------------------------------------------------------------
# Train a perceptron
# ---------------------------------------------------------------
def train_perceptron(X, y, lr=0.1, epochs=10):
    w = np.zeros(X.shape[1])
    b = 0
    for _ in range(epochs):
        for xi, yi in zip(X, y):
            y_pred = step(np.dot(xi, w) + b)
            w += lr * (yi - y_pred) * xi
            b += lr * (yi - y_pred)
    return w, b

# ---------------------------------------------------------------
# XOR function
# ---------------------------------------------------------------
def xor(x):
    or_out = step(np.dot(x, w_or) + b_or)
    nand_out = step(np.dot(x, w_nand) + b_nand)
    return step(np.dot([or_out, nand_out], w_and) + b_and)

# ---------------------------------------------------------------
# Main execution
# ---------------------------------------------------------------
if __name__ == "__main__":
    # Path to UCI HAR Dataset folder
    base_path = Path("./UCI HAR Dataset")

    print("\n===Part 1===\n")

    X_train, X_test, y_train, y_test = load_har_dataset(base_path)
    best_k = find_best_k(X_train, y_train) # We found the best k by cross-validation
    run_knn_aggregate(base_path, k=best_k) 

    user_df = load_har_user_dataset(base_path)     
    run_knn_per_user(user_df, k=5) # We chose the best k by trying k manually for user-specific (individual) data

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
