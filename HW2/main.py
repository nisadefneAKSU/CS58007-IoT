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

'''Part 1:
Implement KNN classifiers both for aggregate data and individual data
Train the classifier using the training feature set. Test your classifier with the test data. 
Compute precision, recall, and F1'''

# ---------------------------------------------------------------
# Find the best k using cross-validation (Aggregate level)
# ---------------------------------------------------------------
def find_best_k(X_train, y_train, k_values=range(1, 21)):
    # Load training and test data
    best_k = None
    best_f1 = 0

    for k in k_values:
        # Create KNN with k neighbors
        knn = KNeighborsClassifier(n_neighbors=k)
        # Weighted F1 using 5-fold cross-validation
        f1_scores = cross_val_score(knn, X_train, y_train, cv=5, scoring='f1_weighted')
        mean_f1 = f1_scores.mean()  # Average F1 across folds
        print(f"k={k}, Mean F1-score={mean_f1:.4f}")
        if mean_f1 > best_f1:
            best_f1 = mean_f1
            best_k = k

    print(f"\nBest k: {best_k} with F1-score={best_f1:.4f}")
    return best_k

# ---------------------------------------------------------------
# Load the UCI HAR dataset (Aggregate level)
# ---------------------------------------------------------------
def load_har_dataset(base_path):
    """
    X_train = np.loadtxt(os.path.join(base_path, "train", "X_train.txt"))
    y_train = np.loadtxt(os.path.join(base_path, "train", "y_train.txt")).astype(int)
    X_test = np.loadtxt(os.path.join(base_path, "test", "X_test.txt"))
    y_test = np.loadtxt(os.path.join(base_path, "test", "y_test.txt")).astype(int)

    return X_train, X_test, y_train, y_test
    """
    X_train = np.loadtxt(base_path / "train" / "X_train.txt")
    y_train = np.loadtxt(base_path / "train" / "y_train.txt").astype(int)
    X_test = np.loadtxt(base_path / "test" / "X_test.txt")
    y_test = np.loadtxt(base_path / "test" / "y_test.txt").astype(int)
    
    return X_train, X_test, y_train, y_test


# ---------------------------------------------------------------
# Train and evaluate KNNs (Aggregate level)
# ---------------------------------------------------------------
def run_knn_aggregate(base_path, k=5):
    print("=== Running Aggregate KNN Classifier ===")

    # Load dataset
    X_train, X_test, y_train, y_test = load_har_dataset(base_path)

    print(f"Train set: {X_train.shape}, Test set: {X_test.shape}")
    print("Training KNN classifier...")

    # Initialize and train KNN
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)

    # Predict on test data
    y_pred = knn.predict(X_test)

    # Evaluation metrics
    print("\n=== Classification Report ===")
    print(classification_report(y_test, y_pred, digits=4))

    print("=== Confusion Matrix ===")
    print(confusion_matrix(y_test, y_pred))

    return knn

# ---------------------------------------------------------------
# Load user-specific HAR data
# ---------------------------------------------------------------
def load_har_user_dataset(base_path):
    # Load training and test data
    X_train = np.loadtxt(os.path.join(base_path, "train", "X_train.txt"))
    y_train = np.loadtxt(os.path.join(base_path, "train", "y_train.txt")).astype(int)
    subject_train = np.loadtxt(os.path.join(base_path, "train", "subject_train.txt")).astype(int)

    X_test = np.loadtxt(os.path.join(base_path, "test", "X_test.txt"))
    y_test = np.loadtxt(os.path.join(base_path, "test", "y_test.txt")).astype(int)
    subject_test = np.loadtxt(os.path.join(base_path, "test", "subject_test.txt")).astype(int)

    # Merge train + test
    X_all = np.concatenate((X_train, X_test), axis=0)
    y_all = np.concatenate((y_train, y_test), axis=0)
    subj_all = np.concatenate((subject_train, subject_test), axis=0)

    # Convert to DataFrame
    X_all = pd.DataFrame(X_all, columns=[f"feature_{i}" for i in range(X_all.shape[1])])
    df = X_all.copy()
    df["label"] = y_all
    df["subject"] = subj_all

    return df



# This part is to calculate precision, recall, and f1 scores manually
def compute_weighted_metrics(y_true, y_pred): # we get the true labels and predicted labels as input
    
    labels = sorted(set(y_true)) # this is all unique class labels
    metrics = defaultdict(dict) # dictionary to store metrics for each class
    # each key maps to another dictionary by default, that's why we use defaultdict (no key errors :))
    support_total = 0 #total number of smaples
    weighted_precision = 0
    weighted_recall = 0
    weighted_f1 = 0

    for label in labels: # fore each class (sitting, walking,...) we compute
        
        # true positives aka correctly predicted class
        TP = sum((yt == label and yp == label) for yt, yp in zip(y_true, y_pred))

        # false positives aka predicted this class, actually another class
        FP = sum((yt != label and yp == label) for yt, yp in zip(y_true, y_pred))

        # false negatives aka predicted as another class but actually this class
        FN = sum((yt == label and yp != label) for yt, yp in zip(y_true, y_pred))

        # to see how many true samples belong to this class
        support = sum(yt == label for yt in y_true)


        # implementing the formulas here (also considering division by 0)
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0 # actual positives out of all posştives
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0 # actual positives out of trufly classifed positives and falsely classified negatives
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0


        # we store them like this, saving per class
        metrics[label] = {
            "support": support,
            "precision": precision,
            "recall": recall,
            "f1": f1
        }

        # here we multiply each score with support (which is the weight basically) and add to total
        support_total += support
        weighted_precision += support * precision
        weighted_recall += support * recall
        weighted_f1 += support * f1

    # and we divide by total support to get the weighted scores
    weighted_precision /= support_total
    weighted_recall /= support_total
    weighted_f1 /= support_total

    return weighted_precision, weighted_recall, weighted_f1, metrics



# ======================================================
# Train and evaluate user-specific KNNs
# ======================================================
def run_knn_per_user(df, k=5):
    # Get unique user IDs
    users = sorted(df["subject"].unique())
    print(f"\n=== Running Individual User-Specific KNN Classifiers ===")
    print(f"\nTotal users: {len(users)}\n")

    user_metrics = []

    print("Printing both sklearn built-in metrics and manual calculation metrics for comparison. \n")

    for user in users:
        print(f"--- User {user} ---")
        df_user = df[df["subject"] == user] # Filter for current user
        if len(df_user) < 10:
            print(f"Skipping user {user} (too few samples).")
            continue

        X = df_user.drop(columns=["label", "subject"])  # Features
        y = df_user["label"] # Labels

        # Split user data into train/test (80/20)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

        # Train KNN
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test) # Predict test labels

        # Evaluate metrics
        precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
        recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
        f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

        # manual metrics
        manual_precision, manual_recall, manual_f1, _ = compute_weighted_metrics(y_test, y_pred)

        
        print(f"SKLEARN -> Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}")
        print(f"MANUAL  -> Precision: {manual_precision:.3f}, Recall: {manual_recall:.3f}, F1: {manual_f1:.3f}\n")
        

        user_metrics.append((user, manual_precision, manual_recall, manual_f1))

    # Summary
    metrics_df = pd.DataFrame(user_metrics, columns=["User", "Precision", "Recall", "F1"])
    avg_f1 = metrics_df["F1"].mean() if not metrics_df.empty else float("nan")

    print("=== Summary (F1 per user) ===")
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
# Main Execution
# ---------------------------------------------------------------
if __name__ == "__main__":
    # Path to UCI HAR Dataset folder
    #base_path = "./UCI HAR Dataset"
    base_path = Path("./UCI HAR Dataset")

    
    print("\n===Part 1===\n")

    X_train, X_test, y_train, y_test = load_har_dataset(base_path)
    best_k = find_best_k(X_train, y_train)
    run_knn_aggregate(base_path, k=best_k) 

    user_df = load_har_user_dataset(base_path)     
    run_knn_per_user(user_df, k=5) # Chose the best k by trying k manually for user-specific data

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
