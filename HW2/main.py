import numpy as np
import pandas as pd
import os
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score # For choosing the k for aggregate data KNN via k-fold cross-validation
from collections import defaultdict
from pathlib import Path
from sklearn.base import BaseEstimator, ClassifierMixin # Allows integration with sklearn tools (in our case cross_val_score)
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
import warnings

'''
Part 1:
Implement KNN classifiers both for aggregate data and individual data.
Train the classifier using the training feature set. Test your classifier with the test data.
Compute precision, recall, and F1 manually (no built-in metrics).
'''

### Manual KNN classifier class (sklearn-compatible) implemented only for picking best k
class ManualKNNClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, k=5):
        # Constructor that stores the number of k
        self.k = k

    def fit(self, X, y):
        # Stores the training data and labels.
        self.X_train = np.asarray(X, dtype=float)
        self.y_train = np.asarray(y)                
        return self                                

    def predict(self, X):
        # Predicts labels for input samples using our manual KNN implementation.
        # Calls the function 'manual_knn_classifier' below. 
        return manual_knn_classifier(self.X_train, self.y_train, X, self.k)

### Manual KNN classifier using Euclidean distance, also efficient and vectorized to be faster
def manual_knn_classifier(X_train, y_train, X_test, k):
    X_train = np.asarray(X_train, dtype=float)
    X_test = np.asarray(X_test, dtype=float)
    y_train = np.asarray(y_train)
    predictions = [] # Store predicted labels for each testing

    # Precompute squared norms of training samples for efficient distance calculation
    train_sq = np.sum(X_train ** 2, axis=1)

    # Loop over each test sample
    for x in X_test:
        # Compute squared Euclidean distance using vectorized formula
        dists = np.sqrt(train_sq - 2 * X_train.dot(x) + np.sum(x ** 2))

        # Get indices of k smallest distances
        nearest_idx = np.argpartition(dists, k)[:k]

        # Get the labels of those k nearest samples
        nearest_labels = y_train[nearest_idx]

        # Count occurrences of each label and select the majority one
        values, counts = np.unique(nearest_labels, return_counts=True)
        predicted_label = values[np.argmax(counts)]
        predictions.append(predicted_label)

    return np.array(predictions)

### Find the best k using 5-fold cross-validation and use weighted F1-score as the evaluation metric
def find_best_k(X_train, y_train, k_values=range(1, 11)):
    best_k = None       # Best k value found so far
    best_f1 = -1.0      # Best F1-score found so far

    for k in k_values:
        # Initialize our manual KNN model with current k
        knn = ManualKNNClassifier(k=k)

        # Evaluate using 5-fold cross-validation with weighted F1-score
        f1_scores = cross_val_score(knn, X_train, y_train, cv=5, scoring='f1_weighted')

        # Compute the mean F1-score across all folds
        mean_f1 = f1_scores.mean()
        print(f"k ={k:2d} → Mean F1-score = {mean_f1:.4f}")

        # If this k gives a better mean F1, update best parameters
        if mean_f1 > best_f1:
            best_f1 = mean_f1
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

### Helper function to load raw signal files from the 'Inertial Signals' folder
def load_raw_data_files(split, base_path):
    """
    Loads the 9 raw signal files (body_acc, body_gyro, total_acc) for a given
    split ('train' or 'test') and stacks them into a single 3D numpy array.
    
    Output shape: (num_samples, 128_timesteps, 9_channels)
    """
    signals = []
    signal_names = [
        'body_acc_x', 'body_acc_y', 'body_acc_z',
        'body_gyro_x', 'body_gyro_y', 'body_gyro_z',
        'total_acc_x', 'total_acc_y', 'total_acc_z'
    ]
    
    for sig in signal_names:
        filename = base_path / split / "Inertial Signals" / f"{sig}_{split}.txt"
        signals.append(np.loadtxt(filename))
    
    # Stack along the third dimension (channels)
    # np.dstack stacks arrays along the third axis (depth)
    return np.dstack(signals)

### Load the raw UCI HAR dataset (Part 2 - Raw Data)
def load_har_raw_dataset(base_path):
    """
    Loads raw data for aggregate model.
    Converts labels from 1-6 to 0-5 for Keras/TensorFlow.
    """
    X_train = load_raw_data_files("train", base_path)
    X_test = load_raw_data_files("test", base_path)
    
    y_train = np.loadtxt(base_path / "train" / "y_train.txt").astype(int)
    y_test = np.loadtxt(base_path / "test" / "y_test.txt").astype(int)
    
    # IMPORTANT: Labels are 1-6. Convert to 0-5 for Keras
    # (which expects 0-indexed classes for sparse_categorical_crossentropy)
    y_train = y_train - 1
    y_test = y_test - 1
    
    return X_train, X_test, y_train, y_test

### Load raw user-specific HAR data (Part 2 - Raw Data)
def load_har_raw_user_dataset(base_path):
    """
    Loads and merges train/test raw data and subjects for user-specific models.
    Converts labels from 1-6 to 0-5.
    """
    X_train = load_raw_data_files("train", base_path)
    X_test = load_raw_data_files("test", base_path)
    
    y_train = np.loadtxt(base_path / "train" / "y_train.txt").astype(int) - 1 # 0-5
    y_test = np.loadtxt(base_path / "test" / "y_test.txt").astype(int) - 1 # 0-5
    
    subject_train = np.loadtxt(base_path / "train" / "subject_train.txt").astype(int)
    subject_test = np.loadtxt(base_path / "test" / "subject_test.txt").astype(int)
    
    # Concatenate train and test
    X_all = np.concatenate((X_train, X_test), axis=0)
    y_all = np.concatenate((y_train, y_test), axis=0)
    subject_all = np.concatenate((subject_train, subject_test), axis=0)
    
    return X_all, y_all, subject_all

### Build a 1D Convolutional Neural Network (CNN) model
def build_cnn_model(input_shape=(128, 9), n_classes=6):
    """
    Builds a simple 1D-CNN model using Keras.
    input_shape = (timesteps, channels)
    n_classes = number of output activities
    """
    model = Sequential([
        # 1st Conv block
        Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=input_shape),
        # 2nd Conv block
        Conv1D(filters=64, kernel_size=3, activation='relu'),
        # 1D Max Pooling
        MaxPooling1D(pool_size=2),
        # Dropout for regularization
        Dropout(0.5),
        # Flatten to feed into a dense layer
        Flatten(),
        # Fully connected layer
        Dense(100, activation='relu'),
        # Output layer (softmax for multi-class classification)
        Dense(n_classes, activation='softmax')
    ])
    
    # Compile the model
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy', # Use this loss because our y labels are integers (0-5)
        metrics=['accuracy']
    )
    return model

### Train and evaluate 1D-CNN for aggregate data
def dnn_for_aggregate_data(base_path, epochs=10, batch_size=32):
    print("=== 1D-CNN Classifier For Aggregate Raw Data ===")
    X_train, X_test, y_train, y_test = load_har_raw_dataset(base_path)
    # print(f"Train: {X_train.shape}, Test: {X_test.shape}")

    n_classes = len(np.unique(y_train)) # Should be 6 (0-5)
    input_shape = X_train.shape[1:]     # Should be (128, 9)
    
    model = build_cnn_model(input_shape, n_classes)
    print("\nModel Summary:")
    model.summary()
    
    # print(f"\nTraining 1D-CNN for {epochs} epochs...")
    # Train the model, holding out 20% of training data for validation
    model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.2,
        verbose=0 # Show progress bar
    )
    
    # print("\nEvaluating on test set...")
    # Get probabilities for each class
    y_pred_probs = model.predict(X_test, verbose=0)
    # Get the class with the highest probability
    y_pred = np.argmax(y_pred_probs, axis=1) 
    
    # We can reuse our manual metric function!
    # Note: y_test and y_pred are both 0-5
    precision, recall, f1, metrics = compute_weighted_metrics(y_test, y_pred)
    
    print("\n=== Manual Evaluation Metrics (Aggregate DNN) ===")
    print(f"Precision (weighted): {precision:.3f}")
    print(f"Recall (weighted):    {recall:.3f}")
    print(f"F1-score (weighted):  {f1:.3f}")

    # Print the confusion matrix
    labels = sorted(set(y_test))
    cm = np.zeros((len(labels), len(labels)), dtype=int)
    for yt, yp in zip(y_test, y_pred):
        cm[yt][yp] += 1 # y_test and y_pred are 0-5, so we can use them directly as indices
    
    print("\nConfusion Matrix (Labels 0-5):")
    print(pd.DataFrame(cm, index=labels, columns=labels))
    return

### Train and evaluate user-specific 1D-CNN models
def dnn_per_user(base_path, epochs=5, batch_size=32):
    print("\n=== Running Individual User-Specific 1D-CNN Classifiers ===")
    X_all, y_all, subject_all = load_har_raw_user_dataset(base_path)
    
    users = sorted(np.unique(subject_all))
    n_classes = 6 # Fixed for this problem (0-5)
    input_shape = (128, 9) # Fixed
    
    print(f"Total users: {len(users)}\n")
    user_metrics = []

    for user in users:
        print(f"--- User {user} ---")
        user_mask = (subject_all == user)
        X_user = X_all[user_mask]
        y_user = y_all[user_mask] # Already 0-5
        
        # Need enough samples to create a train/test split
        if len(X_user) < 20: 
            print(f"Skipping user {user} (too few samples).")
            continue

        try:
            # Create a train/test split for this user's data
            X_train, X_test, y_train, y_test = train_test_split(
                X_user, y_user, test_size=0.2, stratify=y_user, random_state=42
            )
        except ValueError:
            # This can happen if a user has < 2 samples for one activity class
            print(f"Stratification failed for user {user} (too few samples of one class). Splitting without stratification.")
            X_train, X_test, y_train, y_test = train_test_split(
                X_user, y_user, test_size=0.2, random_state=42
            )

        # Build a new, untrained model for this user
        model = build_cnn_model(input_shape, n_classes)
        
        # Train with fewer epochs for smaller, user-specific data
        print(f"Training 1D-CNN for user {user} ({len(X_train)} samples)...")
        model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_test, y_test), # Use test set as val set
            verbose=0 # Quieter output inside the loop
        )
        
        y_pred_probs = model.predict(X_test, verbose=0)
        y_pred = np.argmax(y_pred_probs, axis=1)
        
        manual_precision, manual_recall, manual_f1, _ = compute_weighted_metrics(y_test, y_pred)
        print(f"Manual  -> Precision: {manual_precision:.3f}, Recall: {manual_recall:.3f}, F1: {manual_f1:.3f}\n")
        
        user_metrics.append((user, manual_precision, manual_recall, manual_f1))

    # Summarize results across users
    metrics_df = pd.DataFrame(user_metrics, columns=["User", "Precision", "Recall", "F1"])
    avg_f1 = metrics_df["F1"].mean() if not metrics_df.empty else float("nan")

    print("=== Summary Per User (DNN) ===")
    print(metrics_df)
    print(f"\nAverage F1 across users: {avg_f1:.3f}")
    return metrics_df

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

    # print("\n===Part 1===\n")

    # X_train, X_test, y_train, y_test = load_har_aggregate_dataset(base_path)
    # best_k = find_best_k(X_train, y_train) # We found the best k by cross-validation
    # knn_for_aggregate_data(base_path, k=best_k) 

    # user_df = load_har_user_dataset(base_path)     
    # knn_per_user(user_df, k=5) # We chose the best k by trying k manually for user-specific (individual) data

    print("\n===Part 2===\n")

    warnings.filterwarnings("ignore", category=UserWarning) # Suppress TensorFlow warnings for cleaner output

    # Note: DNN training can be slow!
    # Set epochs to 1 or 2 for a quick test, 10-20 for a real run.
    dnn_epochs = 10 
    dnn_for_aggregate_data(base_path, epochs=dnn_epochs)
    
    # User-specific models train on less data, can use fewer epochs
    dnn_user_epochs = 6
    dnn_per_user(base_path, epochs=dnn_user_epochs)

    # print("\n===Part 3===\n")

    # # Training data for XOR
    # X = np.array([[0,0],[0,1],[1,0],[1,1]])

    # # Train OR perceptron
    # y_or = np.array([0,1,1,1])
    # w_or, b_or = train_perceptron(X, y_or)

    # # Train NAND perceptron
    # y_nand = np.array([1,1,1,0])
    # w_nand, b_nand = train_perceptron(X, y_nand)

    # # Train AND perceptron (used to combine OR and NAND outputs)
    # y_and = np.array([0,0,0,1])
    # w_and, b_and = train_perceptron(np.array([[0,0],[0,1],[1,0],[1,1]]), y_and)

    # # Display results in a table
    # print("Input A | Input B | XOR Output")
    # print("-----------------------------")
    # for xi in X:
    #     print(f"   {xi[0]}    |    {xi[1]}    |     {xor(xi)}")
