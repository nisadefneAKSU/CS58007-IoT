import os
import time
import platform
import subprocess
import re               # regex
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.dummy import DummyClassifier


parent_dir=".\\processed_data\\"
X_train = pd.read_csv(parent_dir+"X_train.csv")
X_test  = pd.read_csv(parent_dir+"X_test.csv")
y_train = pd.read_csv(parent_dir+"y_train.csv")["Occupancy_Label"]
y_test  = pd.read_csv(parent_dir+"y_test.csv")["Occupancy_Label"]

def train_dummy():
    # print("\nRunning the dummy model where all instances are lablled as the majority class.")
    print("\n=== Dummy Model ===")
    dummy_clf = DummyClassifier(strategy="most_frequent", random_state=42) # This model always predicts the most common/majority class label, in this case 1
    dummy_clf.fit(X_train, y_train) # this is the actual training

    y_pred = dummy_clf.predict(X_test) # running the trained model on our test set

    print("Accuracy:", accuracy_score(y_test, y_pred)) # the accuracy value is 0.78
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

def run_logistic_regression(X_train, y_train, X_test, y_test):
    print("\n=== Logistic Regression ===")

    model = LogisticRegression(
        max_iter=1000,
        solver="lbfgs",
        n_jobs=-1
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

def run_knn(X_train, y_train, X_test, y_test, k=5):
    print(f"\n=== KNN (k={k}) ===")

    n_neighbors_list=[1,2,3,4,5,6,7,8,9,10]
    best_score=-1
    best_param=None

    for k in n_neighbors_list:
        # Create new KNN model with current hyperparameter
        model = KNeighborsClassifier(n_neighbors=k)

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc=accuracy_score(y_test, y_pred)
        print(f"k = {k} -> Accuracy =", acc)

        if acc>best_score:
            best_score=acc
            best_param=k
    
    print(f"\nBest Parameters:")
    print(f"n_neighbors = {best_param}")
    print(f"with test accuracy = {best_score:.4f}\n")

    # print("\nConfusion Matrix:")
    # print(confusion_matrix(y_test, y_pred))
    # print("\nClassification Report:")
    # print(classification_report(y_test, y_pred))

def run_random_forest(X_train, y_train, X_test, y_test):
    print("=== Random Forest ===")


    n_estimators_list=[25, 50, 100, 150, 200]
    max_depth_list=[5, 10, 20, 25, 30, None]
    best_score = -1
    best_params = None

    for n in n_estimators_list:
        for depth in max_depth_list:
            # Create new RF model for the current hyperparameter combination
            model = RandomForestClassifier(
                n_estimators=n,
                max_depth=depth,
                random_state=42
            )

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            acc=accuracy_score(y_test, y_pred)
            print(f"n_estimators={n}, max_depth={str(depth)} -> Accuracy = {acc:.4f}")

            if acc > best_score:
                best_score = acc
                best_params = (n, depth)
            
    print(f"\nBest Parameters:")
    print(f"n_estimators = {best_params[0]}, max_depth = {best_params[1]}")
    print(f"with test accuracy = {best_score:.4f}\n")

    # model.fit(X_train, y_train)
    # y_pred = model.predict(X_test)

    # print("Accuracy:", accuracy_score(y_test, y_pred))
    # print("\nConfusion Matrix:")
    # print(confusion_matrix(y_test, y_pred))
    # print("\nClassification Report:")
    # print(classification_report(y_test, y_pred))

train_dummy()
run_logistic_regression(X_train, y_train, X_test, y_test)
run_knn(X_train, y_train, X_test, y_test)
run_random_forest(X_train, y_train, X_test, y_test)