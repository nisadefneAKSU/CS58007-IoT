# 1. ML Model Training

This stage handles data preprocessing, feature engineering, model training, and evaluation. Trained models are exported as binary files for later deployment on edge devices.

### 1. Install dependencies

```
pip install -r server-requirements.txt
```


### 2. Run data preprocessing and feature engineering

```
python data_preprocessing_and_feature_engineering.py
```

This script cleans the raw sensor data and extracts engineered features used for training.

### 3. Train and evaluate machine learning models

```
python ml_and_evaluation.py
```

This script trains multiple ML models, evaluates their performance, and dumps the trained models as binary (.pkl) files for deployment.

# 2. Server Deployment

This stage deploys the trained model to Raspberry Pi sensor nodes and hosts a central server that exposes room occupancy data via a web application.

## Raspberry Pi (Client Node)

### 1. Install dependencies

```
pip install -r pi-requirements.txt
```

### 2. Run the sensor node

```
python node.py
```

This script performs local feature extraction, runs occupancy inference using the trained model, and sends updates to the server.

## Server (Web Application)

### 1. Start the server

```
python main.py
```

This launches the backend web application/API that receives sensor updates and serves real-time room occupancy information.