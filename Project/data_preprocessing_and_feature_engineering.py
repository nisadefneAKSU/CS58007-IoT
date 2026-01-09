import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

class OccupancyDataProcessor:
    """This class handles cleaning, normalization, feature extraction, and train/test split"""
    
    def __init__(self, csv_files):
        """Initialize processor with CSV file paths
        Args: csv_files (List of CSV file paths)"""

        self.csv_files = csv_files
        self.raw_data = None
        self.cleaned_data = None
        self.features = None
        self.scaler = StandardScaler()
        
    def load_data(self):
        """Load all CSV files and combine them"""
        dfs = []
        
        for file in self.csv_files:
            df = pd.read_csv(file)
            
            # Standardize column names (handle case variations)
            df.columns = df.columns.str.strip()  # Remove whitespace
            
            # Keep only required columns if they exist
            required_cols = ['Timestamp', 'Temperature', 'Humidity', 'Light', 'CO2', 'PIR', 'Microphone', 'Occupancy_Label']
            
            # Check which columns exist
            available_cols = [col for col in required_cols if col in df.columns]
            
            df = df[available_cols]
            
            # Remove rows where all sensor values are missing
            sensor_cols = ['Temperature', 'Humidity', 'Light', 'CO2', 'PIR', 'Microphone']
            existing_sensor_cols = [c for c in sensor_cols if c in df.columns]
            
            before_len = len(df)
            df = df.dropna(subset=existing_sensor_cols, how='all')
            removed = before_len - len(df)
            
            print(f"Loaded {file}: {len(df)} rows, {len(available_cols)} columns")
            if removed > 0:
                print(f"(Removed {removed} completely empty rows)")
            
            dfs.append(df)
        
        self.raw_data = pd.concat(dfs, ignore_index=True)
        print(f"\nTotal rows loaded: {len(self.raw_data)}")
        print(f"Columns in combined data: {list(self.raw_data.columns)}")
        
        # Show data quality summary
        print(f"Rows with complete data: {self.raw_data.dropna().shape[0]}")
        print(f"Missing values per column:")
        for col in self.raw_data.columns:
            missing = self.raw_data[col].isna().sum()
            if missing > 0:
                print(f"{col}: {missing} ({missing/len(self.raw_data)*100:.1f}%)")
            else:
                print(f"{col}: None")
        return self.raw_data
    
    def clean_data(self):
        """Clean and preprocess raw sensor data"""

        df = self.raw_data.copy()
        
        # Convert timestamp to datetime
        df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
        
        # Remove rows with invalid timestamps
        invalid_timestamps = df['Timestamp'].isna().sum()
        if invalid_timestamps > 0:
            print(f"Removed {invalid_timestamps} rows with invalid timestamps")
            df = df.dropna(subset=['Timestamp'])
        
        # Sort by timestamp
        df = df.sort_values('Timestamp').reset_index(drop=True)
        
        # Forward fill for small gaps (up to 3 consecutive missing)
        df = df.ffill(limit=3)
        
        # Remove remaining rows with missing values
        rows_before = len(df)
        df = df.dropna()
        rows_removed = rows_before - len(df)
        if rows_removed > 0:
            print(f"Removed {rows_removed} rows with missing values")
        
        # Remove duplicates
        duplicates = df.duplicated(subset=['Timestamp']).sum()
        df = df.drop_duplicates(subset=['Timestamp'])
        if duplicates > 0:
            print(f"Removed {duplicates} duplicate timestamps")
        
        # Check if we have enough data
        if len(df) < 10:
            print(f"\nWARNING: Only {len(df)} rows remaining after cleaning! This might indicate data quality issues.")
            return None
        
        # Remove outliers using IQR method
        numeric_cols = ['Temperature', 'Humidity', 'Light', 'CO2', 'PIR', 'Microphone']
        existing_numeric_cols = [col for col in numeric_cols if col in df.columns]
        
        print(f"\nRemoving outliers from: {existing_numeric_cols}")
        
        for col in existing_numeric_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 3 * IQR  # Using 3*IQR for sensor data
            upper_bound = Q3 + 3 * IQR
            
            outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
            if outliers > 0:
                print(f"{col}: removed {outliers} outliers")
            
            df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
        
        self.cleaned_data = df.reset_index(drop=True)
        print(f"\nClean dataset: {len(self.cleaned_data)} rows")
        return self.cleaned_data
    
    def extract_features(self, window_size=5):
        """Extract features from sensor data
        Args: window_size (Number of samples for rolling statistics)"""

        print(f"\nExtracting features (window size is {window_size})...")
        df = self.cleaned_data.copy()
        
        # Noise features
        print("Microphone features (noise)")
        # Mean and variance of noise
        df['noise_mean'] = df['Microphone'].rolling(window=window_size, min_periods=1).mean()
        df['noise_variance'] = df['Microphone'].rolling(window=window_size, min_periods=1).var()
        df['noise_std'] = df['Microphone'].rolling(window=window_size, min_periods=1).std()
        df['noise_max'] = df['Microphone'].rolling(window=window_size, min_periods=1).max()
        df['noise_min'] = df['Microphone'].rolling(window=window_size, min_periods=1).min()
        
        # CO2 features
        print("CO2 features")
        # CO2 level and rate of change
        df['co2_mean'] = df['CO2'].rolling(window=window_size, min_periods=1).mean()
        df['co2_delta'] = df['CO2'].diff()  # Rate of increase
        df['co2_delta_mean'] = df['co2_delta'].rolling(window=window_size, min_periods=1).mean()
        df['co2_variance'] = df['CO2'].rolling(window=window_size, min_periods=1).var()
        
        # Light features
        print("Light intensity features")
        # Light intensity statistics
        df['light_mean'] = df['Light'].rolling(window=window_size, min_periods=1).mean()
        df['light_variance'] = df['Light'].rolling(window=window_size, min_periods=1).var()
        df['light_std'] = df['Light'].rolling(window=window_size, min_periods=1).std()
        df['light_delta'] = df['Light'].diff()
        
        # Temperature and humidity features
        print("Temperature & humidity features")
        df['temp_mean'] = df['Temperature'].rolling(window=window_size, min_periods=1).mean()
        df['humidity_mean'] = df['Humidity'].rolling(window=window_size, min_periods=1).mean()
        df['temp_delta'] = df['Temperature'].diff()
        
        # PIR motion features
        print("PIR motion features")
        df['pir_sum'] = df['PIR'].rolling(window=window_size, min_periods=1).sum()
        df['pir_max'] = df['PIR'].rolling(window=window_size, min_periods=1).max()
        
        # Tıme-based features
        print("Time-based features")
        df['hour'] = df['Timestamp'].dt.hour
        df['day_of_week'] = df['Timestamp'].dt.dayofweek
        
        # Remove rows with NaN from rolling calculations
        df = df.dropna().reset_index(drop=True)
        
        self.features = df
        print(f"Feature extraction complete: {df.shape[1]} columns")
        return self.features
    
    def normalize_features(self):
        """Normalize features using StandardScaler"""

        print("\nNormalizing features...")
        
        # Select feature columns (exclude timestamp and label columns)
        exclude_cols = ['Timestamp', 'Occupancy_Label']
        feature_cols = [col for col in self.features.columns 
                       if col not in exclude_cols]
        
        print(f"Features to normalize: {len(feature_cols)} columns")
        
        # Fit and transform
        self.features[feature_cols] = self.scaler.fit_transform(self.features[feature_cols])
        
        print("Features are normalized.")
        return self.features
    
    def prepare_train_test_split(self, test_size=0.2, random_state=42):
        """Split dataset into train and test sets
        Args:
            test_size: Proportion of test set
            random_state: Random seed for reproducibility"""
        
        print(f"\nSplitting dataset (Test size: {test_size})...")
        
        # Define columns to exclude
        exclude_cols = ['Timestamp', 'Occupancy_Label']
        
        # Separate features and labels
        X = self.features.drop([col for col in exclude_cols if col in self.features.columns], axis=1)
        y = self.features['Occupancy_Label']
        
        # Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        print(f"Training set: {len(X_train)} samples")
        print(f"Test set: {len(X_test)} samples")
        print(f"\nClass distribution:")
        print(f"Train -> Occupied: {y_train.sum()}, Empty: {len(y_train) - y_train.sum()}")
        print(f"Test -> Occupied: {y_test.sum()}, Empty: {len(y_test) - y_test.sum()}")
        
        return X_train, X_test, y_train, y_test

    def save_datasets(self, output_dir='processed_data'):
        """Save processed datasets and metadata"""

        print(f"\nSaving processed datasets to '{output_dir}'...")
        
        Path(output_dir).mkdir(exist_ok=True)
        
        # Save clean data
        self.cleaned_data.to_csv(f'{output_dir}/clean_data.csv', index=False)
        print(f"Saved clean_data.csv")
        
        # Save features
        self.features.to_csv(f'{output_dir}/features.csv', index=False)
        print(f"Saved features.csv")
        
        # Prepare train/test split and save
        X_train, X_test, y_train, y_test = self.prepare_train_test_split()
        
        X_train.to_csv(f'{output_dir}/X_train.csv', index=False)
        X_test.to_csv(f'{output_dir}/X_test.csv', index=False)
        y_train.to_csv(f'{output_dir}/y_train.csv', index=False, header=True)
        y_test.to_csv(f'{output_dir}/y_test.csv', index=False, header=True)
        
        print(f"Saved X_train.csv, X_test.csv, y_train.csv, y_test.csv")
        
    def visualize_features(self, save_path='feature_analysis.png'):
        """Generate feature visualization plots"""

        print(f"\nGenerating visualizations...")
        
        fig, axes = plt.subplots(3, 2, figsize=(15, 12))
        fig.suptitle('Feature Analysis for Occupancy Detection', fontsize=16, fontweight='bold')
        
        # 1. Noise (microphone) distribution
        axes[0, 0].hist([self.features[self.features['Occupancy_Label']==0]['Microphone'],
                         self.features[self.features['Occupancy_Label']==1]['Microphone']], 
                        label=['Empty', 'Occupied'], bins=30, alpha=0.7)
        axes[0, 0].set_xlabel('Microphone Level')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Noise Level Distribution')
        axes[0, 0].legend()
        
        # 2. CO2 distribution
        axes[0, 1].hist([self.features[self.features['Occupancy_Label']==0]['CO2'],
                         self.features[self.features['Occupancy_Label']==1]['CO2']], 
                        label=['Empty', 'Occupied'], bins=30, alpha=0.7)
        axes[0, 1].set_xlabel('CO2 (ppm)')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('CO2 Level Distribution')
        axes[0, 1].legend()
        
        # 3. Light intensity distribution
        axes[1, 0].hist([self.features[self.features['Occupancy_Label']==0]['Light'],
                         self.features[self.features['Occupancy_Label']==1]['Light']], 
                        label=['Empty', 'Occupied'], bins=30, alpha=0.7)
        axes[1, 0].set_xlabel('Light Intensity')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Light Level Distribution')
        axes[1, 0].legend()
        
        # 4. Temperature distribution
        axes[1, 1].hist([self.features[self.features['Occupancy_Label']==0]['Temperature'],
                         self.features[self.features['Occupancy_Label']==1]['Temperature']], 
                        label=['Empty', 'Occupied'], bins=30, alpha=0.7)
        axes[1, 1].set_xlabel('Temperature (°C)')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title('Temperature Distribution')
        axes[1, 1].legend()
        
        # 5. CO2 delta (rate of change)
        axes[2, 0].hist([self.features[self.features['Occupancy_Label']==0]['co2_delta'],
                         self.features[self.features['Occupancy_Label']==1]['co2_delta']], 
                        label=['Empty', 'Occupied'], bins=30, alpha=0.7)
        axes[2, 0].set_xlabel('CO2 Rate of Change (ΔCO2)')
        axes[2, 0].set_ylabel('Frequency')
        axes[2, 0].set_title('CO2 Change Rate Distribution')
        axes[2, 0].legend()
        
        # 6. PIR motion
        axes[2, 1].hist([self.features[self.features['Occupancy_Label']==0]['PIR'],
                         self.features[self.features['Occupancy_Label']==1]['PIR']], 
                        label=['Empty', 'Occupied'], bins=20, alpha=0.7)
        axes[2, 1].set_xlabel('PIR Motion')
        axes[2, 1].set_ylabel('Frequency')
        axes[2, 1].set_title('PIR Motion Distribution')
        axes[2, 1].legend()
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to {save_path}")
        plt.close()

### Main execution
if __name__ == "__main__":
    print("Starting data preprocessing and feature engineering...")
    
    DATA_DIR = Path("collected_datasets")

    csv_files = [
        DATA_DIR / "room data - amphi.csv",
        DATA_DIR / "room data - crowded_and_bg_noise.csv",
        DATA_DIR / "room data - empty_to_human_fulllight.csv",
        DATA_DIR / "room data - empty_to_human_lightson.csv",
    ]
    
    # Initialize the processor
    processor = OccupancyDataProcessor(csv_files)
    # Step 1: Load data
    print(f"\n***Data Quality Summary:\n")
    processor.load_data()
    # Step 2: Clean data
    print(f"\n***Cleaning Data:")
    processor.clean_data()
    # Step 3: Extract features
    print(f"\n***Feature Extraction:")
    processor.extract_features(window_size=5)
    # Step 4: Normalize features
    print(f"\n***Normalizing Features:")
    processor.normalize_features()
    # Step 5: Save all datasets
    print(f"\n***Saving Datasets:")
    processor.save_datasets(output_dir='processed_data')
    # Step 6: Generate visualizations
    print(f"\n***Visualizing Features:")
    processor.visualize_features(save_path='feature_analysis.png')
    
    # Print feature list for reference
    print("\n***Extracted Features Summary:")
    feature_cols = [col for col in processor.features.columns if col not in ['Timestamp', 'Occupancy_Label']]
    
    categories = {
        "Noise Features": [f for f in feature_cols if 'noise' in f],
        "CO2 Features": [f for f in feature_cols if 'co2' in f],
        "Light Features": [f for f in feature_cols if 'light' in f],
        "Temperature/Humidity": [f for f in feature_cols if 'temp' in f or 'humidity' in f],
        "PIR Motion": [f for f in feature_cols if 'pir' in f],
        "Time-based": [f for f in feature_cols if f in ['hour', 'day_of_week']],
        "Raw Sensors": [f for f in feature_cols if f in ['Temperature', 'Humidity', 'Light', 'CO2', 'PIR', 'Microphone']]
    }
    
    for category, features in categories.items():
        if features:
            print(f"\n{category}:")
            for f in features:
                print(f"  • {f}")
    
    print("\nPreprocessing and feature engineering is completed.")
