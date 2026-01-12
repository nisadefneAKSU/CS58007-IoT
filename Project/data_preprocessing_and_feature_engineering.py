import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from pathlib import Path

class OccupancyDataProcessor:
    """This class handles cleaning, normalization, feature extraction, and train/test split"""
    
    def __init__(self, csv_files):
        """Initialize processor with CSV file paths
        Args: csv_files (List of CSV file paths)"""

        self.csv_files = csv_files  # Store the list of CSV file paths
        self.raw_data = None  # Store loaded data
        self.cleaned_data = None  # Store cleaned data
        self.features = None  # Store extracted features
        
    def load_data(self):
        """Load all CSV files and combine them"""
        dfs = [] # Create empty list to store individual dataframes
        
        for file in self.csv_files: # Iterate through each CSV file path
            df = pd.read_csv(file) # Read CSV file into a pandas dataframe
            
            # Standardize column names
            df.columns = df.columns.str.strip()  # Remove leading/trailing whitespace from column names
            
            # Keep only required columns (all of them)
            required_cols = ['Timestamp', 'Temperature', 'Humidity', 'Light', 'CO2', 'PIR', 'Microphone', 'Occupancy_Label']
            
            # Check which columns exist
            available_cols = [col for col in required_cols if col in df.columns]
            df = df[available_cols] # Select only the available columns
            
            sensor_cols = ['Temperature', 'Humidity', 'Light', 'CO2', 'PIR', 'Microphone'] # List of sensor column names
            existing_sensor_cols = [c for c in sensor_cols if c in df.columns] # Filter to only sensor columns
            
            before_len = len(df) # Length of dataframe before removing empty rows
            df = df.dropna(subset=existing_sensor_cols, how='all') # Drop rows where all sensor values are missing
            removed = before_len - len(df) # Calculate number of rows removed
            
            print(f"Loaded {file}: {len(df)} rows, {len(available_cols)} columns") # Print summary of loaded file
            if removed > 0: # Check if any rows were removed
                print(f"(Removed {removed} completely empty rows)") # Print number of empty rows removed
            
            dfs.append(df) # Add dataframe to list
        
        self.raw_data = pd.concat(dfs, ignore_index=True) # Concatenate all dataframes into one, resetting index
        print(f"\nTotal rows loaded: {len(self.raw_data)}") # Total number of rows in combined dataset
        print(f"Columns in combined data: {list(self.raw_data.columns)}") # List of columns in combined dataset
        print(f"Rows with complete data: {self.raw_data.dropna().shape[0]}") # Number of rows with no missing values
        print(f"Missing values per column:")
        for col in self.raw_data.columns: # Iterate through each column
            missing = self.raw_data[col].isna().sum() # Count missing values in column
            if missing > 0: # Check if column has any missing values
                print(f"{col}: {missing} ({missing/len(self.raw_data)*100:.1f}%)")  # Print column info of missing values
            else:
                print(f"{col}: None")  # Print that column has no missing values
        return self.raw_data  # Return the combined raw dataset
    
    def clean_data(self):
        """Clean and preprocess raw sensor data"""

        df = self.raw_data.copy()  # Create a copy of raw data to avoid modifying original
        
        # Convert timestamp to datetime
        df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
        
        invalid_timestamps = df['Timestamp'].isna().sum() # Count number of invalid timestamps
        if invalid_timestamps > 0: # Check if any invalid timestamps exist
            print(f"Removed {invalid_timestamps} rows with invalid timestamps") # Print number of invalid timestamps
            df = df.dropna(subset=['Timestamp']) # Remove rows with invalid timestamps
        
        df = df.sort_values('Timestamp').reset_index(drop=True) # Sort dataframe by timestamp and reset index
        
        df = df.ffill(limit=3) # Forward fill missing values up to 3 consecutive rows
        
        rows_before = len(df) # Store number of rows before removal
        df = df.dropna() # Drop all rows with any remaining missing values
        rows_removed = rows_before - len(df) # Calculate number of rows removed
        if rows_removed > 0: # Check if any rows were removed
            print(f"Removed {rows_removed} rows with missing values") # Print number of rows removed
        
        duplicates = df.duplicated(subset=['Timestamp']).sum() # Count duplicate timestamps
        df = df.drop_duplicates(subset=['Timestamp']) # Remove duplicate timestamps, keeping first occurrence
        if duplicates > 0: # Check if any duplicates were found
            print(f"Removed {duplicates} duplicate timestamps") # Print number of duplicates removed
        
        # Check if we have enough data
        if len(df) < 10:
            print(f"\nWARNING: Only {len(df)} rows remaining after cleaning! This might indicate data quality issues.")
            return None
        
        # Remove outliers using IQR method after temporal interpolation
        numeric_cols = ['Temperature', 'Humidity', 'Light', 'CO2', 'PIR', 'Microphone']
        existing_numeric_cols = [col for col in numeric_cols if col in df.columns]
        
        print(f"\nRemoving outliers from: {existing_numeric_cols}") # Print list of columns being checked for outliers
        
        for col in existing_numeric_cols: # Iterate through each numeric column
            Q1 = df[col].quantile(0.25) # Calculate first quartile (25th percentile)
            Q3 = df[col].quantile(0.75) # Calculate third quartile (75th percentile)
            IQR = Q3 - Q1 # Calculate interquartile range
            lower_bound = Q1 - 3 * IQR # Calculate lower bound using 3*IQR method
            upper_bound = Q3 + 3 * IQR # Calculate upper bound using 3*IQR method
            
            outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum() # Count number of outliers
            if outliers > 0:  # Check if any outliers were found
                print(f"{col}: removed {outliers} outliers")  # Print column name and number of outliers removed
            
            df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]  # Filter dataframe to remove outliers
        
        self.cleaned_data = df.reset_index(drop=True) # Store cleaned data and reset index
        print(f"\nClean dataset: {len(self.cleaned_data)} rows") # Print final number of rows in cleaned dataset
        return self.cleaned_data # Return the cleaned dataset
    
    def extract_features(self, window_size=5):
        """Extract features from sensor data
        Args: window_size (Number of samples for rolling statistics)""" 

        print(f"\nExtracting features (window size is {window_size})...")

        df = self.cleaned_data.copy()  # Create copy of cleaned data for feature extraction
        
        # Noise features
        print("Microphone features (noise)")
        df['noise_mean'] = df['Microphone'].rolling(window=window_size, min_periods=1).mean() # Calculate rolling mean of microphone values
        df['noise_variance'] = df['Microphone'].rolling(window=window_size, min_periods=1).var() # Calculate rolling variance of microphone values
        df['noise_std'] = df['Microphone'].rolling(window=window_size, min_periods=1).std() # Calculate rolling standard deviation of microphone values
        df['noise_max'] = df['Microphone'].rolling(window=window_size, min_periods=1).max() # Calculate rolling maximum of microphone values
        df['noise_min'] = df['Microphone'].rolling(window=window_size, min_periods=1).min() # Calculate rolling minimum of microphone values
        
        # CO2 features
        print("CO2 features")
        df['co2_mean'] = df['CO2'].rolling(window=window_size, min_periods=1).mean() # Calculate rolling mean of CO2 values
        df['co2_delta'] = df['CO2'].diff() # Calculate first-order difference (rate of change) of CO2
        df['co2_delta_mean'] = df['co2_delta'].rolling(window=window_size, min_periods=1).mean() # Calculate rolling mean of CO2 rate of change
        df['co2_variance'] = df['CO2'].rolling(window=window_size, min_periods=1).var() # Calculate rolling variance of CO2 values
        
        # Light features
        print("Light intensity features")
        df['light_mean'] = df['Light'].rolling(window=window_size, min_periods=1).mean() # Calculate rolling mean of light values
        df['light_variance'] = df['Light'].rolling(window=window_size, min_periods=1).var() # Calculate rolling variance of light values
        df['light_std'] = df['Light'].rolling(window=window_size, min_periods=1).std() # Calculate rolling standard deviation of light values
        df['light_delta'] = df['Light'].diff() # Calculate first-order difference (rate of change) of light
        
        # Temperature and humidity features
        print("Temperature & humidity features")
        df['temp_mean'] = df['Temperature'].rolling(window=window_size, min_periods=1).mean() # Calculate rolling mean of temperature values
        df['humidity_mean'] = df['Humidity'].rolling(window=window_size, min_periods=1).mean() # Calculate rolling mean of humidity values
        df['temp_delta'] = df['Temperature'].diff() # Calculate first-order difference (rate of change) of temperature
        
        # PIR motion features
        print("PIR motion features")
        df['pir_sum'] = df['PIR'].rolling(window=window_size, min_periods=1).sum() # Calculate rolling sum of PIR motion values
        df['pir_max'] = df['PIR'].rolling(window=window_size, min_periods=1).max() # Calculate rolling maximum of PIR motion values
        
        # Time-based features
        print("Time-based features")
        df['hour'] = df['Timestamp'].dt.hour # Extract hour of day from timestamp
        df['day_of_week'] = df['Timestamp'].dt.dayofweek # Extract day of week from timestamp (0=Monday, 6=Sunday)
        
        df = df.dropna().reset_index(drop=True) # Drop rows with NaN values and reset index
        
        self.features = df # Store feature dataframe as instance variable
        print(f"Feature extraction complete: {df.shape[1]} columns") # Print total number of columns after feature extraction
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
        X = self.features.drop([col for col in exclude_cols if col in self.features.columns], axis=1) # Create feature matrix by dropping excluded columns
        y = self.features['Occupancy_Label'] # Create label vector from Occupancy_Label column
        
        X_train, X_test, y_train, y_test = train_test_split(  # Split features and labels into train and test sets
            X, y, test_size=test_size, random_state=random_state, stratify=y 
        )
        
        print(f"Training set: {len(X_train)} samples")
        print(f"Test set: {len(X_test)} samples")
        print(f"\nClass distribution:")
        print(f"Train -> Occupied: {y_train.sum()}, Empty: {len(y_train) - y_train.sum()}")  # Print training set class counts
        print(f"Test -> Occupied: {y_test.sum()}, Empty: {len(y_test) - y_test.sum()}")  # Print test set class counts
        
        return X_train, X_test, y_train, y_test

    def save_datasets(self, output_dir='processed_data'):
        """Save processed datasets and metadata"""

        print(f"\nSaving processed datasets to '{output_dir}'...")
        
        Path(output_dir).mkdir(exist_ok=True)  # Create output directory if it doesn't exist
        
        # Save clean data
        self.cleaned_data.to_csv(f'{output_dir}/clean_data.csv', index=False)
        print(f"Saved clean_data.csv")
        
        # Save features
        self.features.to_csv(f'{output_dir}/features.csv', index=False)
        print(f"Saved features.csv")
        
        # Prepare train/test split and save
        X_train, X_test, y_train, y_test = self.prepare_train_test_split()
        
        X_train.to_csv(f'{output_dir}/X_train.csv', index=False) # Save training features to CSV without index column
        X_test.to_csv(f'{output_dir}/X_test.csv', index=False) # Save test features to CSV without index column
        y_train.to_csv(f'{output_dir}/y_train.csv', index=False, header=True) # Save training labels to CSV with header
        y_test.to_csv(f'{output_dir}/y_test.csv', index=False, header=True) # Save test labels to CSV with header
        
        print(f"Saved X_train.csv, X_test.csv, y_train.csv, y_test.csv")
        
    def visualize_features(self, save_path='feature_analysis.png'): 
        """Generate feature visualization plots""" 
        """Note: Visualizations are based on processed sensor values, not raw signals."""

        print(f"\nGenerating visualizations...")
        
        fig, axes = plt.subplots(3, 2, figsize=(15, 12))  # Create 3x2 subplot grid with specified figure size
        fig.suptitle('Feature Analysis for Occupancy Detection', fontsize=16, fontweight='bold')
        
        # 1. Noise (microphone) distribution
        axes[0, 0].hist([self.features[self.features['Occupancy_Label']==0]['Microphone'], # Create histogram for empty room microphone values
                         self.features[self.features['Occupancy_Label']==1]['Microphone']], # Create histogram for occupied room microphone values
                        label=['Empty', 'Occupied'], bins=30, alpha=0.7) 
        axes[0, 0].set_xlabel('Microphone Level') 
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Noise Level Distribution')
        axes[0, 0].legend() 
        
        # 2. CO2 distribution
        axes[0, 1].hist([self.features[self.features['Occupancy_Label']==0]['CO2'],  # Create histogram for empty room CO2 values
                         self.features[self.features['Occupancy_Label']==1]['CO2']],  # Create histogram for occupied room CO2 values
                        label=['Empty', 'Occupied'], bins=30, alpha=0.7)
        axes[0, 1].set_xlabel('CO2 (ppm)') 
        axes[0, 1].set_ylabel('Frequency') 
        axes[0, 1].set_title('CO2 Level Distribution') 
        axes[0, 1].legend() 
        
        # 3. Light intensity distribution 
        axes[1, 0].hist([self.features[self.features['Occupancy_Label']==0]['Light'],  # Create histogram for empty room light values
                         self.features[self.features['Occupancy_Label']==1]['Light']],  # Create histogram for occupied room light values
                        label=['Empty', 'Occupied'], bins=30, alpha=0.7) 
        axes[1, 0].set_xlabel('Light Intensity') 
        axes[1, 0].set_ylabel('Frequency') 
        axes[1, 0].set_title('Light Level Distribution') 
        axes[1, 0].legend() 
        
        # 4. Temperature distribution
        axes[1, 1].hist([self.features[self.features['Occupancy_Label']==0]['Temperature'],  # Create histogram for empty room temperature values
                         self.features[self.features['Occupancy_Label']==1]['Temperature']],  # Create histogram for occupied room temperature values
                        label=['Empty', 'Occupied'], bins=30, alpha=0.7) 
        axes[1, 1].set_xlabel('Temperature (°C)') 
        axes[1, 1].set_ylabel('Frequency') 
        axes[1, 1].set_title('Temperature Distribution')
        axes[1, 1].legend() 
        
        # 5. CO2 delta (rate of change) 
        axes[2, 0].hist([self.features[self.features['Occupancy_Label']==0]['co2_delta'],  # Create histogram for empty room CO2 change rate
                         self.features[self.features['Occupancy_Label']==1]['co2_delta']],  # Create histogram for occupied room CO2 change rate
                        label=['Empty', 'Occupied'], bins=30, alpha=0.7)  
        axes[2, 0].set_xlabel('CO2 Rate of Change (ΔCO2)') 
        axes[2, 0].set_ylabel('Frequency') 
        axes[2, 0].set_title('CO2 Change Rate Distribution') 
        axes[2, 0].legend() 
        
        # 6. PIR motion 
        axes[2, 1].hist([self.features[self.features['Occupancy_Label']==0]['PIR'],  # Create histogram for empty room PIR values
                         self.features[self.features['Occupancy_Label']==1]['PIR']],  # Create histogram for occupied room PIR values
                        label=['Empty', 'Occupied'], bins=20, alpha=0.7) 
        axes[2, 1].set_xlabel('PIR Motion') 
        axes[2, 1].set_ylabel('Frequency') 
        axes[2, 1].set_title('PIR Motion Distribution')
        axes[2, 1].legend() 
        
        plt.tight_layout()  # Adjust subplot spacing to prevent overlap
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to {save_path}")
        plt.close()  

### Main execution
if __name__ == "__main__":
    print("Starting data preprocessing and feature engineering...")
    
    DATA_DIR = Path("collected_datasets")  # Define path to directory containing datasets

    csv_files = [  # Create list of CSV file paths
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
    # Step 4: Save all datasets 
    print(f"\n***Saving Datasets:") 
    processor.save_datasets(output_dir='processed_data') 
    # Step 5: Generate visualizations 
    print(f"\n***Visualizing Features:") 
    processor.visualize_features(save_path='feature_analysis.png') 
    
    # Print feature list for reference
    print("\n***Extracted Features Summary:")
     # Get list of feature columns excluding timestamp and label
    feature_cols = [col for col in processor.features.columns if col not in ['Timestamp', 'Occupancy_Label']]
    
    categories = {  # Create dictionary
        "Noise Features": [f for f in feature_cols if 'noise' in f], # List all noise-related features
        "CO2 Features": [f for f in feature_cols if 'co2' in f], # List all CO2-related features
        "Light Features": [f for f in feature_cols if 'light' in f], # List all light-related features
        "Temperature/Humidity": [f for f in feature_cols if 'temp' in f or 'humidity' in f], # List all temperature and humidity features
        "PIR Motion": [f for f in feature_cols if 'pir' in f], # List all PIR motion features
        "Time-based": [f for f in feature_cols if f in ['hour', 'day_of_week']], # List all time-based features
        "Raw Sensors": [f for f in feature_cols if f in ['Temperature', 'Humidity', 'Light', 'CO2', 'PIR', 'Microphone']] # List all raw sensor values
    }
    
    for category, features in categories.items():  # Iterate through each feature category
        if features:  # Check if category has any features
            print(f"\n{category}:")  # Print category name
            for f in features:  # Iterate through features in category
                print(f"• {f}")  # Print feature name with bullet point
    
    print("\nPreprocessing and feature engineering is completed.")
