import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import joblib
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler

class OccupancyMLTrainer:
    """This class trains ML models and evaluates them to pick the best one."""
    
    def __init__(self, data_dir='processed_data'): # Initialize the trainer with data directory path
        """Initialize ML trainer
        Args: data_dir: directory containing processed datasets"""

        self.data_dir = data_dir  # Store the directory path containing processed datasets
        self.X_train = None  # Initialize training features attribute
        self.X_test = None  # Initialize test features attribute
        self.y_train = None  # Initialize training labels attribute
        self.y_test = None  # Initialize test labels attribute
        self.models = {}  # Initialize dictionary to store trained models
        self.results = {}  # Initialize dictionary to store evaluation results
        self.sensor_combinations = {}  # Initialize dictionary to store sensor combination results
        self.scaler = None  # Initialize scaler attribute for feature normalization
        self.X_train_scaled = None  # Initialize scaled training features attribute
        self.X_test_scaled = None  # Initialize scaled test features attribute

    def load_data(self):
        """Load processed training and test data"""
        print("Loading processed datasets...")
        
        self.X_train = pd.read_csv(f'{self.data_dir}/X_train.csv')  # Load training features from CSV file
        self.X_test = pd.read_csv(f'{self.data_dir}/X_test.csv')  # Load test features from CSV file
        self.y_train = pd.read_csv(f'{self.data_dir}/y_train.csv').squeeze()  # Load training labels from CSV and convert to Series
        self.y_test = pd.read_csv(f'{self.data_dir}/y_test.csv').squeeze()  # Load test labels from CSV and convert to Series
        
        print(f"Training samples: {len(self.X_train)}")
        print(f"Test samples: {len(self.X_test)}")
        print(f"Number of features: {self.X_train.shape[1]}")
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def scale_features(self):
        """Scale features for LR and KNN (not for Random Forest)"""
        self.scaler = StandardScaler() # Initialize StandardScaler for feature normalization
        self.X_train_scaled = self.scaler.fit_transform(self.X_train) # Fit scaler on training data and transform it
        self.X_test_scaled = self.scaler.transform(self.X_test) # Transform test data using fitted scaler
        print("Features scaled using StandardScaler (for Logistic Regression and KNN).")
    
    def train_models(self, best_params=None):
        """Train all ML models (optionally with tuned hyperparameters)"""
        print("\nTraining ML models:")

        # Logistic Regression
        print("1-Training Logistic Regression...") 
        lr_params = best_params.get('Logistic Regression', {}) if best_params else {}  # Get tuned parameters if available, otherwise use empty dict
        lr_model = LogisticRegression( 
            max_iter=1000,  # Set maximum number of iterations for convergence
            random_state=42,  # Set random seed for reproducibility
            **lr_params  # Unpack any additional tuned hyperparameters
        )
        lr_model.fit(self.X_train_scaled, self.y_train)  # Train Logistic Regression model on scaled training data
        self.models['Logistic Regression'] = lr_model  # Store trained model in models dictionary
        print("Logistic Regression is trained.")

        # Random Forest
        print("2-Training Random Forest...")
        rf_params = best_params.get('Random Forest', {}) if best_params else {}  # Get tuned parameters if available, otherwise use empty dict
        rf_model = RandomForestClassifier(
            #n_estimators=100,  # Set number of trees in the forest
            random_state=42,  # Set random seed for reproducibility
            n_jobs=-1,  # Use all available CPU cores for parallel processing
            **rf_params  # Unpack any additional tuned hyperparameters
        )
        rf_model.fit(self.X_train, self.y_train)  # Train Random Forest model on unscaled training data
        self.models['Random Forest'] = rf_model  # Store trained model in models dictionary
        print("Random Forest is trained.")

        # KNN
        print("3-Training KNN...")
        knn_params = best_params.get('KNN', {}) if best_params else {}  # Get tuned parameters if available, otherwise use empty dict
        knn_model = KNeighborsClassifier(**knn_params)  # Create KNN model instance with tuned parameters
        knn_model.fit(self.X_train_scaled, self.y_train)  # Train KNN model on scaled training data
        self.models['KNN'] = knn_model  # Store trained model in models dictionary
        print("KNN is trained.")

        print(f"All {len(self.models)} models trained successfully.")
        return self.models  # Return dictionary of all trained models

    def tune_hyperparameters(self, cv=5):
        """Tune hyperparameters for all models using cross-validated F1-score.
        Returns a dict of best parameters per model."""
        print("\nHyperparameter tuning (F1-score): ")

        skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)  # Create stratified k-fold cross-validator

        param_grids = {  # Define dictionary of parameter grids for each model
            'Logistic Regression': {
                'model': LogisticRegression(max_iter=1000, random_state=42),  # Base model for tuning
                'params': {  # Hyperparameter search space
                    'C': [0.1, 1.0, 10.0]  # Regularization strength values to try
                }
            },
            'Random Forest': {
                'model': RandomForestClassifier(  # Base model for tuning
                    random_state=42,  # Random seed for reproducibility
                    n_jobs=-1  # Use all CPU cores
                ),  
                'params': {  # Hyperparameter search space
                    'max_depth': [6, 8, 10, 12],  # Maximum depth of trees to try
                    'n_estimators': [20, 30, 40, 50],  # Tune number of trees
                    'min_samples_split': [2, 5],  # Minimum samples required to split node
                    'min_samples_leaf': [1, 2]  # Minimum samples required at leaf node
                }  
            },
            'KNN': {
                'model': KNeighborsClassifier(),  # Base model for tuning
                'params': {  # Hyperparameter search space
                    'n_neighbors': [3, 5, 7, 9],  # Number of neighbors to try
                    'weights': ['uniform', 'distance']  # Weight functions to try
                }
            } 
        } 

        best_params = {}  # Initialize dictionary to store best parameters for each model

        for model_name, config in param_grids.items():  # Iterate through each model configuration
            print(f"**Tuning {model_name}...")

            X_train_used = self.X_train  # Initialize training data (default to unscaled)

            if model_name in ['Logistic Regression', 'KNN']:  # Check if model requires scaled features
                # Use the trainer's scaler if available, otherwise fit new
                if self.scaler is None:
                    self.scaler = StandardScaler()
                    self.scaler.fit(self.X_train)
                X_train_used = self.scaler.transform(self.X_train)


            grid = GridSearchCV(  # Create grid search cross-validator
                estimator=config['model'],  # Set base model to tune
                param_grid=config['params'],  # Set hyperparameter search space
                scoring='f1',  # Use F1-score as optimization metric
                cv=skf,  # Use stratified k-fold cross-validation
                n_jobs=-1  # Use all CPU cores for parallel processing
            )

            grid.fit(X_train_used, self.y_train)  # Perform grid search on training data

            best_params[model_name] = grid.best_params_  # Store best parameters for this model

            print(f"Best F1: {grid.best_score_:.4f}")  # Print best cross-validated F1 score
            print(f"Best params: {grid.best_params_}")  # Print best hyperparameters found

        return best_params  # Return dictionary of best parameters for all models

    def evaluate_models(self):
        """Evaluate all trained models"""
        
        for model_name, model in self.models.items():  # Iterate through each trained model
            print(f"\n================================")
            print(f"Evaluating: {model_name}")
            print(f"================================")

            if model_name in ['Logistic Regression', 'KNN']:  # Check if model requires scaled features
                y_pred_train = model.predict(self.X_train_scaled)  # Predict on scaled training data
                y_pred_test = model.predict(self.X_test_scaled)  # Predict on scaled test data
            else:  # For models that don't require scaling (Random Forest)
                y_pred_train = model.predict(self.X_train)  # Predict on unscaled training data
                y_pred_test = model.predict(self.X_test)  # Predict on unscaled test data

            # Calculate metrics for training set
            train_metrics = {  # Create dictionary of training metrics
                'accuracy': accuracy_score(self.y_train, y_pred_train), # Calculate training accuracy
                'precision': precision_score(self.y_train, y_pred_train), # Calculate training precision
                'recall': recall_score(self.y_train, y_pred_train), # Calculate training recall
                'f1_score': f1_score(self.y_train, y_pred_train) # Calculate training F1-score
            }
            
            # Calculate metrics for test set
            test_metrics = {  # Create dictionary of test metrics
                'accuracy': accuracy_score(self.y_test, y_pred_test), # Calculate test accuracy
                'precision': precision_score(self.y_test, y_pred_test), # Calculate test precision
                'recall': recall_score(self.y_test, y_pred_test), # Calculate test recall
                'f1_score': f1_score(self.y_test, y_pred_test) # Calculate test F1-score
            }
            
            # Store results
            self.results[model_name] = { # Create results entry for this model
                'train_metrics': train_metrics, # Store training metrics
                'test_metrics': test_metrics, # Store test metrics
                'y_pred_train': y_pred_train, # Store training predictions
                'y_pred_test': y_pred_test, # Store test predictions
                'confusion_matrix': confusion_matrix(self.y_test, y_pred_test) # Store confusion matrix
            }
            
            print("Test set performance:")
            for metric, value in test_metrics.items():  # Iterate through each test metric
                print(f"{metric.replace('_', ' ').title()}: {value:.4f}")  # Print metric name and value
            
            cm = self.results[model_name]['confusion_matrix']  # Get confusion matrix for this model
            print("\nConfusion matrix:")
            print(f"True negatives (correct empty): {cm[0][0]}")  # Print true negative count
            print(f"False positives (wrong occupied): {cm[0][1]}")  # Print false positive count
            print(f"False negatives (wrong empty): {cm[1][0]}")  # Print false negative count
            print(f"True positives (correct occupied): {cm[1][1]}")  # Print true positive count
        
        return self.results

    def get_sensor_feature_groups(self):
        return {
            'Temperature': [c for c in self.X_train.columns if 'temp' in c.lower()], # Temperature, temp_mean, temp_delta
            'Humidity': [c for c in self.X_train.columns if 'humidity' in c.lower()], # Humidity, humidity_mean
            'Light': [c for c in self.X_train.columns if 'light' in c.lower()], # Light, light_mean, light_variance, light_std, light_detla
            'CO2': [c for c in self.X_train.columns if 'co2' in c.lower()], # CO2, co2_mean, co2_delta, co2_delta_mean, co2_variance
            'PIR': [c for c in self.X_train.columns if 'pir' in c.lower()], # PIR, pir_sum, pir_max
            'Noise': [c for c in self.X_train.columns if ('noise' in c.lower() or 'mic' in c.lower())], # Microphone, noise_mean, noise_variance, noise_std, noise_max, noise_min
            'Time': [c for c in self.X_train.columns if ('hour' in c.lower() or 'day' in c.lower())] # hour, day_of_week
        }

    def run_sensor_ablation(self, best_model_name, best_model, output_csv):
        """Compare model performance with different sensor exclusion combinations"""

        sensor_groups = self.get_sensor_feature_groups()
        experiments={
            'All sensors': list(self.X_train.columns),  # Use all available features
            
            'No Temperature': [c for c in self.X_train.columns if c not in sensor_groups['Temperature']],
            'No Humidity': [c for c in self.X_train.columns if c not in sensor_groups['Humidity']],
            'No Light': [c for c in self.X_train.columns if c not in sensor_groups['Light']],
            'No CO2': [c for c in self.X_train.columns if c not in sensor_groups['CO2']],
            'No PIR': [c for c in self.X_train.columns if c not in sensor_groups['PIR']],
            'No Noise': [c for c  in self.X_train.columns if c not in sensor_groups['Noise']],
            'No Time': [c for c in self.X_train.columns if c not in sensor_groups['Time']],

            'Only Temperature': [c for c in self.X_train.columns if c in sensor_groups['Temperature']],
            'Only Humidity': [c for c in self.X_train.columns if c in sensor_groups['Humidity']],
            'Only Light': [c for c in self.X_train.columns if c in sensor_groups['Light']],
            'Only CO2': [c for c in self.X_train.columns if c in sensor_groups['CO2']],
            'Only PIR': [c for c in self.X_train.columns if c in sensor_groups['PIR']],
            'Only Noise': [c for c  in self.X_train.columns if c in sensor_groups['Noise']],
            'Only Time': [c for c in self.X_train.columns if c  in sensor_groups['Time']],
        }

        results=[]
        print(f"\nTesting different sensor combinations for {best_model_name}:")

        for combo_name, features in experiments.items():
            if not features:  # Check if feature list is empty
                continue  # Skip this combination if no features

            print(f"Testing: {combo_name} ({len(features)} features)")

            X_train_subset = self.X_train[features]  # Select subset of training features
            X_test_subset = self.X_test[features]  # Select subset of test features

            # Clone model so original best_model is untouched
            model = clone(best_model)

            if best_model_name in ['Logistic Regression', 'KNN']:  # Check if model requires scaled features
                scaler = StandardScaler()  # Create new scaler instance
                X_train_subset = scaler.fit_transform(X_train_subset)  # Fit and transform training subset
                X_test_subset = scaler.transform(X_test_subset)  # Transform test subset

            model.fit(X_train_subset, self.y_train)  # Train cloned model on feature subset
            y_pred = model.predict(X_test_subset)  # Predict on test subset

            metrics = {  # Create dictionary of metrics for this combination
                'sensor_configuration': combo_name,
                'num_features': len(features),  # Store number of features used
                'accuracy': accuracy_score(self.y_test, y_pred),  # Calculate accuracy
                'precision': precision_score(self.y_test, y_pred),  # Calculate precision
                'recall': recall_score(self.y_test, y_pred),  # Calculate recall
                'f1_score': f1_score(self.y_test, y_pred)  # Calculate F1-score
            }

            self.sensor_combinations[combo_name] = metrics  # Store metrics for this combination
            results.append(metrics)

            print(f"Name: {combo_name:15s} -> Accuracy: {metrics['accuracy']:.4f} | F1: {metrics['f1_score']:.4f}") 

        df=pd.DataFrame(results)
        df.to_csv(output_csv, index=False)
        print(f"Saved ablation results to {output_csv}")
        return df # self.sensor_combinations

    def compare_sensor_combinations(self, best_model_name, best_model): 
        """Compare model performance with different sensor combinations""" 

        sensor_groups = {  # Define dictionary of sensor combinations to test
            'All sensors': list(self.X_train.columns),  # Use all available features
            'Without CO2': [c for c in self.X_train.columns if 'co2' not in c.lower()],  # Exclude CO2-related features
            'Without light': [c for c in self.X_train.columns if 'light' not in c.lower()],  # Exclude light-related features
            'Without microphone': [c for c in self.X_train.columns if 'noise' not in c.lower() and 'microphone' not in c.lower()],  # Exclude microphone-related features
            'Only motion (PIR)': [c for c in self.X_train.columns if 'pir' in c.lower()],  # Use only PIR motion features
            'CO2 + PIR only': [c for c in self.X_train.columns if 'co2' in c.lower() or 'pir' in c.lower()],  # Use only CO2 and PIR features
        } 

        print(f"\nTesting different sensor combinations for {best_model_name}:")

        for combo_name, features in sensor_groups.items():  # Iterate through each sensor combination
            if not features:  # Check if feature list is empty
                continue  # Skip this combination if no features

            print(f"Testing: {combo_name} ({len(features)} features)")

            X_train_subset = self.X_train[features]  # Select subset of training features
            X_test_subset = self.X_test[features]  # Select subset of test features

            # Clone model so original best_model is untouched
            model = clone(best_model)

            if best_model_name in ['Logistic Regression', 'KNN']:  # Check if model requires scaled features
                scaler = StandardScaler()  # Create new scaler instance
                X_train_subset = scaler.fit_transform(X_train_subset)  # Fit and transform training subset
                X_test_subset = scaler.transform(X_test_subset)  # Transform test subset

            model.fit(X_train_subset, self.y_train)  # Train cloned model on feature subset
            y_pred = model.predict(X_test_subset)  # Predict on test subset

            metrics = {  # Create dictionary of metrics for this combination
                'accuracy': accuracy_score(self.y_test, y_pred),  # Calculate accuracy
                'precision': precision_score(self.y_test, y_pred),  # Calculate precision
                'recall': recall_score(self.y_test, y_pred),  # Calculate recall
                'f1_score': f1_score(self.y_test, y_pred),  # Calculate F1-score
                'num_features': len(features)  # Store number of features used
            }

            self.sensor_combinations[combo_name] = metrics  # Store metrics for this combination

            print(f"Accuracy: {metrics['accuracy']:.4f} | F1: {metrics['f1_score']:.4f}") 

        return self.sensor_combinations

    def get_feature_importance(self, model_name, top_n=15):
        """Get feature importance from the given model (if supported)"""

        if model_name not in self.models:  # Check if model exists in trained models
            print(f"Model '{model_name}' not found!")
            return None

        model = self.models[model_name]  # Get the trained model

        # Random Forest
        if hasattr(model, 'feature_importances_'):  # Check if model has feature_importances_ attribute
            importances = model.feature_importances_  # Get feature importances from Random Forest

        # Logistic Regression
        elif hasattr(model, 'coef_'):  # Check if model has coef_ attribute
            importances = np.abs(model.coef_[0])  # Get absolute values of coefficients as importance

        else:  # If model doesn't support feature importance
            print(f"Model '{model_name}' does not support feature importance!")
            return None

        feature_names = self.X_train.columns  # Get feature names from training data

        importance_df = pd.DataFrame({  # Create dataframe for feature importances
            'feature': feature_names,  # Store feature names
            'importance': importances  # Store importance values
        }).sort_values(by='importance', ascending=False)  # Sort by importance in descending order

        print(f"\nTop {top_n} most important features ({model_name}):")
        for _, row in importance_df.head(top_n).iterrows():  # Iterate through top N features
            print(f"* {row['feature']:30s}: {row['importance']:.4f}")  # Print feature name and importance

        return importance_df

    def select_best_model(self):
        """Select the best performing model based on F1-score""" 
        print(f"\n--------------------------------") 
        
        best_model_name = None  # Initialize variable for best model name
        best_f1_score = -1  # Initialize best F1-score to negative value
        
        print("\nModel comparison (Test set F1-Score):") 
        for model_name, results in self.results.items():  # Iterate through each model's results
            f1 = results['test_metrics']['f1_score']  # Get F1-score for this model
            print(f"{model_name:25s}: {f1:.4f}")  # Print model name and F1-score
            
            if f1 > best_f1_score:  # Check if current F1-score is better than best
                best_f1_score = f1  # Update best F1-score
                best_model_name = model_name  # Update best model name
        
        print(f"\nBest model: {best_model_name}")
        best_results = self.results[best_model_name]['test_metrics']  # Get metrics for best model
        print(f"Accuracy: {best_results['accuracy']:.4f}")  
        print(f"Precision: {best_results['precision']:.4f}")  
        print(f"Recall: {best_results['recall']:.4f}") 
        print(f"F1-Score: {best_results['f1_score']:.4f}")  
        
        return best_model_name, self.models[best_model_name]  # Return best model name and model object
    
    def save_models(self, output_dir='trained_models'): 
        """Save all trained models""" 
        print(f"\nSaving trained models to '{output_dir}':")
        
        Path(output_dir).mkdir(exist_ok=True)  # Create output directory if it doesn't exist
        
        for model_name, model in self.models.items():  # Iterate through each trained model
            filename = model_name.replace(' ', '_').lower() + '.pkl'  # Create filename from model name
            filepath = f'{output_dir}/{filename}'  # Construct full file path
            joblib.dump(model, filepath)  # Save model to file using joblib
            print(f"Saved {filename}.")
        print("All models saved successfully.")
            
        # Save feature names
        feature_names = list(self.X_train.columns)  # Get list of feature names
        with open(f'{output_dir}/feature_names.json', 'w') as f:  # Open JSON file for writing
            json.dump(feature_names, f, indent=2)  # Write feature names
        print(f"Saved feature names in feature_names.json.")

    def generate_performance_report(self, best_model_name, output_dir='ml_results'): 
        """Generate comprehensive performance report"""
        
        Path(output_dir).mkdir(exist_ok=True)  # Create output directory if it doesn't exist
        
        report = {  # Initialize report dictionary
            'models': {},  # Initialize models section
            'sensor_combinations': self.sensor_combinations, # Add sensor combination results
            'best_model': None  # Initialize best model field
        } 
        
        for model_name, results in self.results.items():  # Iterate through each model's results
            report['models'][model_name] = {  # Create entry for this model
                'train_metrics': results['train_metrics'],  # Add training metrics
                'test_metrics': results['test_metrics'],  # Add test metrics
                'confusion_matrix': results['confusion_matrix'].tolist()  # Convert confusion matrix to list
            }
        
        report['best_model'] = best_model_name  # Store best model name in report
        
        with open(f'{output_dir}/performance_report.json', 'w') as f:  # Open JSON file for writing
            json.dump(report, f, indent=2)  # Write report to JSON file with indentation

        print(f"Saved performance_report.json.") 
        
        # Create performance comparison table
        comparison_df = pd.DataFrame({
            model_name: results['test_metrics']  # Add test metrics for each model
            for model_name, results in self.results.items()  # Iterate through results
        }).T  # Transpose to have models as rows
        
        comparison_df.to_csv(f'{output_dir}/model_comparison.csv')
        print(f"Saved model_comparison.csv.")
        
        if self.sensor_combinations: # Check if sensor combinations were tested
            sensor_df = pd.DataFrame(self.sensor_combinations).T  # Create dataframe from sensor combinations
            sensor_df.to_csv(f'{output_dir}/sensor_combinations.csv')  # Save to CSV
            print(f"Saved sensor_combinations.csv.") 
        
        return report
    

    def train_dummy_baseline_model(self,):
        """returns baseline accuracy, preciseness, recall, and f1-score values"""
        # Dummy Model
        print("\nTraining Dummy Model...")
        dum_model = DummyClassifier(strategy='most_frequent', random_state=42) # This model always predicts the most common/majority class label, in this case 1
        dum_model.fit(self.X_train, self.y_train)
        # self.models["Dummy Model"]=dum_model

        y_pred_train = dum_model.predict(self.X_train)
        y_pred_test = dum_model.predict(self.X_test)

        test_acc = accuracy_score(self.y_test, y_pred_test)
        test_prec = precision_score(self.y_test, y_pred_test)
        test_rec = recall_score(self.y_test, y_pred_test)  # Calculate recall
        test_f1 = f1_score(self.y_test, y_pred_test)  # Calculate F1-score

        return test_acc, test_prec, test_rec, test_f1

    def visualize_results(self, best_model_name, best_model, output_dir='ml_results'):
        """Generate comprehensive visualizations""" 
        print(f"\nGenerating visualizations:") 
        
        Path(output_dir).mkdir(exist_ok=True)  # Create output directory if it doesn't exist
        
        # 1. Model comparison bar chart 
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Figure 3. Model Performance Comparison', fontsize=16, fontweight='bold')
        
        metrics = ['accuracy', 'precision', 'recall', 'f1_score']  # Define list of metrics to visualize
        metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']  # Define display names for metrics

        baseline_accuracy, _, _, baseline_f1= self.train_dummy_baseline_model()

        for idx, (metric, name) in enumerate(zip(metrics, metric_names)): # Iterate through metrics and their display names with index
            ax = axes[idx // 2, idx % 2]  # Get subplot at position calculated from index (row, column)
            
            model_names = list(self.results.keys())  # Extract list of model names from results dictionary
            values = [self.results[m]['test_metrics'][metric] for m in model_names]  # Extract metric values for all models
            
            bars = ax.bar(model_names, values)  # Create bar chart with model names on x-axis and metric values on y-axis
            ax.set_ylabel(name, fontsize=12)  # Set y-axis label with metric display name
            ax.set_title(f'{name} Comparison', fontsize=14, fontweight='bold')  # Set subplot title with metric name
            ax.set_ylim([0, 1.05])  # Set y-axis limits
            ax.grid(axis='y', alpha=0.3)  # Add horizontal grid lines with 30% transparency

            if metric=="accuracy": 
                ax.axhline(y=baseline_accuracy, color="red", linestyle="--", linewidth=2, label="baseline")
                ax.text(0.99, baseline_accuracy+0.01, f"baseline:{baseline_accuracy:.2f}", color="red", ha="right", va="bottom", transform=ax.get_yaxis_transform())
            if metric=="f1_score": 
                ax.axhline(y=baseline_f1, color="red", linestyle="--", linewidth=2, label="baseline")
                ax.text(0.99, baseline_f1+0.01, f"baseline:{baseline_f1:.2f}", color="red", ha="right", va="bottom", transform=ax.get_yaxis_transform())

            # Add value labels on bars
            for bar in bars:  # Iterate through each bar in the chart
                height = bar.get_height()  # Get the height (value) of the bar
                ax.text(bar.get_x() + bar.get_width()/2., height, f'{height:.3f}', ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/model_comparison.png', dpi=300, bbox_inches='tight')
        print(f"Saved model_comparison.png.") 
        plt.close()
        
        # 2. Confusion matrices
        fig, axes = plt.subplots(1, 3, figsize=(18, 5)) # Create subplots
        fig.suptitle('Figure 4. Model Confusion Matrices', fontsize=16, fontweight='bold') # Add main title to the figure with bold formatting
        
        for idx, (model_name, results) in enumerate(self.results.items()): # Iterate through model results with index
            cm = results['confusion_matrix'] # Extract confusion matrix for current model
            
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', # Create heatmap
                       xticklabels=['Empty', 'Occupied'], # Set x-axis labels for predicted classes
                       yticklabels=['Empty', 'Occupied'], # Set y-axis labels for true classes
                       ax=axes[idx], cbar=True) # Specify subplot and include colorbar
            
            axes[idx].set_title(model_name, fontsize=12, fontweight='bold') # Set subplot title with model name
            axes[idx].set_ylabel('True Label') # Set y-axis label indicating true class
            axes[idx].set_xlabel('Predicted Label') # Set x-axis label indicating predicted class

        plt.tight_layout()
        plt.savefig(f'{output_dir}/confusion_matrices.png', dpi=300, bbox_inches='tight') 
        print(f"Saved confusion_matrices.png.")
        plt.close()
        
        # 3. Sensor combination analysis 
        if self.sensor_combinations:  # Check if sensor combination results exist
            fig, ax = plt.subplots(figsize=(12, 6))  # Create single subplot
            
            combo_names = list(self.sensor_combinations.keys())  # Extract list of sensor combination names
            f1_scores = [self.sensor_combinations[c]['f1_score'] for c in combo_names]  # Extract F1-scores for each combination
            num_features = [self.sensor_combinations[c]['num_features'] for c in combo_names]  # Extract number of features for each combination
            
            colors = plt.cm.viridis(np.linspace(0, 1, len(combo_names)))  # Generate array of colors from viridis colormap
            bars = ax.bar(combo_names, f1_scores, color=colors)  # Create bar chart with combination names and F1-scores
            
            ax.set_ylabel('F1-Score', fontsize=12)
            ax.set_title('Figure 5. Sensor Combination Performance', fontsize=14, fontweight='bold') 
            ax.set_ylim([0, 1.12])  # Set y-axis limits
            ax.grid(axis='y', alpha=0.3)  # Add horizontal grid lines with 30% transparency
            plt.xticks(rotation=25, ha='right', fontsize=14)
            
            # Add feature count labels 
            for bar, num_feat, f1 in zip(bars, num_features, f1_scores):  # Iterate through bars with their feature counts and F1-scores
                ax.text(bar.get_x() + bar.get_width()/2., f1 + 0.02, f'F1: {f1:.3f}\n({num_feat} features)', ha='center', va='bottom', fontsize=12) 
            
            plt.tight_layout()
            plt.savefig(f'{output_dir}/sensor_combinations.png', dpi=300, bbox_inches='tight')
            print(f"Saved sensor_combinations.png.")
            plt.close() 
        
        # 4. Feature importance
        importance_df = self.get_feature_importance(best_model_name, top_n=15)  # Get dataframe of top 15 most important features

        if importance_df is not None:  # Check if feature importance data is available
            fig, ax = plt.subplots(figsize=(10, 8))  # Create single subplot

            top_features = importance_df.head(15)  # Get first 15 rows (top features) from importance dataframe
            ax.barh(top_features['feature'], top_features['importance'], color='#2ecc71') # Create horizontal bar chart
            ax.set_xlabel('Importance', fontsize=12) # Set x-axis label
            ax.set_title(f'Figure 2. Top 15 Most Important Features ({best_model_name})', fontsize=14, fontweight='bold')
            ax.invert_yaxis() # Invert y-axis so highest importance appears at top
            ax.grid(axis='x', alpha=0.3) # Add vertical grid lines with 30% transparency

            plt.tight_layout()
            plt.savefig(f'{output_dir}/feature_importance.png', dpi=300, bbox_inches='tight')
            total_importance = top_features['importance'].sum()
            print(f"Sum of top 15 feature importances: {total_importance:.4f}")
            print(f"Saved feature_importance.png.")
            plt.close()


### Main execution
if __name__ == "__main__":
    print("Starting machine learning and evaluation...")
    
    # Initialize trainer
    trainer = OccupancyMLTrainer(data_dir='processed_data')
    # Step 1: Load data
    trainer.load_data()
    # Step 2: Scale features
    trainer.scale_features()
    # Step 3: Hyperparameter tuning
    best_params = trainer.tune_hyperparameters(cv=5)
    # Step 4a: Obtain baseline parameters by training a dummy model that classifies all instances as the majority class
    test_acc, _, _, test_f1 = trainer.train_dummy_baseline_model()
    print(f"Dummy model is trained, for baseline performance:\
            \nTest Accuracy -> {test_acc:.4f} Test F1-Score -> {test_f1:.4f}")
    # Step 4: Train models using best parameters
    trainer.train_models(best_params=best_params)
    # Step 5: Evaluate models
    trainer.evaluate_models()
    # Step 6: Select best model
    best_model_name, best_model = trainer.select_best_model()
    # Step 7: Compare sensor combinations with best model
    # trainer.compare_sensor_combinations(best_model_name, best_model)
    df=trainer.run_sensor_ablation(best_model_name, best_model, "ablation1.csv")
    latex_table = df.to_latex(
        index=False,
        float_format="%.4f",
        column_format="lccccc",
        caption="Sensor Ablation Study Using Fixed Random Forest Model",
        label="tab:sensor_ablation",
        bold_rows=False,
        longtable=False
    )
    # Step 8: Save models
    trainer.save_models(output_dir='trained_models')
    # Step 9: Generate performance report
    trainer.generate_performance_report(best_model_name, output_dir='ml_results')
    # Step 10: Generate visualizations
    trainer.visualize_results(best_model_name, best_model, output_dir='ml_results')
    
    print("\nML training and evaluation is done.\n")
    # print(latex_table) # this is for producing a nice table later
