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
    
    def __init__(self, data_dir='processed_data'):
        """Initialize ML trainer
        Args: data_dir: directory containing processed datasets"""

        self.data_dir = data_dir
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.models = {}
        self.results = {}
        self.sensor_combinations = {}
        self.scaler = None
        self.X_train_scaled = None
        self.X_test_scaled = None

    def load_data(self):
        """Load processed training and test data"""
        print("Loading processed datasets...")
        
        self.X_train = pd.read_csv(f'{self.data_dir}/X_train.csv')
        self.X_test = pd.read_csv(f'{self.data_dir}/X_test.csv')
        self.y_train = pd.read_csv(f'{self.data_dir}/y_train.csv').squeeze()
        self.y_test = pd.read_csv(f'{self.data_dir}/y_test.csv').squeeze()
        
        print(f"Training samples: {len(self.X_train)}")
        print(f"Test samples: {len(self.X_test)}")
        print(f"Number of features: {self.X_train.shape[1]}")
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def scale_features(self):
        """Scale features for LR and KNN (not for Random Forest)"""
        self.scaler = StandardScaler()
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        print("Features scaled using StandardScaler (for Logistic Regression and KNN).")
    
    def train_models(self, best_params=None):
        """Train all ML models (optionally with tuned hyperparameters)"""
        print("\nTraining ML models:")

        # Dummy Model
        print("0-Training Dummy Model...")
        dum_model = DummyClassifier(strategy='most_frequent', random_state=42) # This model always predicts the most common/majority class label, in this case 1
        dum_model.fit(self.X_train, self.y_train)
        self.models["Dummy Model"]=dum_model

        y_pred_train = dum_model.predict(self.X_train)
        y_pred_test = dum_model.predict(self.X_test)
        print(f"Dummy model is trained, for baseline perofrmance.\
              \nAccuracy -> {accuracy_score(self.y_test, y_pred_test)} F1-Score -> {f1_score(self.y_test, y_pred_test)}")

        # Logistic Regression
        print("1-Training Logistic Regression...")
        lr_params = best_params.get('Logistic Regression', {}) if best_params else {}
        lr_model = LogisticRegression(
            max_iter=1000,
            random_state=42,
            **lr_params
        )
        lr_model.fit(self.X_train_scaled, self.y_train)
        self.models['Logistic Regression'] = lr_model
        print("Logistic Regression is trained.")

        # Random Forest
        print("2-Training Random Forest...")
        rf_params = best_params.get('Random Forest', {}) if best_params else {}
        rf_model = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            n_jobs=-1,
            **rf_params
        )
        rf_model.fit(self.X_train, self.y_train)
        self.models['Random Forest'] = rf_model
        print("Random Forest is trained.")

        # KNN
        print("3-Training KNN...")
        knn_params = best_params.get('KNN', {}) if best_params else {}
        knn_model = KNeighborsClassifier(**knn_params)
        knn_model.fit(self.X_train_scaled, self.y_train)
        self.models['KNN'] = knn_model
        print("KNN is trained.")

        print(f"All {len(self.models)} models trained successfully.")
        return self.models

    
    def tune_hyperparameters(self, cv=5):
        """Tune hyperparameters for all models using cross-validated F1-score.
        Returns a dict of best parameters per model."""
        print("\nHyperparameter tuning (F1-score): ")

        skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)

        param_grids = {
            'Logistic Regression': {
                'model': LogisticRegression(max_iter=1000, random_state=42),
                'params': {
                    'C': [0.1, 1.0, 10.0]
                }
            },
            'Random Forest': {
                'model': RandomForestClassifier(
                    n_estimators=100,
                    random_state=42,
                    n_jobs=-1
                ),
                'params': {
                    'max_depth': [6, 8, 10, 12],
                    # 'n_estimators': [30, 50, 70],
                    'min_samples_split': [2, 5],
                    'min_samples_leaf': [1, 2]
                }
            },
            'KNN': {
                'model': KNeighborsClassifier(),
                'params': {
                    'n_neighbors': [3, 5, 7, 9],
                    'weights': ['uniform', 'distance']
                }
            }
        }

        best_params = {}

        for model_name, config in param_grids.items():
            print(f"**Tuning {model_name}...")

            X_train_used = self.X_train

            if model_name in ['Logistic Regression', 'KNN']:
                scaler = StandardScaler()
                X_train_used = scaler.fit_transform(self.X_train)

            grid = GridSearchCV(
                estimator=config['model'],
                param_grid=config['params'],
                scoring='f1',
                cv=skf,
                n_jobs=-1
            )

            grid.fit(X_train_used, self.y_train)

            best_params[model_name] = grid.best_params_

            print(f"Best F1: {grid.best_score_:.4f}")
            print(f"Best params: {grid.best_params_}")

        return best_params

    def evaluate_models(self):
        """Evaluate all trained models"""
        
        for model_name, model in self.models.items():
            print(f"\n================================")
            print(f"Evaluating: {model_name}")
            print(f"================================")

            if model_name in ['Logistic Regression', 'KNN']:
                y_pred_train = model.predict(self.X_train_scaled)
                y_pred_test = model.predict(self.X_test_scaled)
            else:
                y_pred_train = model.predict(self.X_train)
                y_pred_test = model.predict(self.X_test)

            # Calculate metrics for training set
            train_metrics = {
                'accuracy': accuracy_score(self.y_train, y_pred_train),
                'precision': precision_score(self.y_train, y_pred_train),
                'recall': recall_score(self.y_train, y_pred_train),
                'f1_score': f1_score(self.y_train, y_pred_train)
            }
            
            # Calculate metrics for test set
            test_metrics = {
                'accuracy': accuracy_score(self.y_test, y_pred_test),
                'precision': precision_score(self.y_test, y_pred_test),
                'recall': recall_score(self.y_test, y_pred_test),
                'f1_score': f1_score(self.y_test, y_pred_test)
            }
            
            # Store results
            self.results[model_name] = {
                'train_metrics': train_metrics,
                'test_metrics': test_metrics,
                'y_pred_train': y_pred_train,
                'y_pred_test': y_pred_test,
                'confusion_matrix': confusion_matrix(self.y_test, y_pred_test)
            }
            
            # Print results
            print("Test set performance:")
            for metric, value in test_metrics.items():
                print(f"{metric.replace('_', ' ').title()}: {value:.4f}")
            
            # Print confusion matrix
            cm = self.results[model_name]['confusion_matrix']
            print("\nConfusion matrix:")
            print(f"True negatives (correct empty): {cm[0][0]}")
            print(f"False positives (wrong occupied): {cm[0][1]}")
            print(f"False negatives (wrong empty): {cm[1][0]}")
            print(f"True positives (correct occupied): {cm[1][1]}")
        
        return self.results
    
    def compare_sensor_combinations(self, best_model_name, best_model):
        """Compare model performance with different sensor combinations"""

        sensor_groups = {
            'All sensors': list(self.X_train.columns),
            'Without CO2': [c for c in self.X_train.columns if 'co2' not in c.lower()],
            'Without light': [c for c in self.X_train.columns if 'light' not in c.lower()],
            'Without microphone': [c for c in self.X_train.columns if 'noise' not in c.lower() and 'microphone' not in c.lower()],
            'Only motion (PIR)': [c for c in self.X_train.columns if 'pir' in c.lower()],
            'CO2 + PIR only': [c for c in self.X_train.columns if 'co2' in c.lower() or 'pir' in c.lower()],
        }

        print(f"\nTesting different sensor combinations for {best_model_name}:")

        for combo_name, features in sensor_groups.items():
            if not features:
                continue

            print(f"Testing: {combo_name} ({len(features)} features)")

            X_train_subset = self.X_train[features]
            X_test_subset = self.X_test[features]

            # Clone model so original best_model is untouched
            model = clone(best_model)

            if best_model_name in ['Logistic Regression', 'KNN']:
                scaler = StandardScaler()
                X_train_subset = scaler.fit_transform(X_train_subset)
                X_test_subset = scaler.transform(X_test_subset)

            model.fit(X_train_subset, self.y_train)
            y_pred = model.predict(X_test_subset)

            metrics = {
                'accuracy': accuracy_score(self.y_test, y_pred),
                'precision': precision_score(self.y_test, y_pred),
                'recall': recall_score(self.y_test, y_pred),
                'f1_score': f1_score(self.y_test, y_pred),
                'num_features': len(features)
            }

            self.sensor_combinations[combo_name] = metrics

            print(f"Accuracy: {metrics['accuracy']:.4f} | F1: {metrics['f1_score']:.4f}")

        return self.sensor_combinations

    def get_feature_importance(self, model_name, top_n=15):
        """Get feature importance from the given model (if supported)"""

        if model_name not in self.models:
            print(f"Model '{model_name}' not found!")
            return None

        model = self.models[model_name]

        # Random Forest
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_

        # Logistic Regression
        elif hasattr(model, 'coef_'):
            importances = np.abs(model.coef_[0])

        else:
            print(f"Model '{model_name}' does not support feature importance!")
            return None

        feature_names = self.X_train.columns

        # Create dataframe
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values(by='importance', ascending=False)

        print(f"\nTop {top_n} most important features ({model_name}):")
        for _, row in importance_df.head(top_n).iterrows():
            print(f"* {row['feature']:30s}: {row['importance']:.4f}")

        return importance_df

    def select_best_model(self):
        """Select the best performing model based on F1-score"""
        print(f"\n--------------------------------")
        
        best_model_name = None
        best_f1_score = -1
        
        print("\nModel comparison (Test set F1-Score):")
        for model_name, results in self.results.items():
            f1 = results['test_metrics']['f1_score']
            print(f"{model_name:25s}: {f1:.4f}")
            
            if f1 > best_f1_score:
                best_f1_score = f1
                best_model_name = model_name
        
        print(f"\nBest model: {best_model_name}")
        best_results = self.results[best_model_name]['test_metrics']
        print(f"Accuracy: {best_results['accuracy']:.4f}")
        print(f"Precision: {best_results['precision']:.4f}")
        print(f"Recall: {best_results['recall']:.4f}")
        print(f"F1-Score: {best_results['f1_score']:.4f}")
        
        return best_model_name, self.models[best_model_name]
    
    def save_models(self, output_dir='trained_models'):
        """Save all trained models"""
        print(f"\nSaving trained models to '{output_dir}':")
        
        Path(output_dir).mkdir(exist_ok=True)
        
        for model_name, model in self.models.items():
            filename = model_name.replace(' ', '_').lower() + '.pkl'
            filepath = f'{output_dir}/{filename}'
            joblib.dump(model, filepath)
            print(f"Saved {filename}.")
        
        print("All models saved successfully.")
            
        # Save feature names
        feature_names = list(self.X_train.columns)
        with open(f'{output_dir}/feature_names.json', 'w') as f:
            json.dump(feature_names, f, indent=2)
        print(f"Saved feature names in feature_names.json.")

    def generate_performance_report(self, best_model_name, output_dir='ml_results'):
        """Generate comprehensive performance report"""
        
        Path(output_dir).mkdir(exist_ok=True)
        
        # Create summary report
        report = {
            'models': {},
            'sensor_combinations': self.sensor_combinations,
            'best_model': None
        }
        
        # Add model results
        for model_name, results in self.results.items():
            report['models'][model_name] = {
                'train_metrics': results['train_metrics'],
                'test_metrics': results['test_metrics'],
                'confusion_matrix': results['confusion_matrix'].tolist()
            }
        
        # Determine best model
        report['best_model'] = best_model_name
        
        # Save JSON report
        with open(f'{output_dir}/performance_report.json', 'w') as f:
            json.dump(report, f, indent=2)

        print(f"Saved performance_report.json.")
        
        # Create performance comparison table
        comparison_df = pd.DataFrame({
            model_name: results['test_metrics']
            for model_name, results in self.results.items()
        }).T
        
        comparison_df.to_csv(f'{output_dir}/model_comparison.csv')
        print(f"Saved model_comparison.csv.")
        
        # Create sensor combination comparison
        if self.sensor_combinations:
            sensor_df = pd.DataFrame(self.sensor_combinations).T
            sensor_df.to_csv(f'{output_dir}/sensor_combinations.csv')
            print(f"Saved sensor_combinations.csv.")
        
        return report
    
    def visualize_results(self, best_model_name, best_model, output_dir='ml_results'):
        """Generate comprehensive visualizations"""
        print(f"\nGenerating visualizations:")
        
        Path(output_dir).mkdir(exist_ok=True)
        
        # 1.Model comparison bar chart
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')
        
        metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        
        for idx, (metric, name) in enumerate(zip(metrics, metric_names)):
            ax = axes[idx // 2, idx % 2]
            
            model_names = list(self.results.keys())
            values = [self.results[m]['test_metrics'][metric] for m in model_names]
            
            bars = ax.bar(model_names, values)
            ax.set_ylabel(name, fontsize=12)
            ax.set_title(f'{name} Comparison', fontsize=14, fontweight='bold')
            ax.set_ylim([0, 1.05])
            ax.grid(axis='y', alpha=0.3)
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}',
                       ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/model_comparison.png', dpi=300, bbox_inches='tight')
        print(f"Saved model_comparison.png.")
        plt.close()
        
        # 2.Confusion matrices
        fig, axes = plt.subplots(1, 4, figsize=(18, 4))
        fig.suptitle('Confusion Matrices', fontsize=16, fontweight='bold')
        
        for idx, (model_name, results) in enumerate(self.results.items()):
            cm = results['confusion_matrix']
            
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=['Empty', 'Occupied'],
                       yticklabels=['Empty', 'Occupied'],
                       ax=axes[idx], cbar=True)
            
            axes[idx].set_title(model_name, fontsize=12, fontweight='bold')
            axes[idx].set_ylabel('True Label')
            axes[idx].set_xlabel('Predicted Label')
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/confusion_matrices.png', dpi=300, bbox_inches='tight')
        print(f"Saved confusion_matrices.png.")
        plt.close()
        
        # 3.Sensor combination analysis
        if self.sensor_combinations:
            fig, ax = plt.subplots(figsize=(12, 6))
            
            combo_names = list(self.sensor_combinations.keys())
            f1_scores = [self.sensor_combinations[c]['f1_score'] for c in combo_names]
            num_features = [self.sensor_combinations[c]['num_features'] for c in combo_names]
            
            colors = plt.cm.viridis(np.linspace(0, 1, len(combo_names)))
            bars = ax.bar(combo_names, f1_scores, color=colors)
            
            ax.set_ylabel('F1-Score', fontsize=12)
            ax.set_title('Sensor Combination Performance', 
                        fontsize=14, fontweight='bold')
            ax.set_ylim([0, 1.05])
            ax.grid(axis='y', alpha=0.3)
            plt.xticks(rotation=45, ha='right')
            
            # Add feature count labels
            for bar, num_feat, f1 in zip(bars, num_features, f1_scores):
                ax.text(bar.get_x() + bar.get_width()/2., f1 + 0.02,
                       f'F1: {f1:.3f}\n({num_feat} features)',
                       ha='center', va='bottom', fontsize=9)
            
            plt.tight_layout()
            plt.savefig(f'{output_dir}/sensor_combinations.png', dpi=300, bbox_inches='tight')
            print(f"Saved sensor_combinations.png.")
            plt.close()
        
        # 4.Feature importance
        importance_df = self.get_feature_importance(best_model_name, top_n=15)

        if importance_df is not None:
            fig, ax = plt.subplots(figsize=(10, 8))

            top_features = importance_df.head(15)
            ax.barh(top_features['feature'], top_features['importance'], color='#2ecc71')
            ax.set_xlabel('Importance', fontsize=12)
            ax.set_title(f'Top 15 Most Important Features ({best_model_name})',
                        fontsize=14, fontweight='bold')
            ax.invert_yaxis()
            ax.grid(axis='x', alpha=0.3)

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
    # Step 4: Train models using best parameters
    trainer.train_models(best_params=best_params)
    # Step 5: Evaluate models
    trainer.evaluate_models()
    # Step 6: Select best model
    best_model_name, best_model = trainer.select_best_model()
    # Step 7: Compare sensor combinations with best model
    trainer.compare_sensor_combinations(best_model_name, best_model)
    # Step 8: Save models
    trainer.save_models(output_dir='trained_models')
    # Step 9: Generate performance report
    trainer.generate_performance_report(best_model_name, output_dir='ml_results')
    # Step 10: Generate visualizations
    trainer.visualize_results(best_model_name, best_model, output_dir='ml_results')
    
    print("\nML training and evaluation is done.")