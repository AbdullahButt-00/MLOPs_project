import pandas as pd
import numpy as np
import pickle
import json
from scipy.stats import ks_2samp
from sklearn.utils import shuffle
import mlflow

class DriftDetector:
    def __init__(self, reference_data_path, preprocessor_path):
        self.reference_data = pd.read_excel(reference_data_path, sheet_name="E Comm")
        with open(preprocessor_path, 'rb') as f:
            self.preprocessor = pickle.load(f)
        
        self.reference_data = shuffle(self.reference_data, random_state=42)
        self.drift_threshold = 0.05  # p-value threshold
        
    def detect_drift(self, new_data):
        """
        Detect drift using Kolmogorov-Smirnov test for numerical features
        """
        drift_report = {}
        
        numeric_cols = self.reference_data.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if col != 'Churn']
        
        for col in numeric_cols:
            if col in new_data.columns:
                ref_values = self.reference_data[col].dropna()
                new_values = new_data[col].dropna()
                
                # KS test
                statistic, p_value = ks_2samp(ref_values, new_values)
                
                drift_report[col] = {
                    'statistic': float(statistic),
                    'p_value': float(p_value),
                    'drift_detected': p_value < self.drift_threshold
                }
        
        # Overall drift score
        drift_scores = [v['statistic'] for v in drift_report.values()]
        overall_drift = np.mean(drift_scores)
        
        return {
            'features': drift_report,
            'overall_drift_score': float(overall_drift),
            'features_with_drift': sum(1 for v in drift_report.values() if v['drift_detected']),
            'total_features': len(drift_report)
        }
    
    def log_drift_to_mlflow(self, drift_report):
        """Log drift detection results to MLflow"""
        with mlflow.start_run(run_name="drift_detection"):
            mlflow.log_metric("overall_drift_score", drift_report['overall_drift_score'])
            mlflow.log_metric("features_with_drift", drift_report['features_with_drift'])
            mlflow.log_dict(drift_report, "drift_report.json")

if __name__ == "__main__":
    detector = DriftDetector(
        reference_data_path="E Commerce Dataset.xlsx",
        preprocessor_path="federated_data/preprocessor.pkl"
    )
    
    # Simulate new data (in production, this would be incoming data)
    new_data = pd.read_excel("E Commerce Dataset.xlsx", sheet_name="E Comm").sample(500)
    
    drift_report = detector.detect_drift(new_data)
    print("Drift Detection Report:")
    print(json.dumps(drift_report, indent=2))
    
    if drift_report['features_with_drift'] > 3:
        print("\n⚠️  ALERT: Significant drift detected! Consider retraining the model.")