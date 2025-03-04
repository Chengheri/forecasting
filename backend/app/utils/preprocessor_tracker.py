from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
from .mlflow_utils import BaseModelTracker
from ..utils.logger import Logger

logger = Logger()

class PreprocessorTracker(BaseModelTracker):
    """Tracker for data preprocessing steps."""
    
    def __init__(self, experiment_name: str = "electricity_forecasting"):
        super().__init__("preprocessing", experiment_name)
    
    def log_preprocessing_config(self, config: Dict[str, Any]) -> None:
        """Log preprocessing configuration."""
        logger.info("Logging preprocessing configuration")
        self.log_model_params(config)
    
    def log_missing_values_stats(self, df: pd.DataFrame) -> None:
        """Log statistics about missing values."""
        missing_stats = df.isnull().sum().to_dict()
        missing_percentages = (df.isnull().sum() / len(df) * 100).to_dict()
        
        stats = {
            "total_rows": len(df),
            "missing_values": missing_stats,
            "missing_percentages": missing_percentages
        }
        
        self.log_model_params(stats)
    
    def log_anomaly_detection_stats(self, 
                                  method: str,
                                  n_anomalies: int,
                                  anomaly_percentage: float,
                                  params: Dict[str, Any]) -> None:
        """Log anomaly detection statistics."""
        stats = {
            "anomaly_detection_method": method,
            "n_anomalies": n_anomalies,
            "anomaly_percentage": anomaly_percentage,
            **params
        }
        
        self.log_model_params(stats)
    
    def log_feature_stats(self, df: pd.DataFrame) -> None:
        """Log statistics about features."""
        stats = {}
        
        for column in df.columns:
            if pd.api.types.is_numeric_dtype(df[column]):
                stats[f"{column}_mean"] = df[column].mean()
                stats[f"{column}_std"] = df[column].std()
                stats[f"{column}_min"] = df[column].min()
                stats[f"{column}_max"] = df[column].max()
                stats[f"{column}_skew"] = df[column].skew()
                stats[f"{column}_kurtosis"] = df[column].kurtosis()
        
        self.log_model_params(stats)
    
    def log_scaling_params(self, 
                          method: str,
                          columns: List[str],
                          params: Dict[str, Any]) -> None:
        """Log scaling parameters."""
        stats = {
            "scaling_method": method,
            "scaled_columns": columns,
            **params
        }
        
        self.log_model_params(stats)
    
    def log_feature_engineering(self, 
                              added_features: List[str],
                              lag_features: List[str],
                              rolling_features: List[str]) -> None:
        """Log feature engineering details."""
        stats = {
            "added_features": added_features,
            "lag_features": lag_features,
            "rolling_features": rolling_features
        }
        
        self.log_model_params(stats)
    
    def log_data_split(self, 
                      train_size: int,
                      test_size: int,
                      validation_size: Optional[int] = None) -> None:
        """Log data split information."""
        stats = {
            "train_size": train_size,
            "test_size": test_size,
            "validation_size": validation_size,
            "total_size": train_size + test_size + (validation_size or 0)
        }
        
        self.log_model_params(stats)
    
    def log_preprocessing_pipeline(self, steps: List[Dict[str, Any]]) -> None:
        """Log the complete preprocessing pipeline steps."""
        for i, step in enumerate(steps):
            self.log_model_params({f"pipeline_step_{i}": step}) 