"""
Model and preprocessing tracking utilities for MLflow.
"""

from typing import Dict, Any, List, Optional, Union
import numpy as np
import pandas as pd
from .mlflow_utils import BaseTracker
from ..utils.logger import Logger

logger = Logger()

class PreprocessorTracker(BaseTracker):
    """Tracker for data preprocessing steps."""
    
    def __init__(self, experiment_name: str = "electricity_forecasting", run_name: Optional[str] = None):
        super().__init__(experiment_name, run_name)
    
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

class LSTMTracker(BaseTracker):
    """Tracker for LSTM model training and evaluation."""
    
    def __init__(self, experiment_name: str = "electricity_forecasting", run_name: Optional[str] = None):
        super().__init__(experiment_name, run_name)
    
    def log_model_params(self, params: Dict[str, Any]) -> None:
        """Log LSTM-specific parameters."""
        lstm_params = {
            "input_size": params.get("input_size"),
            "hidden_size": params.get("hidden_size"),
            "num_layers": params.get("num_layers"),
            "dropout": params.get("dropout", 0.2),
            "learning_rate": params.get("learning_rate"),
            "batch_size": params.get("batch_size"),
            "epochs": params.get("epochs")
        }
        super().log_model_params(lstm_params)

class ProphetTracker(BaseTracker):
    """Tracker for Prophet model training and evaluation."""
    
    def __init__(self, experiment_name: str = "electricity_forecasting", run_name: Optional[str] = None):
        super().__init__(experiment_name, run_name)
    
    def log_model_params(self, params: Dict[str, Any]) -> None:
        """Log Prophet-specific parameters."""
        prophet_params = {
            "changepoint_prior_scale": params.get("changepoint_prior_scale"),
            "seasonality_prior_scale": params.get("seasonality_prior_scale"),
            "holidays_prior_scale": params.get("holidays_prior_scale"),
            "seasonality_mode": params.get("seasonality_mode"),
            "changepoint_range": params.get("changepoint_range")
        }
        super().log_model_params(prophet_params)

class ARIMATracker(BaseTracker):
    """Tracker for ARIMA model training and evaluation."""
    
    def __init__(self, experiment_name: str = "electricity_forecasting", run_name: Optional[str] = None):
        super().__init__(experiment_name, run_name)
    
    def log_model_params(self, params: Dict[str, Any]) -> None:
        """Log ARIMA-specific parameters."""
        arima_params = {
            # Basic ARIMA parameters
            "p": params.get("p"),
            "d": params.get("d"),
            "q": params.get("q"),
            # Seasonal parameters
            "seasonal_p": params.get("seasonal_p"),
            "seasonal_d": params.get("seasonal_d"),
            "seasonal_q": params.get("seasonal_q"),
            "seasonal_period": params.get("seasonal_period"),
            # Additional parameters
            "trend": params.get("trend"),
            "enforce_stationarity": params.get("enforce_stationarity", True),
            "enforce_invertibility": params.get("enforce_invertibility", True),
            "concentrate_scale": params.get("concentrate_scale", False),
            "method": params.get("method", "lbfgs"),
            "maxiter": params.get("maxiter", 50)
        }
        super().log_model_params(arima_params)
    
    def log_training_metrics(self, metrics: Dict[str, float]) -> None:
        """Log ARIMA-specific training metrics."""
        training_metrics = {
            "aic": metrics.get("aic"),
            "bic": metrics.get("bic"),
            "llf": metrics.get("llf"),  # log-likelihood
            "rmse": metrics.get("rmse"),
            "mae": metrics.get("mae"),
            "mape": metrics.get("mape"),
            "residual_std": metrics.get("residual_std")
        }
        # Filter out None values
        training_metrics = {k: v for k, v in training_metrics.items() if v is not None}
        super().log_metrics(training_metrics)
    
    def log_model_diagnostics(self, diagnostics: Dict[str, Any]) -> None:
        """Log model diagnostic information."""
        diagnostic_metrics = {
            "residual_autocorrelation": diagnostics.get("residual_autocorrelation"),
            "heteroskedasticity_test": diagnostics.get("heteroskedasticity_test"),
            "normality_test": diagnostics.get("normality_test"),
            "seasonal_strength": diagnostics.get("seasonal_strength")
        }
        super().log_metrics(diagnostic_metrics)
    
    def log_forecast_metrics(self, metrics: Dict[str, float]) -> None:
        """Log forecast evaluation metrics."""
        forecast_metrics = {
            "forecast_rmse": metrics.get("rmse"),
            "forecast_mae": metrics.get("mae"),
            "forecast_mape": metrics.get("mape"),
            "forecast_r2": metrics.get("r2"),
            "forecast_coverage": metrics.get("coverage")  # For confidence intervals
        }
        super().log_metrics(forecast_metrics)

class LightGBMTracker(BaseTracker):
    """Tracker for LightGBM model training and evaluation."""
    
    def __init__(self, experiment_name: str = "electricity_forecasting", run_name: Optional[str] = None):
        super().__init__(experiment_name, run_name)
    
    def log_model_params(self, params: Dict[str, Any]) -> None:
        """Log LightGBM-specific parameters."""
        lgb_params = {
            "learning_rate": params.get("learning_rate"),
            "num_leaves": params.get("num_leaves"),
            "max_depth": params.get("max_depth"),
            "min_child_samples": params.get("min_child_samples"),
            "subsample": params.get("subsample"),
            "colsample_bytree": params.get("colsample_bytree"),
            "reg_alpha": params.get("reg_alpha"),
            "reg_lambda": params.get("reg_lambda")
        }
        super().log_model_params(lgb_params)
    
    def log_feature_importance(self, model: Any) -> None:
        """Log LightGBM feature importance."""
        importance = dict(zip(model.feature_name_, model.feature_importances_))
        super().log_feature_importance(importance)

class XGBoostTracker(BaseTracker):
    """Tracker for XGBoost model training and evaluation."""
    
    def __init__(self, experiment_name: str = "electricity_forecasting", run_name: Optional[str] = None):
        super().__init__(experiment_name, run_name)
    
    def log_model_params(self, params: Dict[str, Any]) -> None:
        """Log XGBoost-specific parameters."""
        xgb_params = {
            "learning_rate": params.get("learning_rate"),
            "max_depth": params.get("max_depth"),
            "min_child_weight": params.get("min_child_weight"),
            "subsample": params.get("subsample"),
            "colsample_bytree": params.get("colsample_bytree"),
            "gamma": params.get("gamma"),
            "reg_alpha": params.get("reg_alpha"),
            "reg_lambda": params.get("reg_lambda")
        }
        super().log_model_params(xgb_params)
    
    def log_feature_importance(self, model: Any) -> None:
        """Log XGBoost feature importance."""
        importance = dict(zip(model.feature_names_in_, model.feature_importances_))
        super().log_feature_importance(importance) 