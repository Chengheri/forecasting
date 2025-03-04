from typing import Dict, Any, Optional, Union
import numpy as np
import pandas as pd
from .mlflow_utils import BaseModelTracker

class LSTMTracker(BaseModelTracker):
    """Tracker for LSTM model training and evaluation."""
    
    def __init__(self, experiment_name: str = "electricity_forecasting"):
        super().__init__("lstm", experiment_name)
    
    def log_training_metrics(self, metrics: Dict[str, float]) -> None:
        """Log LSTM-specific training metrics."""
        super().log_training_metrics(metrics)
    
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

class ProphetTracker(BaseModelTracker):
    """Tracker for Prophet model training and evaluation."""
    
    def __init__(self, experiment_name: str = "electricity_forecasting"):
        super().__init__("prophet", experiment_name)
    
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

class ARIMATracker(BaseModelTracker):
    """Tracker for ARIMA model training and evaluation."""
    
    def __init__(self, experiment_name: str = "electricity_forecasting"):
        super().__init__("arima", experiment_name)
    
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
        super().log_training_metrics(training_metrics)
    
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

class LightGBMTracker(BaseModelTracker):
    """Tracker for LightGBM model training and evaluation."""
    
    def __init__(self, experiment_name: str = "electricity_forecasting"):
        super().__init__("lightgbm", experiment_name)
    
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

class XGBoostTracker(BaseModelTracker):
    """Tracker for XGBoost model training and evaluation."""
    
    def __init__(self, experiment_name: str = "electricity_forecasting"):
        super().__init__("xgboost", experiment_name)
    
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