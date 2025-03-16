"""
Model and preprocessing tracking utilities for MLflow.
"""

from typing import Dict, Any, List, Optional, Union
import numpy as np
import pandas as pd
from .mlflow_utils import MLflowTracker
from ..utils.logger import Logger
import mlflow

logger = Logger()

class PreprocessorTracker(MLflowTracker):
    """Tracker for data preprocessing steps."""
    
    def __init__(self, experiment_name: str = "electricity_forecasting", run_name: Optional[str] = None, run_id: Optional[str] = None):
        super().__init__(experiment_name)
        self.run_id = run_id
        if run_name:
            self.start_run(run_name=run_name)
    
    def _prefix_dict_keys(self, d: Dict[str, Any], prefix: str) -> Dict[str, Any]:
        """Add prefix to all dictionary keys."""
        return {f"{prefix}.{k}": v for k, v in d.items()}
    
    def log_preprocessing_config(self, config: Dict[str, Any]) -> None:
        """Log preprocessing configuration."""
        logger.info("Logging preprocessing configuration")
        prefixed_config = self._prefix_dict_keys(config, "preprocessing.config")
        self.log_params_safely(prefixed_config)
    
    def log_missing_values_stats(self, df: pd.DataFrame) -> None:
        """Log statistics about missing values."""
        missing_stats = df.isnull().sum().to_dict()
        missing_percentages = (df.isnull().sum() / len(df) * 100).to_dict()
        
        stats = {
            "total_rows": len(df),
            "missing_values": missing_stats,
            "missing_percentages": missing_percentages
        }
        
        prefixed_stats = self._prefix_dict_keys(stats, "preprocessing.missing_values")
        self.log_params_safely(prefixed_stats)
    
    def log_anomaly_detection_stats(self, method: str, n_anomalies: int, anomaly_percentage: float, params: Dict[str, Any]) -> None:
        """Log anomaly detection statistics."""
        stats = {
            "method": method,
            "n_detected": n_anomalies,
            "percentage": anomaly_percentage
        }
        
        # Add method-specific parameters
        for param_name, param_value in params.items():
            if param_value is not None:
                stats[f"params.{param_name}"] = param_value
        
        prefixed_stats = self._prefix_dict_keys(stats, "preprocessing.anomalies")
        self.log_params_safely(prefixed_stats)
    
    def log_feature_stats(self, df: pd.DataFrame) -> None:
        """Log statistics about features."""
        stats = {}
        
        for column in df.columns:
            if pd.api.types.is_numeric_dtype(df[column]):
                column_stats = {
                    f"{column}.mean": df[column].mean(),
                    f"{column}.std": df[column].std(),
                    f"{column}.min": df[column].min(),
                    f"{column}.max": df[column].max(),
                    f"{column}.skew": df[column].skew(),
                    f"{column}.kurtosis": df[column].kurtosis()
                }
                stats.update(column_stats)
        
        prefixed_stats = self._prefix_dict_keys(stats, "preprocessing.features")
        self.log_params_safely(prefixed_stats)
    
    def log_feature_engineering(self, feature_type: str, added_features: List[str]) -> None:
        """Log feature engineering steps."""
        feature_info = {
            f"{feature_type}_features": added_features,
        }
        
        prefixed_info = self._prefix_dict_keys(feature_info, "preprocessing.features")
        self.log_params_safely(prefixed_info)
    
    def log_preprocessing_pipeline(self, pipeline_steps: List[Dict[str, Any]]) -> None:
        """Log preprocessing pipeline steps."""
        pipeline_info = {
            "n_steps": len(pipeline_steps),
            "steps": pipeline_steps
        }
        
        prefixed_info = self._prefix_dict_keys(pipeline_info, "preprocessing.pipeline")
        self.log_params_safely(prefixed_info)

class ForecastingTracker(PreprocessorTracker):
    """Base tracker for forecasting models, combining preprocessing and model tracking."""
    
    def __init__(self, experiment_name: str = "electricity_forecasting", run_name: Optional[str] = None):
        try:
            super().__init__(experiment_name=experiment_name, run_name=run_name)
        except Exception as e:
            if "Cannot set a deleted experiment" in str(e):
                logger.warning(f"Experiment '{experiment_name}' was deleted. Creating a new one.")
                # Delete the experiment completely and create a new one
                import mlflow
                try:
                    mlflow.delete_experiment(experiment_name)
                except:
                    pass
                # Create new experiment
                mlflow.create_experiment(experiment_name)
                super().__init__(experiment_name=experiment_name, run_name=run_name)
            else:
                raise
    
    
class LSTMTracker(ForecastingTracker):
    """Tracker for LSTM model training and evaluation."""
    
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
        prefixed_params = self._prefix_dict_keys(lstm_params, "model.lstm")
        self.log_params_safely(prefixed_params)

class ProphetTracker(ForecastingTracker):
    """Tracker for Prophet model training and evaluation."""
    
    def log_model_params(self, params: Dict[str, Any]) -> None:
        """Log Prophet-specific parameters."""
        prophet_params = {
            "changepoint_prior_scale": params.get("changepoint_prior_scale", 0.05),
            "seasonality_prior_scale": params.get("seasonality_prior_scale", 10.0),
            "holidays_prior_scale": params.get("holidays_prior_scale", 10.0),
            "seasonality_mode": params.get("seasonality_mode", "additive"),
            "changepoint_range": params.get("changepoint_range", 0.8),
            "growth": params.get("growth", "linear"),
            "n_changepoints": params.get("n_changepoints", 25),
            "yearly_seasonality": params.get("yearly_seasonality", "auto"),
            "weekly_seasonality": params.get("weekly_seasonality", "auto"),
            "daily_seasonality": params.get("daily_seasonality", "auto")
        }
        prefixed_params = self._prefix_dict_keys(prophet_params, "model.prophet")
        self.log_params_safely(prefixed_params)
    
    def log_training_metrics(self, metrics: Dict[str, float]) -> None:
        """Log Prophet training metrics."""
        training_metrics = {
            "mse": metrics.get("mse"),
            "rmse": metrics.get("rmse"),
            "mae": metrics.get("mae"),
            "mape": metrics.get("mape"),
            "r2": metrics.get("r2")
        }
        # Filter out None values
        training_metrics = {k: v for k, v in training_metrics.items() if v is not None}
        prefixed_metrics = self._prefix_dict_keys(training_metrics, "metrics.training")
        self.log_metrics_safely(prefixed_metrics)
    
    def log_optimization_results(self, results: Dict[str, Any], method: str = "optuna") -> None:
        """Log hyperparameter optimization results."""
        if method == "optuna":
            optimization_info = {
                "best_value": results.get("best_value"),
                "n_trials": results.get("n_trials"),
                "timeout": results.get("timeout")
            }
            # Log best parameters separately
            if "best_params" in results:
                for param_name, param_value in results["best_params"].items():
                    optimization_info[f"best_params.{param_name}"] = param_value
        else:  # grid search
            optimization_info = {
                "total_combinations": results.get("total_combinations_tested"),
                "best_rmse": results.get("best_metrics", {}).get("rmse") if results.get("best_metrics") else None
            }
            # Log best parameters separately
            if "best_params" in results:
                for param_name, param_value in results["best_params"].items():
                    optimization_info[f"best_params.{param_name}"] = param_value
        
        prefixed_info = self._prefix_dict_keys(optimization_info, f"optimization.{method}")
        self.log_params_safely(prefixed_info)
    
    def log_forecast_metrics(self, metrics: Dict[str, float]) -> None:
        """Log forecast evaluation metrics."""
        forecast_metrics = {
            "rmse": metrics.get("rmse"),
            "mae": metrics.get("mae"),
            "mape": metrics.get("mape"),
            "r2": metrics.get("r2")
        }
        prefixed_metrics = self._prefix_dict_keys(forecast_metrics, "metrics.forecast")
        self.log_metrics_safely(prefixed_metrics)

class ARIMATracker(ForecastingTracker):
    """Tracker for ARIMA model training and evaluation."""
    
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
        prefixed_params = self._prefix_dict_keys(arima_params, "model.arima")
        self.log_params_safely(prefixed_params)
    
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
        prefixed_metrics = self._prefix_dict_keys(training_metrics, "metrics.training")
        self.log_metrics_safely(prefixed_metrics)
    
    def log_model_diagnostics(self, diagnostics: Dict[str, Any]) -> None:
        """Log model diagnostic information."""
        diagnostic_metrics = {
            "residual_autocorrelation": diagnostics.get("residual_autocorrelation"),
            "heteroskedasticity_test": diagnostics.get("heteroskedasticity_test"),
            "normality_test": diagnostics.get("normality_test"),
            "seasonal_strength": diagnostics.get("seasonal_strength")
        }
        prefixed_metrics = self._prefix_dict_keys(diagnostic_metrics, "metrics.diagnostics")
        self.log_metrics_safely(prefixed_metrics)
    
    def log_forecast_metrics(self, metrics: Dict[str, float]) -> None:
        """Log forecast evaluation metrics."""
        forecast_metrics = {
            "rmse": metrics.get("rmse"),
            "mae": metrics.get("mae"),
            "mape": metrics.get("mape"),
            "r2": metrics.get("r2"),
            "coverage": metrics.get("coverage")  # For confidence intervals
        }
        prefixed_metrics = self._prefix_dict_keys(forecast_metrics, "metrics.forecast")
        self.log_metrics_safely(prefixed_metrics)

class LightGBMTracker(ForecastingTracker):
    """Tracker for LightGBM model training and evaluation."""
    
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
        prefixed_params = self._prefix_dict_keys(lgb_params, "model.lightgbm")
        self.log_params_safely(prefixed_params)
    
    def log_feature_importance(self, model: Any) -> None:
        """Log LightGBM feature importance."""
        importance = dict(zip(model.feature_name_, model.feature_importances_))
        prefixed_importance = self._prefix_dict_keys(importance, "model.lightgbm.feature_importance")
        self.log_metrics_safely(prefixed_importance)

class XGBoostTracker(ForecastingTracker):
    """Tracker for XGBoost model training and evaluation."""
    
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
        prefixed_params = self._prefix_dict_keys(xgb_params, "model.xgboost")
        self.log_params_safely(prefixed_params)
    
    def log_feature_importance(self, model: Any) -> None:
        """Log XGBoost feature importance."""
        importance = dict(zip(model.feature_names_in_, model.feature_importances_))
        prefixed_importance = self._prefix_dict_keys(importance, "model.xgboost.feature_importance")
        self.log_metrics_safely(prefixed_importance) 