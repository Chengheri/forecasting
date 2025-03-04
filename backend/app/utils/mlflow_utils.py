import mlflow
import os
from datetime import datetime
from typing import Dict, Any, Optional, List, Union
from ..utils.logger import Logger
import pandas as pd
import numpy as np
from mlflow.tracking import MlflowClient

logger = Logger()

class BaseModelTracker:
    """Base class for model tracking with MLflow."""
    
    def __init__(self, model_name: str, experiment_name: str = "electricity_forecasting"):
        """Initialize model tracker."""
        self.model_name = model_name
        self.tracker = MLflowTracker(experiment_name)
        self.current_run = None
    
    def start_tracking(self, run_name: Optional[str] = None) -> None:
        """Start tracking a new run."""
        if run_name is None:
            run_name = f"{self.model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.current_run = self.tracker.start_run(run_name)
    
    def end_tracking(self) -> None:
        """End current tracking run."""
        if self.current_run:
            self.tracker.end_run()
            self.current_run = None
    
    def log_model_params(self, params: Dict[str, Any]) -> None:
        """Log model parameters."""
        self.tracker.log_parameters(params)
    
    def log_training_metrics(self, metrics: Dict[str, float]) -> None:
        """Log training metrics."""
        self.tracker.log_metrics(metrics)
    
    def log_evaluation_metrics(self, metrics: Dict[str, float]) -> None:
        """Log evaluation metrics."""
        self.tracker.log_metrics({f"eval_{k}": v for k, v in metrics.items()})
    
    def log_model(self, model: Any) -> None:
        """Log model artifact."""
        self.tracker.log_model(model, f"{self.model_name}_model")
    
    def log_predictions(self, predictions: Union[np.ndarray, pd.Series], 
                       actual_values: Union[np.ndarray, pd.Series],
                       file_name: str = "predictions.csv") -> None:
        """Log predictions and actual values."""
        if isinstance(predictions, np.ndarray):
            predictions = pd.Series(predictions)
        if isinstance(actual_values, np.ndarray):
            actual_values = pd.Series(actual_values)
            
        df = pd.DataFrame({
            'predictions': predictions,
            'actual': actual_values,
            'error': predictions - actual_values
        })
        
        # Save to temporary file
        temp_path = f"temp_{file_name}"
        df.to_csv(temp_path, index=False)
        
        # Log as artifact
        self.tracker.log_artifact(temp_path, file_name)
        
        # Clean up
        os.remove(temp_path)
    
    def log_feature_importance(self, feature_importance: Dict[str, float],
                             file_name: str = "feature_importance.csv") -> None:
        """Log feature importance scores."""
        df = pd.DataFrame({
            'feature': list(feature_importance.keys()),
            'importance': list(feature_importance.values())
        }).sort_values('importance', ascending=False)
        
        # Save to temporary file
        temp_path = f"temp_{file_name}"
        df.to_csv(temp_path, index=False)
        
        # Log as artifact
        self.tracker.log_artifact(temp_path, file_name)
        
        # Clean up
        os.remove(temp_path)

class MLflowTracker:
    def __init__(self, experiment_name: str = "electricity_forecasting"):
        """Initialize MLflow tracker with experiment name."""
        logger.info(f"Initializing MLflow tracker for experiment: {experiment_name}")
        self.experiment_name = experiment_name
        self.client = MlflowClient()
        self.current_run = None
        
        # Set up experiment
        try:
            self.experiment_id = mlflow.create_experiment(experiment_name)
        except Exception:
            self.experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id
        
        mlflow.set_experiment(experiment_name)
    
    def start_run(self, run_name: Optional[str] = None, nested: bool = True) -> mlflow.ActiveRun:
        """Start a new MLflow run."""
        if run_name is None:
            run_name = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        logger.info(f"Starting MLflow run: {run_name}")
        self.current_run = mlflow.start_run(run_name=run_name, nested=nested)
        return self.current_run
    
    def log_parameters(self, params: Dict[str, Any]) -> None:
        """Log parameters to MLflow."""
        logger.info(f"Logging parameters: {params}")
        if not mlflow.active_run():
            self.start_run()
        mlflow.log_params(params)
    
    def log_metrics(self, metrics: Dict[str, float]) -> None:
        """Log metrics to MLflow."""
        logger.info(f"Logging metrics: {metrics}")
        if not mlflow.active_run():
            self.start_run()
        mlflow.log_metrics(metrics)
    
    def log_model(self, model: Any, model_name: str) -> None:
        """Log model to MLflow."""
        logger.info(f"Logging model: {model_name}")
        mlflow.sklearn.log_model(model, model_name)
    
    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None) -> None:
        """Log artifact to MLflow."""
        logger.info(f"Logging artifact from: {local_path}")
        if not mlflow.active_run():
            self.start_run()
        mlflow.log_artifact(local_path, artifact_path)
    
    def end_run(self) -> None:
        """End the current MLflow run."""
        logger.info("Ending MLflow run")
        if mlflow.active_run():
            mlflow.end_run()
    
    def search_runs(self, filter_string: str = "") -> list:
        """Search for runs with given filter string."""
        logger.info(f"Searching runs with filter: {filter_string}")
        return mlflow.search_runs(filter_string=filter_string)
    
    def get_best_run(self, metric_name: str, ascending: bool = True) -> Dict[str, Any]:
        """Get the best run based on a metric."""
        logger.info(f"Getting best run for metric: {metric_name}")
        runs = mlflow.search_runs()
        if runs.empty:
            logger.warning("No runs found")
            return {}
        
        best_run = runs.sort_values(by=[f"metrics.{metric_name}"], ascending=ascending).iloc[0]
        return {
            "run_id": best_run.run_id,
            "experiment_id": best_run.experiment_id,
            "metrics": {k: v for k, v in best_run.items() if k.startswith("metrics.")},
            "params": {k: v for k, v in best_run.items() if k.startswith("params.")}
        }

    def __enter__(self):
        """Context manager entry."""
        if not mlflow.active_run():
            self.start_run()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.end_run()

class BaseTracker:
    def __init__(self, experiment_name):
        self.tracker = MLflowTracker(experiment_name)
    
    def start_tracking(self, run_name=None):
        """Start tracking with MLflow."""
        return self.tracker.start_run(run_name=run_name, nested=True)
    
    def end_tracking(self):
        """End tracking with MLflow."""
        self.tracker.end_run()
    
    def log_model_params(self, params):
        """Log model parameters."""
        self.tracker.log_parameters(params)
    
    def log_metrics(self, metrics):
        """Log metrics."""
        self.tracker.log_metrics(metrics)
    
    def log_artifact(self, local_path):
        """Log an artifact."""
        self.tracker.log_artifact(local_path)
    
    def __enter__(self):
        """Context manager entry."""
        return self.tracker.__enter__()
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        return self.tracker.__exit__(exc_type, exc_val, exc_tb)

class PreprocessorTracker(BaseTracker):
    def __init__(self, experiment_name="preprocessing"):
        super().__init__(experiment_name)
    
    def log_preprocessing_config(self, config):
        """Log preprocessing configuration."""
        with self.tracker:
            self.log_model_params(config)
    
    def log_feature_stats(self, data):
        """Log feature statistics."""
        with self.tracker:
            stats = {}
            for column in data.columns:
                desc = data[column].describe()
                for stat in desc.index:
                    stats[f"{column}_{stat}"] = desc[stat]
                if data[column].dtype != 'datetime64[ns]':
                    stats[f"{column}_skew"] = data[column].skew()
                    stats[f"{column}_kurtosis"] = data[column].kurtosis()
            self.log_model_params(stats)

# ARIMATracker has been moved to model_trackers.py for better organization 