import mlflow
import os
from datetime import datetime
from typing import Dict, Any, Optional, List, Union
from ..utils.logger import Logger
import pandas as pd
import numpy as np
from mlflow.tracking import MlflowClient

logger = Logger()

class MLflowTracker:
    """Base MLflow tracking functionality."""
    
    def __init__(self, experiment_name: str = "electricity_forecasting"):
        """Initialize MLflow tracker with experiment name."""
        logger.info(f"Initializing MLflow tracker for experiment: {experiment_name}")
        self.experiment_name = experiment_name
        self.client = MlflowClient()
        self.current_run = None
        
        # Set up experiment
        try:
            # Try to get existing experiment
            experiment = mlflow.get_experiment_by_name(experiment_name)
            if experiment is None:
                # Create new experiment if it doesn't exist
                logger.info(f"Creating new experiment: {experiment_name}")
                self.experiment_id = mlflow.create_experiment(experiment_name)
            elif experiment.lifecycle_stage == "deleted":
                # If experiment is deleted, delete it completely and create a new one
                logger.warning(f"Found deleted experiment '{experiment_name}'. Creating a new one.")
                mlflow.delete_experiment(experiment.experiment_id)
                self.experiment_id = mlflow.create_experiment(experiment_name)
            else:
                self.experiment_id = experiment.experiment_id
                logger.info(f"Using existing experiment: {experiment_name}")
            
            # Set the experiment
            mlflow.set_experiment(experiment_name)
            
        except Exception as e:
            logger.error(f"Error setting up experiment: {str(e)}")
            raise
    
    def format_mlflow_param(self, value: Any) -> str:
        """Format a parameter value to be MLflow-compatible."""
        if isinstance(value, (list, dict, tuple)):
            return str(value)
        elif isinstance(value, (bool, int, float)):
            return str(value)
        elif isinstance(value, str):
            return value
        elif value is None:
            return "None"
        else:
            return str(value)

    def log_params_safely(self, params: Dict[str, Any]) -> None:
        """Log parameters with proper error handling."""
        try:
            # Format all parameters first
            formatted_params = {
                key: self.format_mlflow_param(value)
                for key, value in params.items()
            }
            mlflow.log_params(formatted_params)
        except Exception as e:
            logger.warning(f"Failed to log parameters: {str(e)}")

    def log_metrics_safely(self, metrics: Dict[str, float]) -> None:
        """Log metrics with proper error handling."""
        try:
            # Ensure all metrics are float values
            formatted_metrics = {
                key: float(value)
                for key, value in metrics.items()
                if value is not None
            }
            mlflow.log_metrics(formatted_metrics)
        except Exception as e:
            logger.warning(f"Failed to log metrics: {str(e)}")
    
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
        self.log_params_safely(params)
    
    def log_metrics(self, metrics: Dict[str, float]) -> None:
        """Log metrics to MLflow."""
        logger.info(f"Logging metrics: {metrics}")
        if not mlflow.active_run():
            self.start_run()
        self.log_metrics_safely(metrics)
    
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
    
    def search_runs(self, filter_string: str = "") -> pd.DataFrame:
        """Search for runs with given filter string."""
        logger.info(f"Searching runs with filter: {filter_string}")
        try:
            runs_df = mlflow.search_runs(experiment_ids=[self.experiment_id], filter_string=filter_string)
            return pd.DataFrame(runs_df)
        except Exception as e:
            logger.error(f"Error searching runs: {str(e)}")
            return pd.DataFrame()
    
    def get_best_run(self, metric_name: str, ascending: bool = True) -> Dict[str, Any]:
        """Get the best run based on a metric."""
        logger.info(f"Getting best run for metric: {metric_name}")
        try:
            runs_df = pd.DataFrame(mlflow.search_runs(experiment_ids=[self.experiment_id]))
            if runs_df.empty:
                logger.warning("No runs found")
                return {}
            
            best_run = runs_df.sort_values(by=[f"metrics.{metric_name}"], ascending=ascending).iloc[0]
            return {
                "run_id": best_run.run_id,
                "experiment_id": best_run.experiment_id,
                "metrics": {k: v for k, v in best_run.items() if k.startswith("metrics.")},
                "params": {k: v for k, v in best_run.items() if k.startswith("params.")}
            }
        except Exception as e:
            logger.error(f"Error getting best run: {str(e)}")
            return {}
    
    def end_run(self) -> None:
        """End the current MLflow run."""
        logger.info("Ending MLflow run")
        if mlflow.active_run():
            mlflow.end_run()
    
    def __enter__(self):
        """Context manager entry."""
        if not mlflow.active_run():
            self.start_run()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.end_run()

# ARIMATracker has been moved to model_trackers.py for better organization 