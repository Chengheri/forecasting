import mlflow
import os
from datetime import datetime
from typing import Dict, Any, Optional
from ..utils.logger import Logger

logger = Logger()

class MLflowTracker:
    def __init__(self, experiment_name: str = "electricity_forecasting"):
        """Initialize MLflow tracker with experiment name."""
        logger.info(f"Initializing MLflow tracker for experiment: {experiment_name}")
        self.experiment_name = experiment_name
        mlflow.set_experiment(experiment_name)
        
    def start_run(self, run_name: Optional[str] = None) -> mlflow.ActiveRun:
        """Start a new MLflow run."""
        if run_name is None:
            run_name = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        logger.info(f"Starting MLflow run: {run_name}")
        return mlflow.start_run(run_name=run_name)
    
    def log_parameters(self, params: Dict[str, Any]) -> None:
        """Log parameters to MLflow."""
        logger.info(f"Logging parameters: {params}")
        mlflow.log_params(params)
    
    def log_metrics(self, metrics: Dict[str, float]) -> None:
        """Log metrics to MLflow."""
        logger.info(f"Logging metrics: {metrics}")
        mlflow.log_metrics(metrics)
    
    def log_model(self, model: Any, model_name: str) -> None:
        """Log model to MLflow."""
        logger.info(f"Logging model: {model_name}")
        mlflow.sklearn.log_model(model, model_name)
    
    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None) -> None:
        """Log artifact to MLflow."""
        logger.info(f"Logging artifact from: {local_path}")
        mlflow.log_artifact(local_path, artifact_path)
    
    def end_run(self) -> None:
        """End the current MLflow run."""
        logger.info("Ending MLflow run")
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