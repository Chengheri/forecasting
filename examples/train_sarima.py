import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

import pandas as pd
import numpy as np
from datetime import datetime
import mlflow
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import acf, pacf
from backend.app.models.arima_model import TimeSeriesModel
from backend.app.utils.trackers import ARIMATracker
from backend.app.utils.logger import Logger
from backend.app.utils.advanced_preprocessing import AdvancedPreprocessor
from backend.app.utils.analyzer import Analyzer
from backend.app.utils.decorators import log_execution_time
from backend.app.pipelines.sarima_pipeline import SARIMAPipeline
import json
from statsmodels.tsa.statespace.sarimax import SARIMAX
import joblib
from typing import Dict, Any, Tuple, Optional, Union

# Initialize logger
logger = Logger()

@log_execution_time
def main():
    """Main function to train and evaluate ARIMA/SARIMA model."""
    try:
        # Load configuration and initialize tracking
        config = load_config()
        run_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        tracker = initialize_tracking(config, run_timestamp)
        
        # Initialize pipeline and process data
        pipeline = SARIMAPipeline(config=config, tracker=tracker)
        data = process_data(pipeline)
        
        # Train and evaluate model
        model_results = train_and_evaluate_model(pipeline, data)
        model, model_params, metrics, grid_search_results, mean_forecast, confidence_intervals = model_results
        
        # Save model and analyze results
        model_artifacts = save_model_artifacts(pipeline, model, model_params, test_data, mean_forecast, 
                                            confidence_intervals, metrics, stationarity_results)
        model_path, analysis_path, model_file_path, analysis_results = model_artifacts
        
        # Track results and create summary
        track_and_summarize_results(pipeline, model_params, metrics, analysis_results, stationarity_results,
                                  analysis_path, model_file_path, model_path, train_data, test_data,
                                  grid_search_results)
        
        # Log execution time and MLflow info
        log_execution_info(tracker, run_timestamp)
        
        return model_summary
        
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        raise
    
    finally:
        if tracker:
            tracker.end_run()

def load_config() -> Dict[str, Any]:
    """Load configuration from file."""
    with open('config/config.json', 'r') as f:
        return json.load(f)

def initialize_tracking(config: Dict[str, Any], run_timestamp: str) -> ARIMATracker:
    """Initialize MLflow tracking."""
    return ARIMATracker(
        experiment_name=config['mlflow']['experiment_name'], 
        run_name=f"run_{run_timestamp}"
    )

def process_data(pipeline: SARIMAPipeline) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """Load, prepare and split data using pipeline."""
    data = pipeline.load_data()
    data = pipeline.prepare_data(data)
    suggested_params, stationarity_results = pipeline.get_model_parameters(data)
    train_data, test_data = pipeline.split_data(data)
    return train_data, test_data, suggested_params, stationarity_results

def train_and_evaluate_model(pipeline: SARIMAPipeline, 
                           train_data: pd.DataFrame,
                           test_data: pd.DataFrame,
                           suggested_params: Dict[str, Any],
                           stationarity_results: Dict[str, Any]) -> Tuple:
    """Train model and generate predictions."""
    # Train model
    model, model_params, metrics, grid_search_results = pipeline.train_model(
        train_data=train_data,
        test_data=test_data,
        suggested_params=suggested_params,
        stationarity_results=stationarity_results,
        preprocessor=pipeline.preprocessor
    )
    
    # Generate predictions
    mean_forecast, confidence_intervals = pipeline.test_model(model, test_data)
    
    return model, model_params, metrics, grid_search_results, mean_forecast, confidence_intervals

def save_model_artifacts(pipeline: SARIMAPipeline,
                        model: Union[TimeSeriesModel, SARIMAX],
                        model_params: Dict[str, Any],
                        test_data: pd.DataFrame,
                        mean_forecast: np.ndarray,
                        confidence_intervals: Optional[Tuple[np.ndarray, np.ndarray]],
                        metrics: Dict[str, Any],
                        stationarity_results: Dict[str, Any]) -> Tuple:
    """Save model and generate analysis artifacts."""
    # Create directories
    model_path, analysis_path, model_file_path = create_model_directories(model_params)
    
    # Save model
    save_model(model, model_file_path, model_params)
    
    # Analyze results
    analysis_results = pipeline.analyze_results(
        test_data=test_data,
        mean_forecast=mean_forecast,
        confidence_intervals=confidence_intervals,
        analysis_path=analysis_path
    )
    
    return model_path, analysis_path, model_file_path, analysis_results


def create_model_directories(model_params: Dict[str, Any]) -> Tuple[str, str, str]:
    """Create necessary directories for model artifacts.
    
    Args:
        model_params: Model parameters containing model type and order
        
    Returns:
        Tuple containing model path, analysis path, and model file path
    """
    # Create model-specific directory name
    if model_params.get('model_type') == 'sarima':
        model_name = f"sarima_p{model_params['p']}_d{model_params['d']}_q{model_params['q']}_P{model_params['P']}_D{model_params['D']}_Q{model_params['Q']}_s{model_params['s']}"
    else:
        model_name = f"arima_p{model_params['p']}_d{model_params['d']}_q{model_params['q']}"
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_path = os.path.join("data/models", model_name, timestamp)
    analysis_path = os.path.join(model_path, "analysis")
    model_file_path = os.path.join(model_path, "model", "sarima_model.joblib")
    
    os.makedirs(os.path.join(model_path, "model"), exist_ok=True)
    os.makedirs(analysis_path, exist_ok=True)
    
    return model_path, analysis_path, model_file_path

def save_model(model: Union[TimeSeriesModel, SARIMAX], model_file_path: str, model_params: Dict[str, Any]) -> None:
    """Save model to file.
    
    Args:
        model: Model instance to save
        model_file_path: Path to save the model
        model_params: Model parameters
    """
    if isinstance(model, TimeSeriesModel):
        model.save_model(model_file_path)
    else:
        model_data = {
            'fitted_model': model,
            'config': model_params,
            'model_type': 'sarima'
        }
        joblib.dump(model_data, model_file_path)
    logger.info("Model saved successfully")

def track_and_summarize_results(pipeline: SARIMAPipeline,
                              model_params: Dict[str, Any],
                              metrics: Dict[str, Any],
                              analysis_results: Dict[str, Any],
                              stationarity_results: Dict[str, Any],
                              analysis_path: str,
                              model_file_path: str,
                              model_path: str,
                              train_data: pd.DataFrame,
                              test_data: pd.DataFrame,
                              grid_search_results: Optional[Dict] = None) -> Dict[str, Any]:
    """Track results and create model summary."""
    # Track results
    pipeline.track_results(
        model_params=model_params,
        metrics=metrics,
        analysis_results=analysis_results,
        stationarity_results=stationarity_results,
        analysis_path=analysis_path,
        model_file_path=model_file_path
    )
    
    # Create and save model summary
    model_summary = pipeline.create_model_summary(
        train_data=train_data,
        test_data=test_data,
        model_params=model_params,
        metrics=metrics,
        analysis_results=analysis_results,
        stationarity_results=stationarity_results,
        model_path=model_path,
        analysis_path=analysis_path,
        grid_search_results=grid_search_results,
        preprocessor=pipeline.preprocessor
    )
    
    # Save model summary
    summary_file = os.path.join(model_path, "model_summary.json")
    with open(summary_file, 'w') as f:
        json.dump(model_summary, f, indent=4)
    
    return model_summary

def log_execution_info(tracker: ARIMATracker, run_timestamp: str) -> None:
    """Log execution time and MLflow information."""
    end_time = datetime.now()
    execution_time = (datetime.strptime(run_timestamp, '%Y%m%d_%H%M%S') - end_time).total_seconds()
    logger.info(f"Total training time: {execution_time:.2f} seconds")
    log_mlflow_run_info(tracker, run_timestamp)

def log_mlflow_run_info(tracker: ARIMATracker, run_timestamp: str) -> None:
    """Log MLflow run information.
    
    Args:
        tracker: MLflow tracker instance
        run_timestamp: Timestamp of the run
    """
    if mlflow.active_run():
        run_id = mlflow.active_run().info.run_id
        experiment_id = mlflow.active_run().info.experiment_id
        print(f"🏃 View run {tracker.current_run.info.run_name} at: http://localhost:5001/#/experiments/{experiment_id}/runs/{run_id}")
        print(f"🧪 View experiment at: http://localhost:5001/#/experiments/{experiment_id}")

if __name__ == "__main__":
    main() 