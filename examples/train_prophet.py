import sys
import os
import argparse
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

import pandas as pd
import numpy as np
from datetime import datetime
import mlflow
import matplotlib.pyplot as plt
from backend.app.models.prophet_model import ProphetModel
from backend.app.models.neuralprophet_model import NeuralProphetModel
from backend.app.utils.trackers import ProphetTracker
from backend.app.utils.logger import Logger
from backend.app.utils.advanced_preprocessing import AdvancedPreprocessor
from backend.app.utils.analyzer import Analyzer
from backend.app.utils.decorators import log_execution_time
from backend.app.pipelines.prophet_pipeline import ProphetPipeline
import json
from prophet import Prophet
import joblib
from typing import Dict, Any, Tuple, Optional, Union

# Initialize logger
logger = Logger()

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train a forecasting model')
    parser.add_argument('--model', type=str, choices=['prophet', 'neuralprophet'], 
                        default='prophet', help='Model type to train (prophet or neuralprophet)')
    return parser.parse_args()

@log_execution_time
def main():
    """Main function to train and evaluate Prophet model."""
    # Initialize tracker to None so it's always defined
    tracker = None
    
    try:
        # Parse command line arguments
        args = parse_arguments()
        model_type = args.model
        
        # Load configuration and initialize tracking
        config = load_config(model_type)
        run_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        tracker = initialize_tracking(config, model_type, run_timestamp)
        
        # Initialize pipeline and process data
        pipeline = ProphetPipeline(config=config, tracker=tracker)
        train_data, test_data = process_data(pipeline)
        
        # Train and evaluate model
        model_results = train_and_evaluate_model(pipeline, train_data, test_data)
        model, model_params, metrics, grid_search_results, mean_forecast, confidence_intervals = model_results
        
        # Save model and analyze results
        model_artifacts = save_model_artifacts(pipeline, model, model_params, test_data, mean_forecast, 
                                            confidence_intervals, model_type)
        model_path, analysis_path, model_file_path, analysis_results = model_artifacts
        
        # Track results and create summary
        model_summary = track_and_summarize_results(pipeline, model_params, metrics, grid_search_results, analysis_results,
                                  analysis_path, model_file_path, model_path, train_data, test_data)
        
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        raise
    
    finally:
        if tracker:
            tracker.end_run()

def load_config(model_type: str) -> Dict[str, Any]:
    """Load configuration from file based on model type."""
    config_file = f'config/config_{model_type}.json'
    logger.info(f"Loading configuration from {config_file}")
    
    try:
        with open(config_file, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        logger.error(f"Configuration file {config_file} not found")
        raise

def initialize_tracking(config: Dict[str, Any], model_type: str, run_timestamp: str) -> ProphetTracker:
    """Initialize MLflow tracking."""
    experiment_name = config['mlflow']['experiment_name']
    run_name = f"{model_type}_{run_timestamp}"
    
    logger.info(f"Initializing MLflow tracker for experiment: {experiment_name}")
    tracker = ProphetTracker(
        experiment_name=experiment_name,
        run_name=run_name
    )
    return tracker

def process_data(pipeline: ProphetPipeline) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load, prepare and split data using pipeline."""
    data = pipeline.load_data()
    data = pipeline.prepare_data(data)
    train_data, test_data = pipeline.split_data(data)
    return train_data, test_data

def train_and_evaluate_model(pipeline: ProphetPipeline, 
                           train_data: pd.DataFrame,
                           test_data: pd.DataFrame,
) -> Tuple:
    """Train model and generate predictions."""
    # Train model
    model, model_params, metrics, grid_search_results = pipeline.train_model(
        train_data=train_data
    )
    
    # Generate predictions
    mean_forecast, confidence_intervals = pipeline.test_model(model, test_data)
    
    return model, model_params, metrics, grid_search_results, mean_forecast, confidence_intervals

def save_model_artifacts(pipeline: ProphetPipeline,
                        model: Union[ProphetModel, NeuralProphetModel],
                        model_params: Dict[str, Any],
                        test_data: pd.DataFrame,
                        mean_forecast: np.ndarray,
                        confidence_intervals: Optional[Tuple[np.ndarray, np.ndarray]],
                        model_type: str,
) -> Tuple:
    """Save model and generate analysis artifacts."""
    # Create directories
    model_path, analysis_path, model_file_path = create_model_directories(model_type, model_params)
    
    # Save model
    model.save(model_file_path)
    
    # Analyze results
    analysis_results = pipeline.analyze_results(
        test_data=test_data,
        mean_forecast=mean_forecast,
        confidence_intervals=confidence_intervals,
        analysis_path=analysis_path
    )
    
    return model_path, analysis_path, model_file_path, analysis_results

def create_model_directories(model_type: str, model_params: Dict[str, Any]) -> Tuple[str, str, str]:
    """Create necessary directories for model artifacts."""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_name = f"{model_type}_{timestamp}"
    
    model_path = os.path.join("data/models", model_name)
    analysis_path = os.path.join(model_path, "analysis")
    model_file_path = os.path.join(model_path, "model", f"{model_type}_model.joblib")
    
    os.makedirs(os.path.join(model_path, "model"), exist_ok=True)
    os.makedirs(analysis_path, exist_ok=True)
    
    return model_path, analysis_path, model_file_path

def track_and_summarize_results(pipeline: ProphetPipeline,
                              model_params: Dict[str, Any],
                              metrics: Dict[str, Any],
                              grid_search_results: Optional[Dict[str, Any]],
                              analysis_results: Dict[str, Any],
                              analysis_path: str,
                              model_file_path: str,
                              model_path: str,
                              train_data: pd.DataFrame,
                              test_data: pd.DataFrame,
    ) -> Dict[str, Any]:
    """Track results and create model summary."""
    # Track results
    pipeline.track_results(
        model_params=model_params,
        metrics=metrics,
        analysis_results=analysis_results,
        grid_search_results=grid_search_results,
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

if __name__ == "__main__":
    # Print usage instructions
    print("Training forecasting model")
    print("Usage: python train_prophet.py [--model {prophet,neuralprophet}]")
    print("Example: python train_prophet.py --model neuralprophet")
    print("\nStarting training process...\n")
    
    # Run the main function
    main() 