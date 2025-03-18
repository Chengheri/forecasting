import sys
import os
import argparse
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

import pandas as pd
import numpy as np
import json
from typing import Dict, Any, Tuple, Optional, Union
from datetime import datetime

from backend.app.models.base_model import BaseForecastingModel
from backend.app.utils.logger import Logger
from backend.app.utils.decorators import log_execution_time, handle_pipeline_errors
from backend.app.pipelines.base_pipeline import BasePipeline
from backend.app.utils.trackers import ForecastingTracker

# Initialize logger
logger = Logger()

class ModelTrainer:
    def parse_arguments(self):
        """Parse command line arguments."""
        parser = argparse.ArgumentParser(description='Train a forecasting model')
        parser.add_argument('--config', type=str, default='config/config.json',
                            help='Path to configuration file')
        parser.add_argument('--model', type=str, default='lstm',
                            help='Type of model to train (e.g. lstm, prophet, arima)')
        return parser.parse_args()

    @handle_pipeline_errors
    @log_execution_time
    def main(self, pipeline: BasePipeline):
        """Main function to train and evaluate LSTM model."""

        train_data, test_data = self.process_data(pipeline)        
        # Train and evaluate model
        model_results = self.train_and_evaluate_model(pipeline, train_data, test_data)
        model, metrics, optimization_results, mean_forecast, confidence_intervals = model_results
        
        # Save model and analyze results
        model_artifacts = self.save_model_artifacts(pipeline, model, test_data, mean_forecast, 
                                            confidence_intervals)
        model_path, analysis_path, model_file_path, analysis_results = model_artifacts
        
        # Track results and create summary
        model_summary = self.track_and_summarize_results(pipeline, metrics, optimization_results, analysis_results,
                                    analysis_path, model_file_path, model_path, train_data, test_data)

    def load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from file."""
        logger.info(f"Loading configuration from {config_path}")

        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.error(f"Configuration file {config_path} not found")
            raise


    def initialize_tracking(self, config: Dict[str, Any], run_timestamp: str) -> ForecastingTracker:
        """Initialize MLflow tracking."""
        pass

    def process_data(self, pipeline: BasePipeline) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load, prepare and split data using pipeline."""
        data = pipeline.load_data()
        data = pipeline.prepare_data(data)
        train_data, test_data = pipeline.split_data(data)
        return train_data, test_data

    def train_and_evaluate_model(self, pipeline: BasePipeline, 
                            train_data: pd.DataFrame,
                            test_data: pd.DataFrame,
    ) -> Tuple:
        """Train model and generate predictions."""
        # Train model
        model, metrics, optimization_results = pipeline.train_model(train_data)
        
        # Generate predictions
        mean_forecast, confidence_intervals = pipeline.test_model(model, test_data)
        
        return model, metrics, optimization_results, mean_forecast, confidence_intervals

    def save_model_artifacts(self, pipeline: BasePipeline,
                            model: BaseForecastingModel,
                            test_data: pd.DataFrame,
                            mean_forecast: np.ndarray,
                            confidence_intervals: Optional[Tuple[np.ndarray, np.ndarray]],
    ) -> Tuple:
        """Save model and generate analysis artifacts."""
        # Create directories
        model_type = pipeline.config['model']['model_type']
        model_path, analysis_path, model_file_path = self.create_model_directories(model_type)
        
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

    def create_model_directories(self, model_type: str) -> Tuple[str, str, str]:
        """Create necessary directories for model artifacts."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_name = f"{model_type}_{timestamp}"
        
        model_path = os.path.join("data/models", model_name)
        analysis_path = os.path.join(model_path, "analysis")
        model_file_path = os.path.join(model_path, "model", f"{model_type}_model.joblib")
        
        os.makedirs(os.path.join(model_path, "model"), exist_ok=True)
        os.makedirs(analysis_path, exist_ok=True)
        
        return model_path, analysis_path, model_file_path

    def track_and_summarize_results(self, pipeline: BasePipeline,
                                metrics: Dict[str, Any],
                                optimization_results: Dict[str, Any],
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
            metrics=metrics,
            optimization_results=optimization_results,
            analysis_results=analysis_results,
            analysis_path=analysis_path,
            model_file_path=model_file_path
        )
        
        # Create and save model summary
        model_summary = pipeline.create_model_summary(
            train_data=train_data,
            test_data=test_data,
            metrics=metrics,
            optimization_results=optimization_results,
            analysis_results=analysis_results,
            model_path=model_path,
            analysis_path=analysis_path,
            preprocessor=pipeline.preprocessor
        )
        
        # Save model summary
        summary_file = os.path.join(model_path, "model_summary.json")
        with open(summary_file, 'w') as f:
            json.dump(model_summary, f, indent=4)
        
        return model_summary