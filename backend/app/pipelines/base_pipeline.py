"""Base class for all pipeline implementations."""

from typing import Dict, Any, Callable, TypeVar, Optional, Tuple
from functools import wraps
import pandas as pd
import numpy as np
import json
from datetime import datetime

from ..utils.base import LoggedObject
from ..utils.analyzer import Analyzer
from ..utils.preprocessing import DataPreprocessor
from ..utils.trackers import ForecastingTracker
from ..utils.data_loader import DataLoader, convert_to_native_types
from ..utils.logger import Logger
from ..utils.decorators import handle_pipeline_errors
from ..models.base_model import BaseForecastingModel
logger = Logger()

class BasePipeline(LoggedObject):
    """Base class for all pipeline implementations."""
    
    def __init__(self, config: Dict[str, Any], tracker: ForecastingTracker):
        """Initialize base pipeline with logging."""
        super().__init__()
        self.config = config
        self.tracker = tracker
        self.analyzer = Analyzer(config=config)
        self.preprocessor = DataPreprocessor(config=config['preprocessing'], tracker=tracker) 

    @handle_pipeline_errors
    def load_data(self) -> pd.DataFrame:
        """Load data using DataLoader based on configuration.
        
        Returns:
            pd.DataFrame: Loaded data
        """
        data_loader = DataLoader(config=self.config)
        data = data_loader.load_csv()
        return data

    @handle_pipeline_errors
    def split_data(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Split data into training and test sets.
        
        Args:
            data: Input data to split
            
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: Training and test datasets
        """        
        # Convert data to DataFrame if it's a Series
        if isinstance(data, pd.Series):
            data = data.to_frame()
        
        # Get date column name
        date_column = str(data.index.name) if data.index.name else None
        
        # Split data using preprocessor's method
        split_result = self.preprocessor.train_test_split_timeseries(
            data,
            train_ratio=self.config['preprocessing']['train_ratio'],
            validation_ratio=self.config['preprocessing'].get('validation_ratio', 0.0),
            target_column=self.config['data']['target_column'],
            date_column=date_column,
            gap=self.config['preprocessing'].get('gap', 0)
        )
        
        # Ensure we only get train and test data
        train_data, test_data = split_result[0], split_result[1]            
        return train_data, test_data

    @handle_pipeline_errors
    def prepare_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare data using DataPreprocessor based on configuration.
        
        Args:
            data: Raw input data
            
        Returns:
            pd.DataFrame: Prepared data
        """
        prepared_data = self.preprocessor.preprocess_data(
            data, 
            target_column=self.config['data']['target_column']
        )
        return pd.DataFrame(prepared_data)

    @handle_pipeline_errors
    def train_model(self, model: BaseForecastingModel, train_data: pd.DataFrame) -> Tuple[Dict[str, Any], Dict[str, Any], Optional[Dict[str, Any]]]:
        """Train the forecasting model.
        
        Args:
            train_data: Training dataset            
        Returns:
            Tuple containing:
                - Trained model instance
                - Model parameters used
                - Training metrics
                - Grid search results (if performed)
        """                
        optimization_results = {}
            
        if self.config['model'].get('optimize_hyperparameters'):
            # Perform hyperparameter optimization
            optimization_results = model.fit_with_optuna(
                data=train_data,
                n_trials=self.config['model']['optimization']['n_trials'],
                timeout=self.config['model']['optimization'].get('timeout', 600)
            )
            metrics = optimization_results['best_metrics']
        else:
            # Train model with suggested parameters
            metrics = model.fit(data=train_data)
        
        return model, metrics, optimization_results
    
    @handle_pipeline_errors
    def test_model(self, model: BaseForecastingModel, test_data: pd.DataFrame) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, np.ndarray]]]:
        """Generate predictions for test data.
        
        Args:
            model: Trained model instance
            test_data: Test dataset to generate predictions for
            
        Returns:
            Tuple containing:
                - Mean forecast as numpy array
                - Optional tuple of confidence intervals (lower, upper) as numpy arrays
        """
        predictions, confidence_intervals = model.predict(test_data)
        
        return predictions, confidence_intervals

    @handle_pipeline_errors
    def analyze_results(self, test_data: pd.DataFrame, mean_forecast: np.ndarray, 
                      confidence_intervals: Optional[Tuple[np.ndarray, np.ndarray]], 
                      analysis_path: str) -> Dict[str, Any]:
        """Analyze model results and generate visualizations.
        
        Args:
            test_data: Test dataset
            mean_forecast: Mean forecast values
            confidence_intervals: Optional tuple of (lower, upper) confidence intervals
            analysis_path: Path to save analysis artifacts
            
        Returns:
            Dict[str, Any]: Analysis results and metrics
        """
        if len(mean_forecast) < len(test_data):
            test_data = test_data.iloc[:len(mean_forecast)]
        # Convert test data to numpy array if needed
        target_col = self.config['data']['target_column']
        actual_values = test_data[target_col].to_numpy()
        
        # Get timestamps from test data
        timestamps = test_data.index.to_numpy()
        
        # Analyze model results
        analysis_results = self.analyzer.analyze_model_results(
            actual_values=actual_values,
            predicted_values=mean_forecast,
            timestamps=timestamps,
            confidence_intervals=confidence_intervals,
            save_path=analysis_path
        )
        
        return analysis_results 

    @handle_pipeline_errors
    def track_results(self, metrics: Dict[str, Any], 
                     analysis_results: Dict[str, Any],
                     optimization_results: Optional[Dict[str, Any]],
                     model_file_path: str,
                     analysis_path: str) -> None:
        """Track model results using MLflow."""
        self.tracker.log_model_params(self.config['model'], self.config['model']['model_type'])
        self.tracker.log_training_metrics(metrics)
        self.tracker.log_test_metrics(analysis_results)
        self.tracker.log_optimization_results(optimization_results)
        
        # Log artifacts
        self.tracker.log_artifact(model_file_path, "model")
        self.tracker.log_artifact(analysis_path, "analysis")

    
    @handle_pipeline_errors
    def create_model_summary(
        self, train_data: pd.DataFrame, test_data: pd.DataFrame,
        optimization_results: Optional[Dict[str, Any]],
        metrics: Dict[str, Any],
        analysis_results: Dict[str, Any], 
        model_path: str, analysis_path: str, 
        preprocessor: Optional[DataPreprocessor] = None
    ) -> Dict[str, Any]:
        """Create a comprehensive model summary.
        
        Args:
            train_data: Training dataset
            test_data: Test dataset
            model_params: Model parameters used
            metrics: Training metrics
            analysis_results: Results from model analysis
            stationarity_results: Results from stationarity analysis
            model_path: Path where model is saved
            analysis_path: Path where analysis results are saved
            grid_search_results: Optional grid search results
            preprocessor: Optional data preprocessor instance
            
        Returns:
            Dict[str, Any]: Model summary
        """
        model_summary = {
            'initial_configuration': convert_to_native_types(self.config),
            'training_details': {
                'train_samples': len(train_data),
                'test_samples': len(test_data),
                'training_date': datetime.now().isoformat(),
                'features': self.config['data']['features'],
                'parameter_selection_method': 'optimization' if self.config['model'].get('optimize_hyperparameters') else 'default'
            },
            'best_model': {
                'optimization_results': convert_to_native_types(optimization_results),
                'performance': {
                    'training_metrics': convert_to_native_types(metrics),
                    'test_metrics': convert_to_native_types(analysis_results)    
                },
                'artifacts': {
                    'model_path': model_path,
                    'analysis_path': analysis_path,
                    'plots': [
                        'actual_vs_predicted.png',
                        'residuals_analysis.png',
                        'metrics_over_time.png',
                        'components.png'
                    ]
                }
            }
        }        
        # Add preprocessing pipeline if available
        if preprocessor:
            model_summary['preprocessing_pipeline'] = preprocessor.pipeline_steps
        
        return model_summary