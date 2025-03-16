import os
from datetime import datetime
from typing import Dict, Any, Tuple, Optional, Union, Callable, TypeVar
from functools import wraps

import numpy as np
import pandas as pd
import mlflow
import joblib
from prophet import Prophet

from backend.app.models.prophet_model import ProphetModel
from backend.app.utils.trackers import ProphetTracker
from backend.app.utils.preprocessing import DataPreprocessor
from backend.app.utils.analyzer import Analyzer
from backend.app.utils.data_loader import DataLoader, convert_to_native_types
from .base_pipeline import BasePipeline

T = TypeVar('T')

def handle_pipeline_errors(func: Callable[..., T]) -> Callable[..., T]:
    """Decorator to handle common error patterns in pipeline methods."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            # Get the instance (self) from args
            if args and isinstance(args[0], BasePipeline):
                args[0]._log_error(f"Failed in {func.__name__}: {str(e)}")
            raise
    return wrapper

class ProphetPipeline(BasePipeline):
    """Pipeline for training and evaluating Prophet models."""
    
    def __init__(self, config: Dict[str, Any], tracker: ProphetTracker):
        """Initialize the Prophet pipeline.
        
        Args:
            config: Configuration dictionary
            tracker: MLflow tracker instance
        """
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
        self._log_info("Initializing DataLoader")
        data_loader = DataLoader(config=self.config)
        
        data = data_loader.load_csv()
        return data
    
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
    def remove_non_stationarity(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Remove non-stationarity from time series data based on configuration.
        
        Args:
            data: Input time series data
            
        Returns:
            Tuple containing:
            - Processed time series data as pd.DataFrame
            - Dictionary with stationarity analysis results
        """
        self._log_info("Checking stationarity of time series data")
        
        try:
            # Create a copy of the input DataFrame
            processed_data = data.copy()
            
            # Extract target series
            target_col = self.config['data']['target_column']
            ts = pd.Series(data[target_col])

            # For Prophet, we don't need to remove non-stationarity as it handles it internally
            # But we still analyze it for reporting purposes
            self._log_info("Starting stationarity analysis...")
            results = self.analyzer.check_stationarity(ts)
            self._log_info("Stationarity analysis completed")
                
            return processed_data, results
            
        except Exception as e:
            self._log_error(f"Error in stationarity analysis: {str(e)}")
            raise
    
    def split_data(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Split data into training and test sets.
        
        Args:
            data: Input data to split
            
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: Training and test datasets
        """
        self._log_info(f"Splitting time series data with train_ratio={self.config['preprocessing']['train_ratio']}")
        
        try:
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
            
            # Log data split information
            self._log_info(f"Split data into {len(train_data)} training samples and {len(test_data)} test samples")
            self._log_info(f"Training data date range: {train_data.index[0]} to {train_data.index[-1]}")
            self._log_info(f"Test data date range: {test_data.index[0]} to {test_data.index[-1]}")
            
            return train_data, test_data
            
        except Exception as e:
            self._log_error(f"Failed to split data: {str(e)}")
            raise
    
    @handle_pipeline_errors
    def train_model(self, train_data: pd.DataFrame) -> Tuple[ProphetModel, Dict[str, Any], Dict[str, Any], Optional[Dict[str, Any]]]:
        """Train the Prophet model.
        
        Args:
            train_data: Training dataset            
        Returns:
            Tuple containing:
                - Trained model instance
                - Model parameters used
                - Training metrics
                - Grid search results (if performed)
        """                
        # Initialize model
        model = ProphetModel(
            config=self.config,
            tracker=self.tracker
        )
        model_params = model.config['model']
        train_data = model.prepare_data(train_data)
        
        if self.config['model'].get('use_grid_search'):
            # Perform grid search
            grid_search_results = model.grid_search(
                data=train_data,
                param_grid=self.config['grid_search']['param_grid'],
                max_iterations=self.config['grid_search'].get('max_iterations', 50),
                early_stopping=self.config['grid_search'].get('early_stopping', True),
                cv=self.config['grid_search'].get('cv', 3)
            )
            model_params.update(grid_search_results['best_params'])
            metrics = grid_search_results['best_metrics']
        elif self.config['model'].get('optimize_hyperparameters'):
            # Perform hyperparameter optimization
            hyperparameter_optimization_results = model.fit_with_optuna(
                data=train_data,
                n_trials=self.config['model']['optimization']['n_trials'],
                timeout=self.config['model']['optimization'].get('timeout', 600)
            )
            model_params.update(hyperparameter_optimization_results['best_params'])
            metrics = hyperparameter_optimization_results['best_metrics']
            grid_search_results = None
        else:
            # Train model with suggested parameters
            self._log_info("Starting model training with suggested parameters...")
            metrics = model.fit(data=train_data)
            grid_search_results = None
        
        return model, model_params, metrics, grid_search_results
    
    @handle_pipeline_errors
    def test_model(self, model: ProphetModel, test_data: pd.DataFrame) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, np.ndarray]]]:
        """Generate predictions for test data.
        
        Args:
            model: Trained model instance
            test_data: Test dataset to generate predictions for
            
        Returns:
            Tuple containing:
                - Mean forecast as numpy array
                - Optional tuple of confidence intervals (lower, upper) as numpy arrays
        """
        self._log_info(f"Generating predictions for test data with {len(test_data)} steps")
        
        predictions, confidence_intervals = model.predict(len(test_data))
        
        return predictions, confidence_intervals
    
    @handle_pipeline_errors
    def track_results(self, model_params: Dict[str, Any], metrics: Dict[str, Any], 
                     analysis_results: Dict[str, Any], grid_search_results: Optional[Dict[str, Any]],
                     analysis_path: str, model_file_path: str) -> None:
        """Track model results using MLflow."""
        try:
            # Log model parameters
            model_params_with_prefix = {
                'model.prophet.changepoint_prior_scale': model_params.get('changepoint_prior_scale', 0.05),
                'model.prophet.seasonality_prior_scale': model_params.get('seasonality_prior_scale', 10.0),
                'model.prophet.holidays_prior_scale': model_params.get('holidays_prior_scale', 10.0),
                'model.prophet.seasonality_mode': model_params.get('seasonality_mode', 'additive'),
                'model.prophet.changepoint_range': model_params.get('changepoint_range', 0.8)
            }
            self.tracker.log_params_safely(model_params_with_prefix)
            
            # Log training metrics
            metrics_with_prefix = {
                'metrics.train.rmse': metrics['rmse'],
                'metrics.train.mae': metrics['mae'],
                'metrics.train.mape': metrics['mape'],
                'metrics.train.r2': metrics.get('r2')
            }
            self.tracker.log_metrics_safely(metrics_with_prefix)
            
            # Log test metrics
            metrics_with_prefix = {
                'metrics.test.rmse': analysis_results['rmse'],
                'metrics.test.mae': analysis_results['mae'],
                'metrics.test.mape': analysis_results['mape'],
                'metrics.test.r2': analysis_results.get('r2'),
                'metrics.test.directional_accuracy': analysis_results.get('directional_accuracy')
            }
            self.tracker.log_metrics_safely(metrics_with_prefix)

            if grid_search_results:
                # Log grid search results
                grid_search_results_with_prefix = {
                    'grid_search.best_params': grid_search_results['best_params'],
                    'grid_search.best_metrics': grid_search_results['best_metrics']
                }
                self.tracker.log_params_safely(grid_search_results_with_prefix)
            
            # Log artifacts
            self.tracker.log_artifact(model_file_path, "model")
            self.tracker.log_artifact(analysis_path, "analysis")
            
        except Exception as e:
            self._log_error(f"Error tracking results: {str(e)}")
            raise
    
    @handle_pipeline_errors
    def create_model_summary(self, train_data: pd.DataFrame, test_data: pd.DataFrame,
                           model_params: Dict[str, Any], 
                           metrics: Dict[str, Any],
                           analysis_results: Dict[str, Any], 
                           model_path: str, analysis_path: str, 
                           grid_search_results: Optional[Dict[str, Any]] = None,
                           preprocessor: Optional[DataPreprocessor] = None) -> Dict[str, Any]:
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
                'parameter_selection_method': 'optimization' if self.config['model'].get('optimize_hyperparameters') else 'default'
            },
            'best_model': {
                'parameters': convert_to_native_types(model_params),
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
                # Add grid search details if available
        if grid_search_results:
            model_summary['training_details'].update({
                'grid_search_space': convert_to_native_types(grid_search_results.get('param_grid', {})),
                'models_evaluated': int(grid_search_results.get('total_combinations_tested', 0)),
                'convergence_rate': float(grid_search_results.get('convergence_rate', 0)),
                'all_results': convert_to_native_types(grid_search_results.get('all_results', []))
            })
        
        # Add preprocessing pipeline if available
        if preprocessor:
            model_summary['preprocessing_pipeline'] = preprocessor.pipeline_steps
        
        return model_summary
    
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