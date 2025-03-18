import os
from datetime import datetime
from typing import Dict, Any, Tuple, Optional, Union, Callable, TypeVar
from functools import wraps

import numpy as np
import pandas as pd

from backend.app.models.prophet_model import ProphetModel
from backend.app.models.neuralprophet_model import NeuralProphetModel
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
        super().__init__(config, tracker)
    
    @handle_pipeline_errors
    def train_model(self, train_data: pd.DataFrame) -> Tuple[Union[ProphetModel, NeuralProphetModel], Dict[str, Any], Dict[str, Any], Optional[Dict[str, Any]]]:
        """Train the Prophet or NeuralProphet model.
        
        Args:
            train_data: Training dataset            
        Returns:
            Tuple containing:
                - Trained model instance
                - Model parameters used
                - Training metrics
                - Grid search results (if performed)
        """                
        # Initialize model based on configuration
        if self.config['model'].get('model_type', 'prophet').lower() == 'neuralprophet':
            model = NeuralProphetModel(config=self.config)
        else:
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
            model.initialize_model()
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
        if isinstance(model, NeuralProphetModel):
            test_data = model.prepare_data(test_data)
            predictions, confidence_intervals = model.predict(test_data)
        else:
            predictions, confidence_intervals = model.predict(len(test_data))
        
        return predictions, confidence_intervals
    
 
