from typing import Dict, Any, Optional, Tuple, Union, List
import pandas as pd
import numpy as np
import torch
from neuralprophet import NeuralProphet
import optuna

from .prophet_model import ProphetModel
from ..utils.logger import Logger
from ..utils.trackers import ProphetTracker

logger = Logger()

class NeuralProphetModel(ProphetModel):
    """NeuralProphet time series model extending Prophet with neural network capabilities."""
    
    def __init__(
        self,
        config: Dict[str, Any],
        tracker: Optional[ProphetTracker] = None,    
    ):
        """Initialize NeuralProphetModel with parameters."""
        super().__init__(config, tracker)
        logger.info("Initializing NeuralProphetModel")
        
    def prepare_data(self, data: Union[pd.DataFrame, pd.Series, np.ndarray]) -> pd.DataFrame:
        """Prepare data for NeuralProphet model."""
        # Use the same data preparation as Prophet
        return super().prepare_data(data)
        
    def initialize_model(self) -> None:
        """Initialize the NeuralProphet model with data."""
        logger.info("Initializing NeuralProphet model...")
        
        try:
            # Get NeuralProphet specific parameters
            np_params = {
                # Common Prophet parameters
                'seasonality_mode': self.config['model'].get('seasonality_mode', 'additive'),
                'yearly_seasonality': self.config['model'].get('yearly_seasonality', 'auto'),
                'weekly_seasonality': self.config['model'].get('weekly_seasonality', 'auto'),
                'daily_seasonality': self.config['model'].get('daily_seasonality', 'auto'),
                'growth': self.config['model'].get('growth', 'linear'),
                'changepoints_range': self.config['model'].get('changepoint_range', 0.8),
                'n_changepoints': self.config['model'].get('n_changepoints', 25),
                
                # NeuralProphet specific parameters
                'learning_rate': self.config['model'].get('learning_rate', 0.001),
                'epochs': self.config['model'].get('epochs', 100),
                'batch_size': self.config['model'].get('batch_size', 64),
                'n_forecasts': self.config['model'].get('n_forecasts', 1),
                'n_lags': self.config['model'].get('n_lags', 0),
                'num_hidden_layers': self.config['model'].get('num_hidden_layers', 2),
                'dropout': self.config['model'].get('dropout', 0.1),
            }
            
            # Create NeuralProphet model
            self.model = NeuralProphet(**np_params)
            logger.info("NeuralProphet model initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing NeuralProphet model: {str(e)}")
            raise
    
    def fit(self, data: Union[pd.Series, pd.DataFrame, np.ndarray], params: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
        """Fit the NeuralProphet model to the provided data."""
        try:
            # Update parameters if provided
            if params:
                self.config['model'].update(params)
            self.initialize_model()
            
            if self.model is None:
                raise ValueError("Model must be initialized before fitting")
            
            # Prepare data if needed
            if isinstance(data, (pd.Series, np.ndarray)) or (isinstance(data, pd.DataFrame) and ('y' not in data.columns or 'ds' not in data.columns)):
                prepared_data = self.prepare_data(data)
            else:
                prepared_data = data
            
            # Store training data for later use
            self._training_data = prepared_data.copy() if isinstance(prepared_data, pd.DataFrame) else prepared_data
            
            # Fit model
            logger.info("Fitting NeuralProphet model")
            metrics = self.model.fit(prepared_data, **self.config['model'].get('fit_params', {}))
            
            # Calculate metrics on training data
            train_metrics = self._calculate_metrics(prepared_data)
            
            # Convert numpy types to Python native types
            train_metrics = {k: float(v) for k, v in train_metrics.items()}
            
            logger.info("Model training completed successfully")
            logger.info(f"Training metrics: {train_metrics}")
            
            return train_metrics
            
        except Exception as e:
            logger.error(f"Error in model training: {str(e)}")
            raise
    
    def fit_with_optuna(self, data: Union[pd.Series, pd.DataFrame, np.ndarray], n_trials: int = 100, timeout: int = 600) -> Dict[str, Any]:
        """Optimize NeuralProphet hyperparameters using Optuna."""
        if data is None:
            raise ValueError("Data cannot be None")
            
        logger.info(f"Starting hyperparameter optimization with {n_trials} trials and {timeout}s timeout")
        
        def objective(trial):
            """Optimization objective function."""
            params = {
                # Prophet Parameters
                'changepoint_prior_scale': trial.suggest_loguniform('changepoint_prior_scale', 0.001, 0.5),
                'seasonality_prior_scale': trial.suggest_loguniform('seasonality_prior_scale', 0.01, 10),
                'holidays_prior_scale': trial.suggest_loguniform('holidays_prior_scale', 0.01, 10),
                'seasonality_mode': trial.suggest_categorical('seasonality_mode', ['additive', 'multiplicative']),
                
                # NeuralProphet specific parameters
                'learning_rate': trial.suggest_loguniform('learning_rate', 0.0001, 0.1),
                'epochs': trial.suggest_int('epochs', 50, 200),
                'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64, 128]),
                'num_hidden_layers': trial.suggest_int('num_hidden_layers', 1, 4),
                'hidden_size': trial.suggest_int('hidden_size', 32, 256),
                'dropout': trial.suggest_uniform('dropout', 0.1, 0.5),
            }
            
            try:
                # Create and fit model with trial parameters
                self.config['model'].update(params)
                self.initialize_model()
                model_metrics = self.fit(data)
                return float(model_metrics['rmse'])  # Convert to Python float
            except Exception as e:
                logger.warning(f"Trial failed: {str(e)}")
                return float('inf')
        
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=n_trials, timeout=timeout)
        
        best_params = study.best_params
        logger.info(f"Best parameters found: {best_params}")
        
        # Update model with best parameters and fit
        self.config['model'].update(best_params)
        self.initialize_model()
        final_metrics = self.fit(data)
        
        # Convert all metrics to Python native types
        final_metrics = {k: float(v) for k, v in final_metrics.items()}
        
        hyperparameter_optimization_results = {
            'best_params': best_params,
            'best_metrics': final_metrics
        }
        
        if self.tracker:
            logger.debug("Logging optimization results to tracker")
            self.tracker.log_params_safely({
                'optimization.n_trials': n_trials,
                'optimization.timeout': timeout,
                'optimization.best_value': float(study.best_value),
                **{f'optimization.best_params.{k}': float(v) if isinstance(v, (np.floating, np.integer)) else v 
                   for k, v in best_params.items()}
            })
        
        logger.info("Optuna optimization completed successfully")
        return hyperparameter_optimization_results
    
    def save(self, path: str) -> None:
        """Save the model to disk."""
        logger.info(f"Saving NeuralProphet model to {path}")
        try:
            import os
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, 'wb') as f:
                torch.save(self.fitted_model, f)
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            raise
    
    def load(self, path: str) -> None:
        """Load the model from disk."""
        logger.info(f"Loading NeuralProphet model from {path}")
        try:
            with open(path, 'rb') as f:
                self.fitted_model = torch.load(f)
            self.model = self.fitted_model
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise 