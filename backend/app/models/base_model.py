from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import optuna
from ..utils.model_trackers import BaseModelTracker
from ..utils.logger import Logger

logger = Logger()

class BaseForecastingModel(ABC):
    def __init__(self, config: Dict[str, Any], tracker: Optional[BaseModelTracker] = None):
        """Initialize the base forecasting model."""
        self.config = config
        self.tracker = tracker
        self.model = None
        self._training_data = None

    def _log_initialization(self) -> None:
        """Log initialization parameters."""
        logger.info(f"Initialized {self.config['model']['model_type'].title()} model...")
        
        if self.tracker:
            logger.debug("Logging model parameters to tracker")
            self.tracker.log_params_safely({
                'model.type': self.config['model']['model_type']
            })
    
    @abstractmethod
    def initialize_model(self) -> None:
        """Initialize the model."""
        pass
    
    @abstractmethod
    def prepare_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare the data for the model."""
        pass
    
    @abstractmethod
    def fit(self, data: pd.DataFrame) -> None:
        """Fit the model to the data."""
        pass

    @abstractmethod
    def fit_with_optuna(self, data: pd.DataFrame, objective: Callable[[optuna.Trial], float], n_trials: int, timeout: int) -> None:
        """Fit the model to the data with optuna."""
                
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=n_trials, timeout=timeout)
        
        best_params = study.best_params
        logger.info(f"Best parameters found: {best_params}")
        
        # Update model with best parameters and train final model
        self.config['model'].update(best_params)
        final_metrics = self.fit(data)
        
        hyperparameter_optimization_results = {
            'best_params': best_params,
            'best_value': float(study.best_value),
            'best_metrics': final_metrics,
            'n_trials': n_trials,
            'timeout': timeout,
            'study': study
        }        
        logger.info("Hyperparameter optimization completed successfully")
        return hyperparameter_optimization_results
    
    @abstractmethod
    def predict(self, data: pd.DataFrame) -> pd.DataFrame:
        """Predict the model on the data."""
        pass
    
    @abstractmethod
    def save(self, path: str) -> None:
        """Save the model to a file."""
        pass
    
    @abstractmethod 
    def load(self, path: str) -> None:
        """Load the model from a file."""
        pass
    
    @abstractmethod
    def cross_validate(self, data: pd.DataFrame) -> None:
        """Cross-validate the model on the data."""
        pass    

    @abstractmethod
    def predict_into_future(self, data: pd.DataFrame) -> pd.DataFrame:
        """Predict the model into the future."""
        pass 

    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate evaluation metrics.
        
        Args:
            y_true: True target values
            y_pred: Predicted values
            
        Returns:
            Dictionary of metrics
        """
        # Handle dimensionality
        if y_true.ndim > 1 and y_true.shape[1] == 1:
            y_true = y_true.ravel()
        if y_pred.ndim > 1 and y_pred.shape[1] == 1:
            y_pred = y_pred.ravel()
            
        # Ensure same length
        min_len = min(len(y_true), len(y_pred))
        y_true = y_true[:min_len]
        y_pred = y_pred[:min_len]
        
        # Handle NaN values
        mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
        y_true = y_true[mask]
        y_pred = y_pred[mask]
        
        # Calculate metrics
        mse = np.mean((y_true - y_pred) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(y_true - y_pred))
        
        # Avoid division by zero in MAPE calculation
        non_zero_mask = y_true != 0
        if np.any(non_zero_mask):
            mape = np.mean(np.abs((y_true[non_zero_mask] - y_pred[non_zero_mask]) / y_true[non_zero_mask])) * 100
        else:
            mape = 0.0
        
        # Calculate R-squared
        if np.var(y_true) > 0:
            r2 = 1 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2)
            # Cap negative R² at -1 for interpretability
            r2 = max(-1.0, r2)
        else:
            r2 = 0.0
        
        metrics = {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'mape': mape,
            'r2': r2
        }
        
        return metrics   
    
        