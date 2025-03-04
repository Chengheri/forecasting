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
    def __init__(self, config: Dict[str, Any]):
        """Initialize the base forecasting model."""
        self.config = config
        self.model = None
        self.scaler = StandardScaler()
        self.history = []
        self.tracker = None
        
    def set_tracker(self, tracker: BaseModelTracker) -> None:
        """Set the MLflow tracker for this model."""
        self.tracker = tracker
        
    def preprocess_data(self, data: pd.DataFrame) -> np.ndarray:
        """Preprocess the data before training or prediction."""
        raise NotImplementedError
        
    def postprocess_data(self, data: np.ndarray) -> pd.DataFrame:
        """Convert predictions back to original scale."""
        raise NotImplementedError
        
    def train(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Train the model with MLflow tracking."""
        logger.info("Starting model training")
        try:
            # Start tracking if tracker is set
            if self.tracker:
                self.tracker.start_tracking()
                self.tracker.log_model_params(self.config)
            
            # Preprocess data
            X, y = self.preprocess_data(data)
            
            # Train model
            training_metrics = self._train_model(X, y)
            
            # Log training metrics if tracker is set
            if self.tracker:
                self.tracker.log_training_metrics(training_metrics)
                self.tracker.log_model(self.model)
            
            logger.info("Model training completed successfully")
            return training_metrics
        except Exception as e:
            logger.error(f"Error during model training: {str(e)}")
            raise
        finally:
            if self.tracker:
                self.tracker.end_tracking()
        
    def _train_model(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Internal method to train the model."""
        raise NotImplementedError
        
    def predict(self, steps: int) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, np.ndarray]]]:
        """Generate predictions with MLflow tracking."""
        logger.info(f"Generating predictions for {steps} steps")
        try:
            # Start tracking if tracker is set
            if self.tracker:
                self.tracker.start_tracking()
            
            # Generate predictions
            predictions, confidence_intervals = self._generate_predictions(steps)
            
            # Log predictions if tracker is set
            if self.tracker:
                self.tracker.log_predictions(predictions, None)  # No actual values for future predictions
            
            return predictions, confidence_intervals
        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}")
            raise
        finally:
            if self.tracker:
                self.tracker.end_tracking()
        
    def _generate_predictions(self, steps: int) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, np.ndarray]]]:
        """Internal method to generate predictions."""
        raise NotImplementedError
        
    def evaluate(self, test_data: pd.DataFrame) -> Dict[str, float]:
        """Evaluate the model with MLflow tracking."""
        logger.info("Starting model evaluation")
        try:
            # Start tracking if tracker is set
            if self.tracker:
                self.tracker.start_tracking()
            
            # Generate predictions
            predictions, _ = self.predict(len(test_data))
            
            # Calculate metrics
            metrics = self._calculate_metrics(test_data['value'].values, predictions)
            
            # Log evaluation metrics if tracker is set
            if self.tracker:
                self.tracker.log_evaluation_metrics(metrics)
                self.tracker.log_predictions(predictions, test_data['value'].values)
            
            return metrics
        except Exception as e:
            logger.error(f"Error during model evaluation: {str(e)}")
            raise
        finally:
            if self.tracker:
                self.tracker.end_tracking()
        
    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate evaluation metrics."""
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        
        return {
            'mse': mean_squared_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred),
            'r2': r2_score(y_true, y_pred)
        }
        
    def save(self, path: str) -> None:
        """Save the model to disk."""
        raise NotImplementedError
        
    def load(self, path: str) -> None:
        """Load the model from disk."""
        raise NotImplementedError
        
    def optimize_hyperparameters(self, data: pd.DataFrame, n_trials: int = 100) -> Dict[str, Any]:
        """Optimize hyperparameters using Optuna."""
        raise NotImplementedError 