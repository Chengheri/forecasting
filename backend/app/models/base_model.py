from typing import Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import optuna

class BaseForecastingModel:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model = None
        self.scaler = StandardScaler()
        self.history = []
        
    def preprocess_data(self, data: pd.DataFrame) -> np.ndarray:
        """Preprocess the data before training or prediction."""
        raise NotImplementedError
        
    def postprocess_data(self, data: np.ndarray) -> pd.DataFrame:
        """Convert predictions back to original scale."""
        raise NotImplementedError
        
    def train(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Train the model on the provided data."""
        raise NotImplementedError
        
    def predict(self, steps: int) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, np.ndarray]]]:
        """Generate predictions for the specified number of steps."""
        raise NotImplementedError
        
    def evaluate(self, test_data: pd.DataFrame) -> Dict[str, float]:
        """Evaluate model performance on test data."""
        raise NotImplementedError
        
    def save(self, path: str) -> None:
        """Save the model to disk."""
        raise NotImplementedError
        
    def load(self, path: str) -> None:
        """Load the model from disk."""
        raise NotImplementedError
        
    def optimize_hyperparameters(self, data: pd.DataFrame, n_trials: int = 100) -> Dict[str, Any]:
        """Optimize hyperparameters using Optuna."""
        raise NotImplementedError 