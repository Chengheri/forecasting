from typing import Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np
import torch
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from .base_model import BaseForecastingModel
import optuna
from ..utils.logger import Logger

logger = Logger()

class XGBoostModel(BaseForecastingModel):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        logger.info(f"Initializing XGBoost model with config: {config}")
        self.model = xgb.XGBRegressor(**config)
        self.scaler = StandardScaler()
        
    def preprocess_data(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        logger.debug("Preprocessing data for XGBoost model")
        X = self.scaler.fit_transform(data.drop(columns=["date", "value"]))
        y = data["value"].values
        return X, y
        
    def postprocess_data(self, data: np.ndarray) -> pd.DataFrame:
        logger.debug("Postprocessing XGBoost model predictions")
        return pd.DataFrame(data, columns=["value"])
        
    def train(self, data: pd.DataFrame) -> Dict[str, Any]:
        logger.info("Starting XGBoost model training")
        X, y = self.preprocess_data(data)
        self.model.fit(X, y)
        logger.info("XGBoost model training completed")
        return {"status": "success"}
        
    def predict(self, steps: int) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, np.ndarray]]]:
        logger.info(f"Generating predictions for {steps} steps")
        # Create future features (placeholder)
        future_features = np.zeros((steps, self.model.n_features_in_))
        predictions = self.model.predict(future_features)
        
        # Calculate confidence intervals
        std = np.std(predictions)
        lower_bound = predictions - 1.96 * std
        upper_bound = predictions + 1.96 * std
        
        return predictions, (lower_bound, upper_bound)
        
    def evaluate(self, test_data: pd.DataFrame) -> Dict[str, float]:
        logger.info("Evaluating XGBoost model performance")
        X, y_true = self.preprocess_data(test_data)
        y_pred = self.model.predict(X)
        
        metrics = {
            "rmse": np.sqrt(np.mean((y_true - y_pred) ** 2)),
            "mae": np.mean(np.abs(y_true - y_pred)),
            "mape": np.mean(np.abs((y_true - y_pred) / y_true)) * 100,
            "r2": 1 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2)
        }
        
        logger.info(f"XGBoost model evaluation metrics: {metrics}")
        return metrics
        
    def save(self, path: str) -> None:
        logger.info(f"Saving XGBoost model to {path}")
        self.model.save_model(path)
            
    def load(self, path: str) -> None:
        logger.info(f"Loading XGBoost model from {path}")
        self.model.load_model(path)
        
    def optimize_hyperparameters(self, data: pd.DataFrame, n_trials: int = 100) -> Dict[str, Any]:
        """Optimize XGBoost hyperparameters using Optuna."""
        logger.info(f"Starting hyperparameter optimization with {n_trials} trials")
        def objective(trial):
            hyperparameters = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
                "learning_rate": trial.suggest_loguniform("learning_rate", 0.01, 0.1),
                "max_depth": trial.suggest_int("max_depth", 3, 8),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 7),
                "subsample": trial.suggest_uniform("subsample", 0.6, 0.9),
                "colsample_bytree": trial.suggest_uniform("colsample_bytree", 0.6, 0.9),
                "gamma": trial.suggest_loguniform("gamma", 1e-8, 1.0),
            }
            
            # Create and train model with trial hyperparameters
            model = xgb.XGBRegressor(**hyperparameters)
            X, y = self.preprocess_data(data)
            model.fit(X, y)
            
            # Evaluate model
            y_pred = model.predict(X)
            
            # Return validation metric (RMSE)
            return np.sqrt(np.mean((y - y_pred) ** 2))
        
        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=n_trials)
        
        logger.info(f"Hyperparameter optimization completed. Best parameters: {study.best_params}")
        return study.best_params 