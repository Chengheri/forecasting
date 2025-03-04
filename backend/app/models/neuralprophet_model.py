from typing import Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np
import torch
from neuralprophet import NeuralProphet
from .base_model import BaseForecastingModel
import optuna
from ..utils.logger import Logger

logger = Logger()

class NeuralProphetModel(BaseForecastingModel):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        logger.info(f"Initializing NeuralProphet model with config: {config}")
        self.model = NeuralProphet(**config)
        
    def preprocess_data(self, data: pd.DataFrame) -> np.ndarray:
        logger.debug("Preprocessing data for NeuralProphet model")
        df = data.rename(columns={"date": "ds", "value": "y"})
        return df
        
    def postprocess_data(self, data: np.ndarray) -> pd.DataFrame:
        logger.debug("Postprocessing NeuralProphet model predictions")
        return pd.DataFrame(data, columns=["ds", "y"])
        
    def train(self, data: pd.DataFrame) -> Dict[str, Any]:
        logger.info("Starting NeuralProphet model training")
        df = self.preprocess_data(data)
        self.model.fit(df)
        logger.info("NeuralProphet model training completed")
        return {"status": "success"}
        
    def predict(self, steps: int) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, np.ndarray]]]:
        logger.info(f"Generating predictions for {steps} steps")
        future = self.model.make_future_dataframe(periods=steps)
        forecast = self.model.predict(future)
        predictions = forecast["yhat"].values[-steps:]
        lower_bound = forecast["yhat_lower"].values[-steps:]
        upper_bound = forecast["yhat_upper"].values[-steps:]
        return predictions, (lower_bound, upper_bound)
        
    def evaluate(self, test_data: pd.DataFrame) -> Dict[str, float]:
        logger.info("Evaluating NeuralProphet model performance")
        df = self.preprocess_data(test_data)
        forecast = self.model.predict(df)
        y_true = df["y"].values
        y_pred = forecast["yhat"].values
        
        metrics = {
            "rmse": np.sqrt(np.mean((y_true - y_pred) ** 2)),
            "mae": np.mean(np.abs(y_true - y_pred)),
            "mape": np.mean(np.abs((y_true - y_pred) / y_true)) * 100,
            "r2": 1 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2)
        }
        
        logger.info(f"NeuralProphet model evaluation metrics: {metrics}")
        return metrics
        
    def save(self, path: str) -> None:
        logger.info(f"Saving NeuralProphet model to {path}")
        with open(path, 'wb') as f:
            torch.save(self.model, f)
            
    def load(self, path: str) -> None:
        logger.info(f"Loading NeuralProphet model from {path}")
        with open(path, 'rb') as f:
            self.model = torch.load(f)
            
    def optimize_hyperparameters(self, data: pd.DataFrame, n_trials: int = 100) -> Dict[str, Any]:
        """Optimize NeuralProphet hyperparameters using Optuna."""
        logger.info(f"Starting hyperparameter optimization with {n_trials} trials")
        def objective(trial):
            hyperparameters = {
                "changepoint_prior_scale": trial.suggest_loguniform("changepoint_prior_scale", 0.001, 0.5),
                "seasonality_prior_scale": trial.suggest_loguniform("seasonality_prior_scale", 0.01, 10),
                "holidays_prior_scale": trial.suggest_loguniform("holidays_prior_scale", 0.01, 10),
                "seasonality_mode": trial.suggest_categorical("seasonality_mode", ["additive", "multiplicative"]),
                "learning_rate": trial.suggest_loguniform("learning_rate", 0.0001, 0.1),
                "epochs": trial.suggest_int("epochs", 50, 200),
                "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64, 128]),
                "num_hidden_layers": trial.suggest_int("num_hidden_layers", 1, 4),
                "hidden_layer_size": trial.suggest_int("hidden_layer_size", 32, 256),
                "dropout": trial.suggest_uniform("dropout", 0.1, 0.5),
            }
            
            # Create and train model with trial hyperparameters
            model = NeuralProphet(**hyperparameters)
            df = self.preprocess_data(data)
            model.fit(df)
            
            # Evaluate model
            forecast = model.predict(df)
            y_true = df["y"].values
            y_pred = forecast["yhat"].values
            
            # Return validation metric (RMSE)
            return np.sqrt(np.mean((y_true - y_pred) ** 2))
        
        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=n_trials)
        
        logger.info(f"Hyperparameter optimization completed. Best parameters: {study.best_params}")
        return study.best_params 