import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Tuple, Dict, Any, Optional
from sklearn.preprocessing import StandardScaler
import pandas as pd
import optuna
from .base_model import BaseForecastingModel
from ..utils.logger import Logger

logger = Logger()

class LSTMForecast(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, output_size: int,
                 dropout: float = 0.2):
        super(LSTMForecast, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))
        
        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        return out

class LSTMTrainer(BaseForecastingModel):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        logger.info(f"Initializing LSTM model with config: {config}")
        self.model = LSTMForecast(
            input_size=config.get("input_size", 1),
            hidden_size=config.get("hidden_size", 64),
            num_layers=config.get("num_layers", 2),
            output_size=1,
            dropout=config.get("dropout", 0.2)
        )
        
    def preprocess_data(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        logger.debug("Preprocessing data for LSTM model")
        X = self.scaler.fit_transform(data.drop(columns=["date", "value"]))
        y = data["value"].values
        return X, y
        
    def postprocess_data(self, data: np.ndarray) -> pd.DataFrame:
        logger.debug("Postprocessing LSTM model predictions")
        return pd.DataFrame(data, columns=["value"])
        
    def train(self, data: pd.DataFrame) -> Dict[str, Any]:
        logger.info("Starting LSTM model training")
        X, y = self.preprocess_data(data)
        X = torch.FloatTensor(X)
        y = torch.FloatTensor(y)
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.get("learning_rate", 0.001))
        criterion = nn.MSELoss()
        
        self.model.train()
        for epoch in range(self.config.get("epochs", 100)):
            optimizer.zero_grad()
            outputs = self.model(X)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            self.history.append({"epoch": epoch, "loss": loss.item()})
            
            if (epoch + 1) % 10 == 0:
                logger.info(f"Epoch {epoch + 1}/{self.config.get('epochs', 100)}, Loss: {loss.item():.4f}")
            
        logger.info("LSTM model training completed")
        return {"history": self.history}
        
    def predict(self, steps: int) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, np.ndarray]]]:
        logger.info(f"Generating predictions for {steps} steps")
        self.model.eval()
        with torch.no_grad():
            # Generate predictions for future steps
            future_data = torch.zeros((steps, self.config.get("input_size", 1)))
            predictions = self.model(future_data).numpy()
            
            # Calculate confidence intervals
            std = np.std(predictions)
            lower_bound = predictions - 1.96 * std
            upper_bound = predictions + 1.96 * std
            
            return predictions, (lower_bound, upper_bound)
        
    def evaluate(self, test_data: pd.DataFrame) -> Dict[str, float]:
        logger.info("Evaluating LSTM model performance")
        X, y_true = self.preprocess_data(test_data)
        X = torch.FloatTensor(X)
        
        self.model.eval()
        with torch.no_grad():
            y_pred = self.model(X).numpy()
            
        metrics = {
            "rmse": np.sqrt(np.mean((y_true - y_pred) ** 2)),
            "mae": np.mean(np.abs(y_true - y_pred)),
            "mape": np.mean(np.abs((y_true - y_pred) / y_true)) * 100,
            "r2": 1 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2)
        }
        
        logger.info(f"LSTM model evaluation metrics: {metrics}")
        return metrics
        
    def save(self, path: str) -> None:
        logger.info(f"Saving LSTM model to {path}")
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'scaler': self.scaler,
            'config': self.config,
            'history': self.history
        }, path)
            
    def load(self, path: str) -> None:
        logger.info(f"Loading LSTM model from {path}")
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.scaler = checkpoint['scaler']
        self.config = checkpoint['config']
        self.history = checkpoint['history']
        
    def optimize_hyperparameters(self, data: pd.DataFrame, n_trials: int = 100) -> Dict[str, Any]:
        """Optimize LSTM hyperparameters using Optuna."""
        logger.info(f"Starting hyperparameter optimization with {n_trials} trials")
        def objective(trial):
            hyperparameters = {
                "hidden_size": trial.suggest_int("hidden_size", 32, 256),
                "num_layers": trial.suggest_int("num_layers", 1, 4),
                "learning_rate": trial.suggest_loguniform("learning_rate", 0.0001, 0.01),
                "epochs": trial.suggest_int("epochs", 50, 200),
                "dropout": trial.suggest_uniform("dropout", 0.1, 0.5),
                "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64, 128]),
                "input_size": data.shape[1] - 1,  # Number of features
            }
            
            # Create and train model with trial hyperparameters
            model = LSTMForecast(
                input_size=hyperparameters["input_size"],
                hidden_size=hyperparameters["hidden_size"],
                num_layers=hyperparameters["num_layers"],
                output_size=1,
                dropout=hyperparameters["dropout"]
            )
            
            X, y = self.preprocess_data(data)
            X = torch.FloatTensor(X)
            y = torch.FloatTensor(y)
            
            optimizer = torch.optim.Adam(model.parameters(), lr=hyperparameters["learning_rate"])
            criterion = nn.MSELoss()
            
            # Training loop
            model.train()
            for epoch in range(hyperparameters["epochs"]):
                optimizer.zero_grad()
                outputs = model(X)
                loss = criterion(outputs, y)
                loss.backward()
                optimizer.step()
            
            # Evaluate model
            model.eval()
            with torch.no_grad():
                y_pred = model(X).numpy()
            
            # Return validation metric (RMSE)
            return np.sqrt(np.mean((y.numpy() - y_pred) ** 2))
        
        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=n_trials)
        
        logger.info(f"Hyperparameter optimization completed. Best parameters: {study.best_params}")
        return study.best_params 