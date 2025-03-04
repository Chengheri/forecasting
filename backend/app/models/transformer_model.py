import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np
from .base_model import BaseForecastingModel
import optuna
from sklearn.preprocessing import StandardScaler
from ..utils.logger import Logger

logger = Logger()

class TransformerModel(nn.Module):
    def __init__(self, d_model: int, nhead: int, num_layers: int, dim_feedforward: int = 2048,
                 dropout: float = 0.1):
        super(TransformerModel, self).__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        
        self.embedding = nn.Linear(1, d_model)
        self.pos_encoder = nn.Linear(1, d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        
        self.decoder = nn.Linear(d_model, 1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Add positional encoding
        x = self.embedding(x) + self.pos_encoder(x)
        
        # Pass through transformer encoder
        x = self.transformer_encoder(x)
        
        # Decode the last time step
        x = self.decoder(x[:, -1, :])
        return x

class TransformerForecastingModel(BaseForecastingModel):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        logger.info(f"Initializing Transformer model with config: {config}")
        self.model = TransformerModel(
            d_model=config.get("d_model", 64),
            nhead=config.get("nhead", 4),
            num_layers=config.get("num_layers", 2),
            dim_feedforward=config.get("dim_feedforward", 256),
            dropout=config.get("dropout", 0.1)
        )
        self.scaler = StandardScaler()
        
    def preprocess_data(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        logger.debug("Preprocessing data for Transformer model")
        X = self.scaler.fit_transform(data.drop(columns=["date", "value"]))
        y = data["value"].values
        return X, y
        
    def postprocess_data(self, data: np.ndarray) -> pd.DataFrame:
        logger.debug("Postprocessing Transformer model predictions")
        return pd.DataFrame(data, columns=["value"])
        
    def train(self, data: pd.DataFrame) -> Dict[str, Any]:
        logger.info("Starting Transformer model training")
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
            
        logger.info("Transformer model training completed")
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
        logger.info("Evaluating Transformer model performance")
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
        
        logger.info(f"Transformer model evaluation metrics: {metrics}")
        return metrics
        
    def save(self, path: str) -> None:
        logger.info(f"Saving Transformer model to {path}")
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'scaler': self.scaler,
            'config': self.config,
            'history': self.history
        }, path)
            
    def load(self, path: str) -> None:
        logger.info(f"Loading Transformer model from {path}")
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.scaler = checkpoint['scaler']
        self.config = checkpoint['config']
        self.history = checkpoint['history']
        
    def optimize_hyperparameters(self, data: pd.DataFrame, n_trials: int = 100) -> Dict[str, Any]:
        """Optimize Transformer hyperparameters using Optuna."""
        logger.info(f"Starting hyperparameter optimization with {n_trials} trials")
        def objective(trial):
            hyperparameters = {
                "d_model": trial.suggest_int("d_model", 32, 256),
                "nhead": trial.suggest_int("nhead", 2, 8),
                "num_layers": trial.suggest_int("num_layers", 1, 4),
                "dim_feedforward": trial.suggest_int("dim_feedforward", 128, 512),
                "dropout": trial.suggest_uniform("dropout", 0.1, 0.5),
                "learning_rate": trial.suggest_loguniform("learning_rate", 0.0001, 0.01),
                "epochs": trial.suggest_int("epochs", 50, 200),
            }
            
            # Create and train model with trial hyperparameters
            model = TransformerModel(
                d_model=hyperparameters["d_model"],
                nhead=hyperparameters["nhead"],
                num_layers=hyperparameters["num_layers"],
                dim_feedforward=hyperparameters["dim_feedforward"],
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