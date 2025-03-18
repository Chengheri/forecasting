import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, Any, Optional, Tuple, Union, List, cast
import pandas as pd
import optuna
import joblib
import os
from pathlib import Path
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import matplotlib.pyplot as plt

from ..utils.logger import Logger
from ..utils.trackers import ProphetTracker
from .torch_model import TorchModel
logger = Logger()

class LSTMModel(nn.Module):
    """LSTM neural network for time series forecasting."""
    
    def __init__(self, 
                 input_size: int, 
                 hidden_size: int, 
                 num_layers: int, 
                 output_size: int,
                 dropout: float = 0.2,
                 bidirectional: bool = False):
        """Initialize LSTM model with parameters.
        
        Args:
            input_size: Number of input features
            hidden_size: Number of hidden units in LSTM layers
            num_layers: Number of LSTM layers
            output_size: Number of output features (prediction horizon)
            dropout: Dropout rate for regularization
            bidirectional: Whether to use bidirectional LSTM
        """
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        # Output layer - adjust for bidirectional
        lstm_output_size = hidden_size * 2 if bidirectional else hidden_size
        self.fc = nn.Linear(lstm_output_size, output_size)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_size)
            
        Returns:
            Output tensor of shape (batch_size, output_size)
        """
        # Initialize hidden state with zeros
        batch_size = x.size(0)
        device = x.device
        
        h0 = torch.zeros(self.num_layers * (2 if self.bidirectional else 1), 
                          batch_size, self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers * (2 if self.bidirectional else 1), 
                          batch_size, self.hidden_size).to(device)
        
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))
        
        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        return out

class LSTMForecastModel(TorchModel):
    """LSTM forecasting model with preprocessing, training, and prediction capabilities."""
    
    def __init__(
        self,
        config: Dict[str, Any],
        tracker: Optional[ProphetTracker] = None,
    ):
        """Initialize LSTM model with configuration and tracker."""
        self.config = config
        self.tracker = tracker
        
        # Initialize data preprocessing
        scaling_method = self.config['preprocessing'].get('scaling_method')
        if scaling_method == 'minmax':
            self.scaler = MinMaxScaler()
        else:
            self.scaler = StandardScaler()
        
    def initialize_model(self) -> None:
        """Initialize the LSTM model with parameters from config."""
        logger.info("Initializing LSTM model...")
        
        try:
            model_params = self.config['model']
            
            self.model = LSTMModel(
                input_size=model_params.get('input_size'),
                hidden_size=model_params.get('hidden_size'),
                num_layers=model_params.get('num_layers'),
                output_size=model_params.get('output_size'),
                dropout=model_params.get('dropout'),
                bidirectional=model_params.get('bidirectional')
            )
            
            logger.info("LSTM model initialized successfully")
            logger.debug(f"LSTM parameters: {model_params}")
            
        except Exception as e:
            logger.error(f"Error initializing LSTM model: {str(e)}")
            raise
            