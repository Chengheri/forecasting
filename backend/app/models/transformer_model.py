import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Any, Optional, Tuple, Union, List, Callable
import pandas as pd
import numpy as np
from .torch_model import TorchModel
import optuna
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from ..utils.trackers import ForecastingTracker
from ..utils.logger import Logger

logger = Logger()

class TransformerModel(nn.Module):
    def __init__(self, d_model: int, nhead: int, num_layers: int, input_size: int = 1,
                 dim_feedforward: int = 2048, dropout: float = 0.1, output_size: int = 1):
        super(TransformerModel, self).__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.input_size = input_size
        
        self.embedding = nn.Linear(input_size, d_model)
        self.pos_encoder = nn.Linear(input_size, d_model)
        
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
        
        self.decoder = nn.Linear(d_model, output_size)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Add positional encoding
        x = self.embedding(x.unsqueeze(-1) if x.dim() == 2 else x) + self.pos_encoder(x.unsqueeze(-1) if x.dim() == 2 else x)
        
        # Pass through transformer encoder
        x = self.transformer_encoder(x)
        
        # Decode the last time step
        x = self.decoder(x[:, -1, :])
        return x

class TransformerForecastingModel(TorchModel):
    def __init__(self, config: Dict[str, Any], tracker: Optional[ForecastingTracker] = None):
        super().__init__(config, tracker)
        self.config = config
        self.tracker = tracker
        
        # Initialize data preprocessing
        scaling_method = self.config.get('preprocessing', {}).get('scaling_method', 'standard')
        if scaling_method == 'minmax':
            self.scaler = MinMaxScaler()
        else:
            self.scaler = StandardScaler()
                
    def initialize_model(self) -> None:
        """Initialize the Transformer model with parameters from config."""
        logger.info("Initializing Transformer model...")
        
        try:
            model_params = self.config.get('model')
            
            self.model = TransformerModel(
                input_size=model_params.get('input_size'),
                d_model=model_params.get('d_model'),
                nhead=model_params.get('nhead'),
                num_layers=model_params.get('num_layers'),
                dim_feedforward=model_params.get('dim_feedforward'),
                dropout=model_params.get('dropout'),
                output_size=model_params.get('output_size')
            )
            
            logger.info("Transformer model initialized successfully")
            logger.debug(f"Transformer parameters: {model_params}")
            
        except Exception as e:
            logger.error(f"Error initializing Transformer model: {str(e)}")
            raise