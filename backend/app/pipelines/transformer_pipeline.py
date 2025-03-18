import os
from datetime import datetime
from typing import Dict, Any, Tuple, Optional, Union, Callable, TypeVar

import pandas as pd

from backend.app.models.transformer_model import TransformerForecastingModel
from backend.app.models.base_model import BaseForecastingModel
from backend.app.utils.trackers import ProphetTracker, LSTMTracker, ForecastingTracker
from backend.app.utils.preprocessing import DataPreprocessor
from backend.app.utils.analyzer import Analyzer
from backend.app.pipelines.base_pipeline import BasePipeline


class TransformerPipeline(BasePipeline):
    """Pipeline for training and evaluating Transformer models."""
    
    def __init__(self, config: Dict[str, Any], tracker: ForecastingTracker):
        """Initialize the Transformer pipeline.
        
        Args:
            config: Configuration dictionary
            tracker: MLflow tracker instance
        """
        super().__init__(config=config, tracker=tracker)
        self.config = config
        self.tracker = tracker
        self.analyzer = Analyzer(config=config)
        self.preprocessor = DataPreprocessor(config=config['preprocessing'], tracker=tracker)
    
    def train_model(self, train_data: pd.DataFrame) -> Tuple[Dict[str, Any], Dict[str, Any], Optional[Dict[str, Any]]]:
        """Train the Transformer model.
        
        Args:
            train_data: Training dataset
            
        Returns:
            Tuple containing (model, training metrics, optimization results)
        """
        model = TransformerForecastingModel(config=self.config)
        
        # Set the tracker if available
        if self.tracker:
            model.tracker = self.tracker
            
        return super().train_model(model, train_data) 