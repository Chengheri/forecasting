import os
from datetime import datetime
from typing import Dict, Any, Tuple, Optional, Union, Callable, TypeVar

import pandas as pd

from backend.app.models.lstm_model import LSTMForecastModel
from backend.app.utils.trackers import ProphetTracker
from backend.app.utils.preprocessing import DataPreprocessor
from backend.app.utils.analyzer import Analyzer
from backend.app.pipelines.base_pipeline import BasePipeline


class LSTMPipeline(BasePipeline):
    """Pipeline for training and evaluating LSTM models."""
    
    def __init__(self, config: Dict[str, Any], tracker: ProphetTracker):
        """Initialize the LSTM pipeline.
        
        Args:
            config: Configuration dictionary
            tracker: MLflow tracker instance
        """
        super().__init__(config=config, tracker=tracker)
        self.config = config
        self.tracker = tracker
        self.analyzer = Analyzer(config=config)
        self.preprocessor = DataPreprocessor(config=config['preprocessing'], tracker=tracker)
    
    def train_model(self, train_data: pd.DataFrame) -> Tuple[LSTMForecastModel, Dict[str, Any], Dict[str, Any], Optional[Dict[str, Any]]]:
        """Train the LSTM model.
        
        Args:
            model: LSTMForecastModel instance
            train_data: Training dataset
        """
        model = LSTMForecastModel(config=self.config, tracker=self.tracker)
        return super().train_model(model, train_data)
