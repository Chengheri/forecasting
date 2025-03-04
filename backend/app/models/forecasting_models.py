from typing import Dict, Any
from .base_model import BaseForecastingModel
from .prophet_model import ProphetModel
from .neuralprophet_model import NeuralProphetModel
from .transformer_model import TransformerForecastingModel
from .lightgbm_model import LightGBMModel
from .xgboost_model import XGBoostModel
from .lstm_model import LSTMTrainer
from ..utils.logger import Logger
from ..utils.mlflow_utils import ARIMATracker
from ..utils.model_trackers import (
    LSTMTracker,
    ProphetTracker,
    LightGBMTracker,
    XGBoostTracker
)

logger = Logger()

def create_model(model_name: str, config: Dict[str, Any]) -> BaseForecastingModel:
    """Create a forecasting model instance based on the model name."""
    model_map = {
        "prophet": ProphetModel,
        "neuralprophet": NeuralProphetModel,
        "transformer": TransformerForecastingModel,
        "lightgbm": LightGBMModel,
        "xgboost": XGBoostModel,
        "lstm": LSTMTrainer
    }
    
    tracker_map = {
        "prophet": ProphetTracker,
        "neuralprophet": ProphetTracker,  # Uses same parameters as Prophet
        "transformer": LSTMTracker,  # Uses similar parameters to LSTM
        "lightgbm": LightGBMTracker,
        "xgboost": XGBoostTracker,
        "lstm": LSTMTracker
    }
    
    if model_name not in model_map:
        logger.error(f"Attempted to create unsupported model: {model_name}")
        raise ValueError(f"Model {model_name} not supported")
    
    logger.info(f"Creating {model_name} model with config: {config}")
    
    # Create model instance
    model = model_map[model_name](config)
    
    # Set up tracker if available
    if model_name in tracker_map:
        tracker = tracker_map[model_name]()
        model.set_tracker(tracker)
        logger.info(f"Added {model_name} tracker to model")
    
    return model 