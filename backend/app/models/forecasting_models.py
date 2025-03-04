from typing import Dict, Any
from .base_model import BaseForecastingModel
from .prophet_model import ProphetModel
from .neuralprophet_model import NeuralProphetModel
from .transformer_model import TransformerForecastingModel
from .lightgbm_model import LightGBMModel
from .xgboost_model import XGBoostModel
from .lstm_model import LSTMTrainer
from ..utils.logger import Logger

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
    
    if model_name not in model_map:
        logger.error(f"Attempted to create unsupported model: {model_name}")
        raise ValueError(f"Model {model_name} not supported")
    
    logger.info(f"Creating {model_name} model with config: {config}")
    return model_map[model_name](config) 