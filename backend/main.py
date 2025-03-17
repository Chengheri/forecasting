from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
import json
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from app.models.prophet_model import ProphetModel
from app.utils.logger import Logger
import os

logger = Logger()

app = FastAPI(title="Electricity Consumption Forecasting API",
             description="API for forecasting electricity consumption using multiple ML models",
             version="1.0.0")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ForecastRequest(BaseModel):
    start_date: str
    end_date: str
    model_name: str = "prophet"  # default model
    features: Optional[List[str]] = None

class ForecastResponse(BaseModel):
    forecast: List[float]
    dates: List[str]
    model_name: str
    metrics: dict

# Load model configurations
def load_model_config(model_name: str) -> dict:
    try:
        with open(f"config/config_{model_name}.json", "r") as f:
            return json.load(f)
    except FileNotFoundError:
        logger.error(f"Configuration file for {model_name} not found")
        raise HTTPException(status_code=404, detail=f"Configuration for {model_name} not found")

def load_trained_model(model_name: str, config: dict) -> ProphetModel:
    """Load a pre-trained model from disk."""
    try:
        model_path = os.path.join(config['paths']['models_dir'], f"{model_name}_model.joblib")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"No trained model found at {model_path}")
            
        logger.info(f"Loading trained model from {model_path}")
        model = ProphetModel(config=config)
        model.load(model_path)
        return model
    except Exception as e:
        logger.error(f"Error loading trained model: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to load trained model: {str(e)}")

@app.get("/")
async def root():
    return {"message": "Welcome to the Electricity Consumption Forecasting API"}

@app.get("/models")
async def available_models():
    return {
        "models": [
            {
                "name": "prophet",
                "description": "Facebook Prophet model"
            },
            {
                "name": "neuralprophet",
                "description": "Neural Prophet model"
            },
            {
                "name": "lstm",
                "description": "LSTM Deep Learning model"
            },
            {
                "name": "transformer",
                "description": "Transformer model"
            },
            {
                "name": "lightgbm",
                "description": "LightGBM model"
            },
            {
                "name": "xgboost",
                "description": "XGBoost model"
            }
        ]
    }

@app.post("/forecast", response_model=ForecastResponse)
async def create_forecast(request: ForecastRequest):
    try:
        logger.info(f"Received forecast request for model: {request.model_name}")
        
        # Load model configuration
        config = load_model_config(request.model_name)
        
        # Initialize model
        if request.model_name == "prophet":
            # Load pre-trained model
            model = load_trained_model(request.model_name, config)
            
            # Calculate number of forecast steps
            start = datetime.strptime(request.start_date, "%Y-%m-%d")
            end = datetime.strptime(request.end_date, "%Y-%m-%d")
            steps = (end - start).days + 1
            
            # Generate forecast
            forecast_values, (lower_bound, upper_bound) = model.predict(steps=steps)
            
            # Generate dates for the forecast period
            forecast_dates = [(start + timedelta(days=x)).strftime("%Y-%m-%d") 
                            for x in range(steps)]
            
            # Get the latest metrics from the model
            metrics = model._calculate_metrics(model.data) if model.data is not None else {}
            
            return {
                "forecast": forecast_values.tolist(),
                "dates": forecast_dates,
                "model_name": request.model_name,
                "metrics": metrics
            }
        else:
            raise HTTPException(status_code=400, detail=f"Model {request.model_name} not implemented yet")
            
    except Exception as e:
        logger.error(f"Error in forecast endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True) 