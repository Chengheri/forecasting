from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
import json
from datetime import datetime, timedelta

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
        # TODO: Implement actual forecasting logic
        # This is a placeholder response
        start = datetime.strptime(request.start_date, "%Y-%m-%d")
        end = datetime.strptime(request.end_date, "%Y-%m-%d")
        dates = [(start + timedelta(days=x)).strftime("%Y-%m-%d") 
                for x in range((end-start).days + 1)]
        
        return {
            "forecast": [0.0] * len(dates),  # placeholder values
            "dates": dates,
            "model_name": request.model_name,
            "metrics": {
                "mape": 0.0,
                "rmse": 0.0,
                "mae": 0.0
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True) 