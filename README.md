# Forecasting System

An advanced forecasting system for time series analysis and prediction.

## Overview

This project is a comprehensive forecasting system that supports multiple forecasting models, featuring a modern web interface and extensive analysis capabilities.

## Features

- Multiple forecasting models support (ARIMA, SARIMA, Prophet, Neural Prophet, Transformer, LSTM, LightGBM, XGBoost)
- Modern web UI with interactive visualizations
- Comprehensive time series analysis
- Detailed performance metrics
- Confidence intervals for predictions
- MLflow experiment tracking
- Detailed logging system

## Project Structure

```
forecasting/
├── backend/
│   ├── app/
│   │   ├── models/          # Forecasting models
│   │   ├── pipelines/       # Data processing pipelines
│   │   └── utils/           # Utility functions
│   ├── tests/               # Test suite
│   └── main.py             # FastAPI application
├── frontend/
│   ├── src/
│   │   ├── components/      # React components
│   │   ├── pages/          # Application pages
│   │   └── utils/          # Frontend utilities
│   └── package.json
└── examples/               # Usage examples
```

## Installation

To install the project:

```bash
# Clone the repository
git clone https://github.com/your-username/forecasting.git
cd forecasting

# Install backend dependencies
pip install -r requirements.txt

# Install frontend dependencies
cd frontend
npm install
```

## Usage

### Backend

To start the backend server:

```bash
cd backend
uvicorn main:app --reload
```

### Frontend

To start the frontend application:

```bash
cd frontend
npm start
```

## Models

The system supports the following models:

- **ARIMA/SARIMA**: For univariate time series with/without seasonality
- **Prophet**: For forecasting with strong seasonal patterns
- **Neural Prophet**: Neural network version of Prophet
- **Transformer**: For complex time series patterns
- **LSTM**: For capturing long-term dependencies
- **LightGBM/XGBoost**: For gradient boosting based forecasting

## Analysis Capabilities

The system provides:

- Stationarity analysis
- Seasonal decomposition
- Residuals analysis
- Performance metrics (RMSE, MAE, MAPE, R², etc.)
- Interactive visualizations
- Confidence intervals

## API Reference

The REST API exposes the following endpoints:

```
POST /forecast
- Generates forecasts for a given time series

GET /models
- Lists available models

POST /analyze
- Performs comprehensive time series analysis
```

## Contributing

Contributions are welcome! Please check CONTRIBUTING.md for guidelines.

## License

This project is licensed under the MIT License. See the LICENSE file for details. 