# Electricity Consumption Forecasting

A Python-based project for forecasting electricity consumption using various time series models.

## Features

- Multiple forecasting models (ARIMA, SARIMA, Prophet, NeuralProphet, LSTM, etc.)
- Advanced data preprocessing and anomaly detection
- Comprehensive logging system
- Database integration for data storage
- Model comparison and evaluation tools

## Project Structure

```
forecasting/
├── backend/
│   └── app/
│       ├── core/           # Core functionality
│       ├── models/         # Forecasting models
│       └── utils/          # Utility functions
├── data/
│   └── logs/              # Application logs
└── examples/              # Example scripts
```

## Setup

1. Clone the repository
2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Set up environment variables:
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

## Usage

See the `examples/` directory for usage examples.

## Logging

Logs are stored in `data/logs/` with the format `forecasting_YYYY-MM-DD.log`. Each day gets its own log file, and logs are kept indefinitely. 