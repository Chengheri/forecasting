# Time Series Forecasting System

![Version Badge](https://img.shields.io/badge/version-1.0.0-blue)
![License Badge](https://img.shields.io/badge/license-MIT-green)
![Python Badge](https://img.shields.io/badge/python-3.8%20%7C%203.9%20%7C%203.10-blue)

A flexible and powerful system for time series forecasting, utilizing advanced models like LSTM and Transformer, with feature engineering capabilities including topological data analysis.

## 📋 Table of Contents 

- [Overview](#-overview)
- [Features](#-features)
- [Project Structure](#-project-structure)
- [Architecture](#-architecture)
- [Installation](#-installation)
- [Usage](#-usage)
  - [Configuration](#configuration)
  - [Training Models](#training-models)
  - [Evaluation](#evaluation)
  - [Prediction](#prediction)
- [Available Models](#available-models)
  - [LSTM](#lstm)
  - [Transformer](#transformer)
  - [Prophet](#prophet)
- [Feature Engineering](#feature-engineering)
  - [Temporal Features](#temporal-features)
  - [Topological Data Analysis (TDA)](#topological-data-analysis-tda)
- [Experiment Tracking](#experiment-tracking)
- [Examples](#examples)
- [Contributing](#contributing)
- [License](#license)
- [Citations](#citations)

## 🔍 Overview  

This time series forecasting system is designed to handle various types of temporal data and provide accurate predictions using advanced algorithms. Our framework supports seamless integration of various models, including recurrent neural networks (LSTM) and attention-based architectures (Transformer).

## ✨ Features

- **Advanced data preprocessing**: automatic handling of missing values, anomaly detection, normalization, and feature transformation.
- **State-of-the-art models**: implementation of multiple forecasting models (ARIMA, SARIMA, Prophet, Neural Prophet, Transformer, LSTM, LightGBM, XGBoost) with support for hyperparameter optimization.
- **Feature engineering**: automatic generation of temporal features, lags, and rolling window statistics.
- **Topological Data Analysis (TDA)**: extraction of features based on persistent homology to capture topological structures in time series data.
- **Cross-validation**: support for time series cross-validation for robust model evaluation.
- **Hyperparameter optimization**: integration of Optuna and grid search for automatic hyperparameter tuning.
- **Built-in visualization**: automatic generation of diagnostic and forecast plots.
- **MLflow integration**: tracking of experiments, metrics, parameters, and artifacts.
- **Modular pipeline**: extensible architecture allowing for the addition of new models and features.

## 📁 Project Structure
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

## 🏗️ Architecture

The system is built on a modular pipeline architecture:

1. **Data loading**: data is loaded from various sources (CSV, databases).
2. **Preprocessing**: cleaning, transforming, and preparing data for modeling.
3. **Feature engineering**: generating new features, including optional topological features.
4. **Training**: models are trained on preprocessed data with hyperparameter optimization as needed.
5. **Evaluation**: model performance is evaluated using various metrics.
6. **Prediction**: generating forecasts with confidence intervals.
7. **Analysis**: visualizing and interpreting the results.

## 🚀 Installation

```bash
# Clone the repository
git clone https://github.com/your-username/forecasting.git
cd forecasting

# Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install frontend dependencies
cd frontend
npm install
```

## 📊 Usage

### Configuration

The system uses JSON configuration files to define model and preprocessing parameters. Examples:

```json
// config_lstm.json
{
  "data": {
    "path": "data/examples/consumption_data_france.csv",
    "target_column": "value",
    "date_column": "ds",
    "frequency": "D"
  },
  "preprocessing": {
    "add_tda_features": true,
    "tda": {
      "max_homology_dim": 1,
      "window_sizes": [10, 20, 30]
    }
  },
  "model": {
    "model_type": "lstm",
    "hidden_size": 64,
    "num_layers": 2
  }
}
```

### Training Models

```bash
# Train an LSTM model
python examples/train_lstm.py --config config/config_lstm.json

# Train a Transformer model
python examples/train_transformer.py --config config/config_transformer.json

# Train a Prophet model
python examples/train_prophet.py --config config/config_prophet.json
```

### Evaluation (TODO)

```bash
# Evaluate a trained model
python examples/evaluate_model.py --model_path data/models/lstm_20230101_120000.pt --test_data data/test_data.csv
```

### Prediction (TODO)

```bash
# Generate predictions
python examples/predict.py --model_path data/models/transformer_20230101_120000.pt --horizon 30 --output predictions.csv
```
### Backend`

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

## 🧠 Available Models

### LSTM

Our LSTM implementation offers:
- Optional bidirectionality
- Stacked LSTM layers
- Dropout for regularization
- Support for teacher forcing
- Gradient clipping for stability

Typical configuration:
```json
"model": {
  "model_type": "lstm",
  "hidden_size": 64,
  "num_layers": 2,
  "dropout": 0.2,
  "bidirectional": false
}
```

### Transformer

Our Transformer model for time series includes:
- Multi-head self-attention mechanisms
- Positional encoding
- Encoder-only structure optimized for time series
- Embedding layer for input data

Typical configuration:
```json
"model": {
  "model_type": "transformer",
  "d_model": 64,
  "nhead": 4,
  "num_layers": 2,
  "dim_feedforward": 256
}
```

### Prophet

Integration of Facebook's Prophet model with:
- Support for multiple seasonality
- Holiday detection
- Automatic changepoints
- Confidence intervals

## 🔧 Feature Engineering

### Temporal Features

The system automatically generates:
- Calendar features (day of week, month, quarter, etc.)
- Lagged variables
- Rolling window statistics (mean, std, min, max)
- Trend and seasonality decomposition

### Topological Data Analysis (TDA)

Our optional TDA module extracts features based on persistent homology:
- Time-delay embedding for phase space reconstruction
- Computation of persistence diagrams for different homology dimensions
- Extraction of statistics on persistence values (sum, max, mean, std, entropy)
- Aggregation of features across multiple window sizes

## 📈 Experiment Tracking

The system integrates with MLflow for experiment tracking:
- Automatic logging of model parameters
- Tracking of performance metrics
- Storage of artifacts (trained models, plots)
- UI for exploring and comparing experiments

## 📚 Examples

### Electricity Consumption Forecasting

```python
from backend.app.models.lstm_model import LSTMTrainer
from backend.app.utils.data_loader import DataLoader

# Load data
config = {
    "data": {
        "path": "data/electricity_consumption.csv",
        "target_column": "value",
        "date_column": "date"
    },
    "model": {
        "hidden_size": this 64,
        "num_layers": 2
    }
}

data_loader = DataLoader(config)
data = data_loader.load_csv()

# Create and train the model
model = LSTMTrainer(config)
metrics = model.fit(data)
print(f"Training metrics: {metrics}")

# Generate predictions
forecast_horizon = 30
predictions, (lower_bound, upper_bound) = model.predict(forecast_horizon)
```

## 👥 Contributing

Contributions are welcome! To contribute:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

Please check our contribution guidelines for more details.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 📚 Citations

If you use this system in your research, please cite it:
@software{time_series_forecasting_2025,
author = {BAO Chengheri},
title = {Time Series Forecasting System},
year = {2025},
url = {https://github.com/username/time-series-forecasting}
}


References for methods used:
- LSTM: Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory. Neural computation, 9(8), 1735-1780.
- Transformer: Vaswani, A., et al. (2017). Attention is all you need. In Advances in neural information processing systems.
- TDA: Edelsbrunner, H., & Harer, J. (2010). Computational topology: an introduction. American Mathematical Society.