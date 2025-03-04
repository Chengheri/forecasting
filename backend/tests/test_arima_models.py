import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from app.models.arima_model import TimeSeriesModel
from app.utils.generate_sample_data import generate_consumption_data
from app.utils.logger import Logger

logger = Logger()

@pytest.fixture
def sample_data():
    """Generate sample data for testing."""
    logger.info("Generating sample data for testing")
    data = generate_consumption_data(
        start_date="2023-01-01",
        end_date="2023-03-31",  # 3 months of data
        frequency="H"
    )
    logger.info(f"Generated {len(data)} data points")
    return data

def test_arima_model_initialization():
    """Test ARIMA model initialization."""
    logger.info("Testing ARIMA model initialization")
    config = {'model_type': 'arima', 'p': 1, 'd': 1, 'q': 1}
    model = TimeSeriesModel(config)
    assert model.model_type == 'arima'
    assert model.config == config
    logger.info("ARIMA model initialization test passed")

def test_sarima_model_initialization():
    """Test SARIMA model initialization."""
    logger.info("Testing SARIMA model initialization")
    config = {
        'model_type': 'sarima',
        'p': 1, 'd': 1, 'q': 1,
        'P': 1, 'D': 1, 'Q': 1, 's': 24
    }
    model = TimeSeriesModel(config)
    assert model.model_type == 'sarima'
    assert model.config == config
    logger.info("SARIMA model initialization test passed")

def test_data_preparation(sample_data):
    """Test data preparation functionality."""
    logger.info("Testing data preparation")
    model = TimeSeriesModel({'model_type': 'arima'})
    ts = model.prepare_data(sample_data)
    
    assert isinstance(ts, pd.Series)
    assert ts.index.name == 'timestamp'
    assert not ts.isnull().any()
    logger.info("Data preparation test passed")

def test_arima_training(sample_data):
    """Test ARIMA model training."""
    logger.info("Testing ARIMA model training")
    model = TimeSeriesModel({
        'model_type': 'arima',
        'p': 1,
        'd': 1,
        'q': 1
    })
    
    result = model.train(sample_data)
    
    assert 'aic' in result
    assert 'bic' in result
    assert 'model_type' in result
    assert result['model_type'] == 'arima'
    logger.info(f"ARIMA training test passed with AIC={result['aic']:.2f}, BIC={result['bic']:.2f}")

def test_sarima_training(sample_data):
    """Test SARIMA model training."""
    logger.info("Testing SARIMA model training")
    model = TimeSeriesModel({
        'model_type': 'sarima',
        'p': 1, 'd': 1, 'q': 1,
        'P': 1, 'D': 1, 'Q': 1, 's': 24
    })
    
    result = model.train(sample_data)
    
    assert 'aic' in result
    assert 'bic' in result
    assert 'model_type' in result
    assert result['model_type'] == 'sarima'
    logger.info(f"SARIMA training test passed with AIC={result['aic']:.2f}, BIC={result['bic']:.2f}")

def test_forecasting(sample_data):
    """Test forecasting functionality."""
    logger.info("Testing forecasting functionality")
    model = TimeSeriesModel({
        'model_type': 'arima',
        'p': 1, 'd': 1, 'q': 1
    })
    
    # Train the model
    logger.info("Training model for forecasting test")
    model.train(sample_data)
    
    # Make predictions
    steps = 24  # Forecast next 24 hours
    logger.info(f"Generating {steps}-hour forecast")
    forecast, (lower, upper) = model.predict(steps)
    
    assert len(forecast) == steps
    assert len(lower) == steps
    assert len(upper) == steps
    assert all(lower <= forecast)
    assert all(forecast <= upper)
    logger.info("Forecasting test passed")

def test_model_evaluation(sample_data):
    """Test model evaluation functionality."""
    logger.info("Testing model evaluation")
    # Split data into train and test
    train_data = sample_data.iloc[:-24]  # Use all but last 24 hours for training
    test_data = sample_data.iloc[-24:]   # Use last 24 hours for testing
    logger.info(f"Split data into {len(train_data)} training and {len(test_data)} test samples")
    
    model = TimeSeriesModel({
        'model_type': 'arima',
        'p': 1, 'd': 1, 'q': 1
    })
    
    # Train model
    logger.info("Training model for evaluation test")
    model.train(train_data)
    
    # Evaluate
    logger.info("Evaluating model performance")
    metrics = model.evaluate(test_data)
    
    assert 'mse' in metrics
    assert 'rmse' in metrics
    assert 'mae' in metrics
    assert 'mape' in metrics
    assert all(value >= 0 for value in metrics.values())
    logger.info("Model evaluation test passed")
    logger.info(f"Evaluation metrics: {metrics}")

def test_model_saving_and_loading(sample_data, tmp_path):
    """Test model saving and loading functionality."""
    logger.info("Testing model saving and loading")
    model = TimeSeriesModel({
        'model_type': 'arima',
        'p': 1, 'd': 1, 'q': 1
    })
    
    # Train model
    logger.info("Training model for save/load test")
    model.train(sample_data)
    
    # Save model
    save_path = tmp_path / "arima_model.joblib"
    logger.info(f"Saving model to {save_path}")
    model.save_model(str(save_path))
    
    # Load model
    logger.info("Loading saved model")
    new_model = TimeSeriesModel({'model_type': 'arima'})
    new_model.load_model(str(save_path))
    
    # Compare predictions
    steps = 24
    logger.info(f"Comparing predictions from original and loaded models for {steps} steps")
    forecast1, _ = model.predict(steps)
    forecast2, _ = new_model.predict(steps)
    
    np.testing.assert_array_almost_equal(forecast1, forecast2)
    logger.info("Model save/load test passed")

def test_grid_search(sample_data):
    """Test grid search functionality."""
    logger.info("Testing grid search functionality")
    param_grid = {
        'p': [1, 2],
        'd': [1],
        'q': [1, 2]
    }
    logger.info(f"Parameter grid: {param_grid}")
    
    result = TimeSeriesModel.grid_search(sample_data, param_grid)
    
    assert 'best_params' in result
    assert 'best_aic' in result
    assert 'best_model' in result
    assert isinstance(result['best_params'], dict)
    assert isinstance(result['best_aic'], (float, np.float64))
    logger.info(f"Grid search test passed. Best parameters: {result['best_params']}, Best AIC: {result['best_aic']:.2f}")

if __name__ == '__main__':
    logger.info("Starting ARIMA model tests")
    pytest.main([__file__])
    logger.info("ARIMA model tests completed") 