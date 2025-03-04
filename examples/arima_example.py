import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from app.models.arima_model import TimeSeriesModel
from app.utils.generate_sample_data import generate_consumption_data, add_anomalies
from app.utils.logger import Logger

logger = Logger()

def run_arima_example():
    """Run example of ARIMA model for electricity consumption forecasting."""
    logger.info("Starting ARIMA example")
    
    logger.info("Generating sample data...")
    # Generate 6 months of hourly data
    data = generate_consumption_data(
        start_date="2023-01-01",
        end_date="2023-06-30",
        frequency="H"
    )
    logger.info(f"Generated {len(data)} data points")
    
    # Split into train and test
    train_size = int(len(data) * 0.8)
    train_data = data.iloc[:train_size]
    test_data = data.iloc[train_size:]
    logger.info(f"Split data into {len(train_data)} training and {len(test_data)} test samples")
    
    logger.info("Training ARIMA model...")
    # Initialize and train ARIMA model
    arima_config = {
        'model_type': 'arima',
        'p': 2,
        'd': 1,
        'q': 2
    }
    arima_model = TimeSeriesModel(arima_config)
    arima_results = arima_model.train(train_data)
    
    logger.info(f"ARIMA model metrics: AIC={arima_results['aic']:.2f}, BIC={arima_results['bic']:.2f}")
    
    logger.info("Training SARIMA model...")
    # Initialize and train SARIMA model
    sarima_config = {
        'model_type': 'sarima',
        'p': 2, 'd': 1, 'q': 2,
        'P': 1, 'D': 1, 'Q': 1, 's': 24  # Daily seasonality
    }
    sarima_model = TimeSeriesModel(sarima_config)
    sarima_results = sarima_model.train(train_data)
    
    logger.info(f"SARIMA model metrics: AIC={sarima_results['aic']:.2f}, BIC={sarima_results['bic']:.2f}")
    
    # Make predictions
    forecast_steps = len(test_data)
    logger.info(f"Generating forecasts for {forecast_steps} hours...")
    
    arima_forecast, arima_conf = arima_model.predict(forecast_steps)
    sarima_forecast, sarima_conf = sarima_model.predict(forecast_steps)
    logger.info("Forecasts generated successfully")
    
    # Evaluate models
    logger.info("Evaluating models...")
    arima_metrics = arima_model.evaluate(test_data)
    sarima_metrics = sarima_model.evaluate(test_data)
    
    logger.info("Model Evaluation Metrics:")
    logger.info("ARIMA:")
    for metric, value in arima_metrics.items():
        logger.info(f"{metric}: {value:.2f}")
    
    logger.info("SARIMA:")
    for metric, value in sarima_metrics.items():
        logger.info(f"{metric}: {value:.2f}")
    
    # Plot results
    logger.info("Generating comparison plot...")
    plt.figure(figsize=(15, 10))
    
    # Plot training data
    plt.plot(train_data.index, train_data['consumption'],
             label='Training Data', color='blue', alpha=0.5)
    
    # Plot test data
    plt.plot(test_data.index, test_data['consumption'],
             label='Test Data', color='green', alpha=0.5)
    
    # Plot forecasts
    plt.plot(test_data.index, arima_forecast,
             label='ARIMA Forecast', color='red', linestyle='--')
    plt.plot(test_data.index, sarima_forecast,
             label='SARIMA Forecast', color='purple', linestyle='--')
    
    # Plot confidence intervals
    plt.fill_between(test_data.index,
                    arima_conf[0], arima_conf[1],
                    color='red', alpha=0.1)
    plt.fill_between(test_data.index,
                    sarima_conf[0], sarima_conf[1],
                    color='purple', alpha=0.1)
    
    plt.title('Electricity Consumption Forecast')
    plt.xlabel('Date')
    plt.ylabel('Consumption (kWh)')
    plt.legend()
    plt.grid(True)
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Save plot
    plt.savefig('forecast_comparison.png')
    logger.info("Plot saved as 'forecast_comparison.png'")

def run_grid_search_example():
    """Run example of grid search for ARIMA and SARIMA models."""
    logger.info("Starting grid search example")
    
    logger.info("Generating sample data for grid search...")
    data = generate_consumption_data(
        start_date="2023-01-01",
        end_date="2023-03-31",
        frequency="H"
    )
    logger.info(f"Generated {len(data)} data points for grid search")
    
    logger.info("Performing grid search for ARIMA model...")
    arima_param_grid = {
        'p': [1, 2, 3],
        'd': [1],
        'q': [1, 2, 3]
    }
    
    arima_results = TimeSeriesModel.grid_search(data, arima_param_grid)
    
    logger.info("Best ARIMA parameters:")
    logger.info(f"Parameters: {arima_results['best_params']}")
    logger.info(f"AIC: {arima_results['best_aic']:.2f}")
    
    logger.info("Performing grid search for SARIMA model...")
    sarima_param_grid = {
        'p': [1, 2],
        'd': [1],
        'q': [1, 2],
        'P': [0, 1],
        'D': [1],
        'Q': [0, 1],
        's': [24]  # Daily seasonality
    }
    
    sarima_results = TimeSeriesModel.grid_search(data, sarima_param_grid)
    
    logger.info("Best SARIMA parameters:")
    logger.info(f"Parameters: {sarima_results['best_params']}")
    logger.info(f"AIC: {sarima_results['best_aic']:.2f}")

if __name__ == "__main__":
    logger.info("Starting ARIMA/SARIMA examples")
    run_arima_example()
    
    logger.info("Starting grid search example")
    run_grid_search_example()
    logger.info("All examples completed successfully") 