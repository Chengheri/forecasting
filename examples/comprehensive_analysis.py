import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns

from app.models.arima_model import TimeSeriesModel
from app.models.lstm_model import LSTMTrainer
from app.models.model_comparison import ModelComparison
from app.utils.generate_sample_data import generate_consumption_data, add_anomalies
from app.utils.advanced_preprocessing import AdvancedPreprocessor
from app.utils.logger import Logger

logger = Logger()

def run_preprocessing_comparison():
    """Compare different preprocessing methods on sample data."""
    logger.info("Starting preprocessing comparison")
    
    # Generate sample data with anomalies
    logger.info("Generating sample data with anomalies...")
    data = generate_consumption_data(
        start_date="2023-01-01",
        end_date="2023-06-30",
        frequency="H"
    )
    data_with_anomalies = add_anomalies(data, anomaly_rate=0.05)
    logger.info(f"Generated {len(data)} data points with {len(data_with_anomalies)} anomalies")
    
    # Initialize preprocessor
    preprocessor = AdvancedPreprocessor()
    
    # Compare different preprocessing methods
    features = ['consumption', 'temperature', 'humidity']
    
    logger.info("Comparing different anomaly detection methods...")
    preprocessor.plot_anomaly_comparison(
        data_with_anomalies,
        target_column='consumption',
        features=features,
        save_path='anomaly_detection_comparison.png'
    )
    logger.info("Anomaly detection comparison plot saved")
    
    # Clean data using ensemble method
    logger.info("Cleaning data using ensemble method...")
    cleaned_data = preprocessor.clean_data(
        data_with_anomalies,
        target_column='consumption',
        features=features
    )
    
    # Generate cleaning report
    report = preprocessor.get_cleaning_report()
    logger.info("Cleaning Report:")
    logger.info(report)
    
    return data, data_with_anomalies, cleaned_data

def run_model_comparison(data: pd.DataFrame):
    """Compare different forecasting models."""
    logger.info("Starting model comparison")
    
    # Split data into train and test sets
    train_size = int(len(data) * 0.8)
    train_data = data.iloc[:train_size]
    test_data = data.iloc[train_size:]
    logger.info(f"Split data into {len(train_data)} training and {len(test_data)} test samples")
    
    # Initialize model comparison
    comparison = ModelComparison()
    
    # Add models
    logger.info("Initializing models...")
    
    # ARIMA
    arima_config = {
        'model_type': 'arima',
        'p': 2, 'd': 1, 'q': 2
    }
    arima_model = TimeSeriesModel(arima_config)
    comparison.add_model('ARIMA', arima_model, arima_config)
    logger.info("Added ARIMA model")
    
    # SARIMA
    sarima_config = {
        'model_type': 'sarima',
        'p': 2, 'd': 1, 'q': 2,
        'P': 1, 'D': 1, 'Q': 1,
        's': 24
    }
    sarima_model = TimeSeriesModel(sarima_config)
    comparison.add_model('SARIMA', sarima_model, sarima_config)
    logger.info("Added SARIMA model")
    
    # LSTM
    lstm_config = {
        'hidden_size': 64,
        'num_layers': 2,
        'dropout': 0.2,
        'learning_rate': 0.001
    }
    lstm_model = LSTMTrainer(lstm_config)
    comparison.add_model('LSTM', lstm_model, lstm_config)
    logger.info("Added LSTM model")
    
    # Train and evaluate models
    logger.info("Training and evaluating models...")
    results = comparison.train_and_evaluate(train_data, test_data)
    
    # Plot comparisons
    logger.info("Generating comparison plots...")
    comparison.plot_comparison(
        train_data,
        test_data,
        save_path='model_comparison.png'
    )
    logger.info("Model comparison plot saved")
    
    comparison.plot_metrics_comparison(
        save_path='metrics_comparison.png'
    )
    logger.info("Metrics comparison plot saved")
    
    # Generate report
    report = comparison.generate_report()
    logger.info("Model Comparison Report:")
    logger.info(report)
    
    # Get best model
    best_model = comparison.get_best_model(metric='rmse')
    logger.info(f"Best model based on RMSE: {best_model}")
    
    # Calculate ensemble forecast
    logger.info("Calculating ensemble forecast...")
    ensemble_forecast = comparison.calculate_ensemble_forecast()
    logger.info("Ensemble forecast calculated")
    
    return comparison, results

def analyze_preprocessing_impact(data: pd.DataFrame,
                              data_with_anomalies: pd.DataFrame,
                              cleaned_data: pd.DataFrame):
    """Analyze the impact of preprocessing on model performance."""
    logger.info("Starting preprocessing impact analysis")
    
    datasets = {
        'Original': data,
        'With Anomalies': data_with_anomalies,
        'Cleaned': cleaned_data
    }
    
    results = {}
    for name, dataset in datasets.items():
        logger.info(f"Training models on {name} dataset...")
        comparison, metrics = run_model_comparison(dataset)
        results[name] = metrics
        logger.info(f"Completed training on {name} dataset")
    
    # Compare results across datasets
    logger.info("Comparing model performance across datasets...")
    
    # Create comparison plots
    metrics = ['rmse', 'mape']
    fig, axes = plt.subplots(len(metrics), 1, figsize=(15, 10))
    
    for ax, metric in zip(axes, metrics):
        logger.debug(f"Creating comparison plot for {metric}")
        data_for_plot = []
        for dataset_name, dataset_results in results.items():
            for model_name, model_metrics in dataset_results.items():
                data_for_plot.append({
                    'Dataset': dataset_name,
                    'Model': model_name,
                    metric.upper(): model_metrics[metric]
                })
        
        df_plot = pd.DataFrame(data_for_plot)
        sns.barplot(
            data=df_plot,
            x='Model',
            y=metric.upper(),
            hue='Dataset',
            ax=ax
        )
        ax.set_title(f'{metric.upper()} Comparison Across Datasets')
        ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('preprocessing_impact.png')
    logger.info("Preprocessing impact comparison plot saved")
    
    return results

def main():
    """Run comprehensive analysis."""
    logger.info("Starting comprehensive analysis")
    
    try:
        # Run preprocessing comparison
        logger.info("Running preprocessing comparison...")
        data, data_with_anomalies, cleaned_data = run_preprocessing_comparison()
        
        # Run model comparison on cleaned data
        logger.info("Running model comparison...")
        comparison, results = run_model_comparison(cleaned_data)
        
        # Analyze preprocessing impact
        logger.info("Analyzing preprocessing impact...")
        impact_results = analyze_preprocessing_impact(
            data, data_with_anomalies, cleaned_data
        )
        
        logger.info("Comprehensive analysis completed successfully")
    except Exception as e:
        logger.error(f"Error in comprehensive analysis: {str(e)}")
        raise

if __name__ == "__main__":
    main() 