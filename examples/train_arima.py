import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

import pandas as pd
import numpy as np
from datetime import datetime
import mlflow
import matplotlib.pyplot as plt
from backend.app.models.arima_model import TimeSeriesModel
from backend.app.utils.trackers import ARIMATracker
from backend.app.utils.logger import Logger
from backend.app.utils.preprocessing import DataPreprocessor
from backend.app.utils.advanced_preprocessing import AdvancedPreprocessor
import json
from backend.app.utils.analysis_utils import analyze_model_results

# Initialize logger
logger = Logger()

def main():
    """Main function to train and optimize ARIMA model."""
    try:
        # Load configuration
        logger.info("Loading configuration from config/config.json")
        with open('config/config.json', 'r') as f:
            config = json.load(f)
        
        # Store initial configuration for summary
        initial_config = {
            'preprocessing': config['preprocessing'].copy(),
            'model': config['model'].copy(),
            'mlflow': config['mlflow'].copy(),
            'paths': config['paths'].copy()
        }
        
        # Initialize logger and MLflow tracker
        logger.info(f"Initializing MLflow tracker for experiment: {config['mlflow']['experiment_name']}")
        tracker = ARIMATracker(experiment_name=config['mlflow']['experiment_name'])
        parent_run_id = mlflow.active_run().info.run_id if mlflow.active_run() else None
        
        # Load and preprocess data
        logger.info(f"Loading data from {config['paths']['data']}")
        df = pd.read_csv(config['paths']['data'])
        df['date'] = pd.to_datetime(df['date'])
        
        # Start preprocessing run with parent suffix
        run_name = f"preprocessing_{parent_run_id[:8] if parent_run_id else ''}"
        with mlflow.start_run(run_name=run_name, nested=True):
            preprocessor = DataPreprocessor(
                config=config['preprocessing'],
                experiment_name=config['mlflow']['experiment_name'],
                run_name=run_name
            )
            df = preprocessor.prepare_data(df)
            mlflow.log_params(config['preprocessing'])
        
        # Rename columns to match ARIMA model expectations
        df = df.rename(columns={'date': 'timestamp', 'value': 'consumption'})
        
        # Split data into train and test sets
        train_size = int(len(df) * 0.8)
        train_data = df[:train_size]
        test_data = df[train_size:]
        logger.info(f"Split data into {len(train_data)} training and {len(test_data)} test samples")
        
        # Grid search for ARIMA parameters
        best_model = None
        best_aic = float('inf')
        best_config = None
        
        # Store results for each model
        model_results = []
        
        for p in range(1, 4):
            for d in [1]:  # Fixed d=1 for first-order differencing
                for q in range(1, 3):
                    run_name = f"arima_p{p}_d{d}_q{q}"
                    
                    with mlflow.start_run(run_name=run_name, nested=True):
                        logger.info(f"Training ARIMA model with p={p}, d={d}, q={q}")
                        
                        # Initialize and train model
                        model_config = {
                            'model_type': config['model']['type'],
                            'p': p,
                            'd': d,
                            'q': q,
                            'trend': config['model']['trend']
                        }
                        model = TimeSeriesModel(config=model_config)
                        
                        # Train the model
                        results = model.train(train_data)
                        
                        # Get model metrics
                        aic = results['aic']
                        bic = results['bic']
                        hqic = results['hqic']
                        
                        # Log parameters and metrics
                        mlflow.log_params(model_config)
                        mlflow.log_metrics({
                            'aic': aic,
                            'bic': bic,
                            'hqic': hqic
                        })
                        
                        # Store results
                        model_results.append({
                            'p': p,
                            'd': d,
                            'q': q,
                            'aic': aic,
                            'bic': bic,
                            'hqic': hqic
                        })
                        
                        # Update best model if current model has lower AIC
                        if aic < best_aic:
                            logger.info(f"New best model found with AIC: {aic}")
                            best_aic = aic
                            best_model = model
                            best_config = model_config.copy()
        
        # Generate predictions with best model
        logger.info(f"Generating predictions for test data with {len(test_data)} steps")
        predictions, (lower, upper) = best_model.predict(len(test_data))
        
        # Evaluate model performance
        logger.info("Evaluating model performance")
        metrics = best_model.evaluate(test_data['consumption'].values, predictions)
        
        # Save best model and results
        model_path = f"data/models/arima_p{best_config['p']}_d{best_config['d']}_q{best_config['q']}/{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(os.path.join(model_path, "model"), exist_ok=True)
        best_model.save_model(os.path.join(model_path, "model", "arima_model.joblib"))
        logger.info(f"Best model saved successfully")
        
        # Log final results
        run_name = f"best_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        with mlflow.start_run(run_name=run_name, nested=True):
            mlflow.log_params(best_config)
            mlflow.log_metrics({
                'best_aic': best_aic,
                'rmse': metrics['rmse'],
                'mae': metrics['mae'],
                'mape': metrics['mape']
            })
        
        # Analyze results
        logger.info("Analyzing model results")
        analysis_path = os.path.join(model_path, "analysis")
        os.makedirs(analysis_path, exist_ok=True)
        
        # Generate and save analysis plots
        analyze_model_results(
            test_data['consumption'].values,
            predictions,
            test_data['timestamp'].values,
            (lower, upper),
            analysis_path
        )
        
        # Create comprehensive model summary
        model_summary = {
            'initial_configuration': initial_config,
            'training_details': {
                'train_samples': len(train_data),
                'test_samples': len(test_data),
                'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'grid_search_space': {
                    'p': list(range(1, 4)),
                    'd': [1],
                    'q': list(range(1, 3))
                },
                'models_evaluated': len(model_results),
                'model_results': model_results
            },
            'best_model': {
                'parameters': best_config,
                'performance': {
                    'aic': float(best_aic),
                    'metrics': {
                        'rmse': float(metrics['rmse']),
                        'mae': float(metrics['mae']),
                        'mape': float(metrics['mape']),
                        'r2': float(metrics['r2']),
                        'directional_accuracy': float(metrics['directional_accuracy'])
                    },
                    'residuals_analysis': {
                        'mean': float(metrics['residuals_mean']),
                        'std': float(metrics['residuals_std']),
                        'skewness': float(metrics['residuals_skewness']),
                        'kurtosis': float(metrics['residuals_kurtosis']),
                        'autocorrelation': float(metrics['residuals_autocorrelation']),
                        'normal_distribution': bool(metrics['residuals_normal']),
                        'independent': bool(metrics['residuals_independent'])
                    }
                },
                'artifacts': {
                    'model_path': os.path.join(model_path, "model", "arima_model.joblib"),
                    'analysis_path': analysis_path,
                    'plots': [
                        'actual_vs_predicted.png',
                        'residuals_analysis.png',
                        'metrics_over_time.png',
                        'seasonal_decomposition.png'
                    ]
                }
            },
            'preprocessing_pipeline': preprocessor.pipeline_steps,
            'mlflow_tracking': {
                'experiment_name': config['mlflow']['experiment_name'],
                'parent_run_id': parent_run_id,
                'best_model_run_name': run_name
            }
        }
        
        # Save comprehensive model summary
        with open(os.path.join(model_path, "model_summary.json"), 'w') as f:
            json.dump(model_summary, f, indent=4)
        logger.info(f"Saved comprehensive model summary to {os.path.join(model_path, 'model_summary.json')}")
        
        # Save metrics summary (keeping this for backward compatibility)
        metrics_summary = {
            'best_model': {
                'p': best_config['p'],
                'd': best_config['d'],
                'q': best_config['q'],
                'aic': float(best_aic),
                'metrics': {
                    'rmse': float(metrics['rmse']),
                    'mae': float(metrics['mae']),
                    'mape': float(metrics['mape']),
                    'r2': float(metrics['r2']),
                    'directional_accuracy': float(metrics['directional_accuracy']),
                    'residuals_mean': float(metrics['residuals_mean']),
                    'residuals_std': float(metrics['residuals_std']),
                    'residuals_skewness': float(metrics['residuals_skewness']),
                    'residuals_kurtosis': float(metrics['residuals_kurtosis']),
                    'residuals_autocorrelation': float(metrics['residuals_autocorrelation']),
                    'residuals_normal': bool(metrics['residuals_normal']),
                    'residuals_independent': bool(metrics['residuals_independent'])
                }
            }
        }
        
        with open(os.path.join(analysis_path, "metrics_summary.json"), 'w') as f:
            json.dump(metrics_summary, f, indent=4)
        
    except Exception as e:
        logger.error(f"Error in ARIMA optimization: {str(e)}")
        raise

if __name__ == "__main__":
    main()