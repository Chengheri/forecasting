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
        
        # Split data into training and test sets
        train_size = int(len(df) * config['model']['train_test_split'])
        train_data = df[:train_size]
        test_data = df[train_size:]
        logger.info(f"Split data into {len(train_data)} training and {len(test_data)} test samples")
        
        # Train and evaluate models
        best_model = None
        best_aic = float('inf')
        best_params = None
        best_results = None
        
        param_grid = config['model']['grid_search']
        for p in param_grid['p']:
            for d in param_grid['d']:
                for q in param_grid['q']:
                    # Start a new run for this configuration
                    run_name = f"arima_p{p}_d{d}_q{q}_{parent_run_id[:8] if parent_run_id else ''}"
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
                        
                        # Update best model if AIC is lower
                        if aic < best_aic:
                            best_aic = aic
                            best_model = model
                            best_params = (p, d, q)
                            best_results = results
                            logger.info(f"New best model found with AIC: {aic}")
                            
                            # Save best model
                            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                            model_dir = f"{config['paths']['models_dir']}/arima_p{p}_d{d}_q{q}/{timestamp}"
                            os.makedirs(f"{model_dir}/model", exist_ok=True)
                            os.makedirs(f"{model_dir}/analysis", exist_ok=True)
                            model.save_model(f"{model_dir}/model/arima_model.joblib")
        
        if best_model is not None:
            # Generate predictions
            predictions, (lower, upper) = best_model.predict(test_data)
            
            # Calculate and log performance metrics
            metrics = best_model.evaluate(test_data['consumption'].values, predictions)
            tracker.log_training_metrics(metrics)
            
            # Create directories for results
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_name = f"arima_p{best_params[0]}_d{best_params[1]}_q{best_params[2]}"
            base_dir = os.path.join(config['paths']['models_dir'], model_name, timestamp)
            analysis_dir = os.path.join(base_dir, config['paths']['analysis_dir'])
            os.makedirs(analysis_dir, exist_ok=True)
            
            # Save the best model
            model_path = os.path.join(base_dir, 'model', 'arima_model.joblib')
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            best_model.save_model(model_path)
            
            # Analyze best model results using the utility function
            analysis_metrics = analyze_model_results(
                actual=test_data['consumption'].values,
                predicted=predictions,
                dates=test_data['timestamp'].values,
                confidence_intervals=(lower, upper),
                output_dir=analysis_dir
            )
            
            # Save predictions and confidence intervals
            results_df = pd.DataFrame({
                'timestamp': test_data['timestamp'],
                'actual': test_data['consumption'],
                'predicted': predictions,
                'lower_bound': lower,
                'upper_bound': upper
            })
            results_df.to_csv(os.path.join(analysis_dir, 'predictions.csv'), index=False)
            
            # Create model summary with all configurations
            summary = {
                'model_name': model_name,
                'timestamp': timestamp,
                'parameters': {
                    'p': best_params[0],
                    'd': best_params[1],
                    'q': best_params[2]
                },
                'metrics': {
                    **metrics,  # Training metrics (mse, rmse, mae, mape)
                    'r2': analysis_metrics['r2'],  # Additional evaluation metric
                    'aic': best_aic,  # Best AIC from model selection
                    'bic': best_results['bic'],  # BIC from best model
                    'hqic': best_results['hqic']  # HQIC from best model
                },
                'data_info': {
                    'train_samples': len(train_data),
                    'test_samples': len(test_data),
                    'features': list(train_data.columns)
                },
                'preprocessing_config': config['preprocessing'],
                'model_config': config['model'],
                'analysis_config': config['analysis']
            }
            
            # Save model summary
            with open(os.path.join(analysis_dir, 'model_summary.json'), 'w') as f:
                json.dump(summary, f, indent=4)
            
            logger.info(f"Model and all related files saved to {base_dir}/")
            logger.info(f"MLflow tracking UI available at: {config['mlflow']['tracking_uri']}")
            
        else:
            logger.error("No successful model found during grid search")
            
    except Exception as e:
        logger.error(f"Error in ARIMA optimization: {str(e)}")
        raise

if __name__ == "__main__":
    main() 