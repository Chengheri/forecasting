import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

import pandas as pd
import numpy as np
from datetime import datetime
import mlflow
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from backend.app.models.arima_model import TimeSeriesModel
from backend.app.utils.model_trackers import ARIMATracker
from backend.app.utils.logger import Logger
from backend.app.utils.preprocessing import DataPreprocessor
from backend.app.utils.advanced_preprocessing import AdvancedPreprocessor
import json
from backend.app.utils.analysis_utils import analyze_model_results

# Initialize logger
logger = Logger()

def plot_actual_vs_predicted(actual, predicted, dates, confidence_intervals=None):
    """Plot actual vs predicted values with confidence intervals."""
    plt.figure(figsize=(15, 7))
    plt.plot(dates, actual, label='Actual', color='blue', alpha=0.7)
    plt.plot(dates, predicted, label='Predicted', color='red', linestyle='--')
    
    if confidence_intervals is not None:
        lower, upper = confidence_intervals
        plt.fill_between(dates, lower, upper, color='red', alpha=0.1, label='95% CI')
    
    plt.title('Actual vs Predicted Values')
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    return plt.gcf()

def plot_residuals_analysis(actual, predicted, dates):
    """Create residuals analysis plots."""
    residuals = actual - predicted
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Time series of residuals
    axes[0, 0].plot(dates, residuals)
    axes[0, 0].set_title('Residuals over Time')
    axes[0, 0].set_xlabel('Date')
    axes[0, 0].set_ylabel('Residual')
    axes[0, 0].grid(True)
    
    # Histogram of residuals
    sns.histplot(residuals, kde=True, ax=axes[0, 1])
    axes[0, 1].set_title('Residuals Distribution')
    axes[0, 1].set_xlabel('Residual')
    
    # Q-Q plot
    from scipy import stats
    stats.probplot(residuals, dist="norm", plot=axes[1, 0])
    axes[1, 0].set_title('Q-Q Plot')
    
    # Residuals vs Predicted
    axes[1, 1].scatter(predicted, residuals, alpha=0.5)
    axes[1, 1].set_title('Residuals vs Predicted')
    axes[1, 1].set_xlabel('Predicted Values')
    axes[1, 1].set_ylabel('Residuals')
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    return fig

def plot_metrics_over_time(actual, predicted, dates, window=24):
    """Plot rolling metrics over time."""
    # Calculate rolling metrics using numpy arrays
    rolling_rmse = []
    rolling_mae = []
    
    for i in range(len(actual)):
        start_idx = max(0, i - window + 1)
        act_window = actual[start_idx:i+1]
        pred_window = predicted[start_idx:i+1]
        
        if len(act_window) > 0:
            mse = mean_squared_error(act_window, pred_window)
            mae = mean_absolute_error(act_window, pred_window)
            rolling_rmse.append(np.sqrt(mse))
            rolling_mae.append(mae)
        else:
            rolling_rmse.append(np.nan)
            rolling_mae.append(np.nan)
    
    fig, axes = plt.subplots(2, 1, figsize=(15, 10))
    
    # Plot rolling RMSE
    axes[0].plot(dates, rolling_rmse, label='Rolling RMSE')
    axes[0].set_title(f'Rolling RMSE (window={window})')
    axes[0].set_xlabel('Date')
    axes[0].set_ylabel('RMSE')
    axes[0].grid(True)
    axes[0].legend()
    
    # Plot rolling MAE
    axes[1].plot(dates, rolling_mae, label='Rolling MAE', color='orange')
    axes[1].set_title(f'Rolling MAE (window={window})')
    axes[1].set_xlabel('Date')
    axes[1].set_ylabel('MAE')
    axes[1].grid(True)
    axes[1].legend()
    
    plt.tight_layout()
    return fig

def plot_seasonal_decomposition(data, dates, period=24):
    """Plot seasonal decomposition of the time series."""
    # Create time series
    ts = pd.Series(data, index=dates)
    
    # Perform seasonal decomposition
    decomposition = seasonal_decompose(ts, period=period)
    
    # Plot decomposition
    fig, axes = plt.subplots(4, 1, figsize=(15, 12))
    
    # Original
    axes[0].plot(dates, decomposition.observed)
    axes[0].set_title('Original Time Series')
    axes[0].grid(True)
    
    # Trend
    axes[1].plot(dates, decomposition.trend)
    axes[1].set_title('Trend')
    axes[1].grid(True)
    
    # Seasonal
    axes[2].plot(dates, decomposition.seasonal)
    axes[2].set_title('Seasonal')
    axes[2].grid(True)
    
    # Residual
    axes[3].plot(dates, decomposition.resid)
    axes[3].set_title('Residual')
    axes[3].grid(True)
    
    plt.tight_layout()
    return fig

def analyze_best_model_results(actual, predicted, dates, confidence_intervals, output_dir):
    """Analyze and visualize best model results."""
    logger.info("Analyzing best model results")
    
    # Create analysis directory
    analysis_dir = output_dir
    os.makedirs(analysis_dir, exist_ok=True)
    
    # 1. Actual vs Predicted plot
    fig_pred = plot_actual_vs_predicted(actual, predicted, dates, confidence_intervals)
    pred_path = os.path.join(analysis_dir, 'actual_vs_predicted.png')
    fig_pred.savefig(pred_path)
    plt.close(fig_pred)
    logger.info(f"Saved actual vs predicted plot to {pred_path}")
    
    # 2. Residuals analysis
    fig_resid = plot_residuals_analysis(actual, predicted, dates)
    resid_path = os.path.join(analysis_dir, 'residuals_analysis.png')
    fig_resid.savefig(resid_path)
    plt.close(fig_resid)
    logger.info(f"Saved residuals analysis to {resid_path}")
    
    # 3. Metrics over time
    fig_metrics = plot_metrics_over_time(actual, predicted, dates)
    metrics_path = os.path.join(analysis_dir, 'metrics_over_time.png')
    fig_metrics.savefig(metrics_path)
    plt.close(fig_metrics)
    logger.info(f"Saved metrics over time plot to {metrics_path}")
    
    # 4. Seasonal decomposition
    fig_seasonal = plot_seasonal_decomposition(actual, dates)
    seasonal_path = os.path.join(analysis_dir, 'seasonal_decomposition.png')
    fig_seasonal.savefig(seasonal_path)
    plt.close(fig_seasonal)
    logger.info(f"Saved seasonal decomposition to {seasonal_path}")
    
    # Calculate additional metrics
    mse = mean_squared_error(actual, predicted)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(actual, predicted)
    r2 = r2_score(actual, predicted)
    
    # Save metrics summary
    metrics_summary = {
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'mse': mse
    }
    
    metrics_df = pd.DataFrame([metrics_summary])
    metrics_df.to_csv(os.path.join(analysis_dir, 'metrics_summary.csv'), index=False)
    logger.info(f"Saved metrics summary to {analysis_dir}/metrics_summary.csv")
    
    return metrics_summary

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
        with mlflow.start_run(run_name=f"preprocessing_{parent_run_id[:8] if parent_run_id else ''}", nested=True):
            preprocessor = DataPreprocessor(config=config['preprocessing'])
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
        best_metrics = None
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