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
from statsmodels.tsa.stattools import acf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from backend.app.models.arima_model import TimeSeriesModel
from backend.app.utils.model_trackers import ARIMATracker
from backend.app.utils.logger import Logger
from backend.app.utils.preprocessing import DataPreprocessor
import json

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
    """Create comprehensive residuals analysis plots."""
    residuals = actual - predicted
    
    fig, axes = plt.subplots(3, 2, figsize=(15, 15))
    
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
    
    # ACF plot of residuals
    plot_acf(residuals, ax=axes[2, 0], lags=40)
    axes[2, 0].set_title('Autocorrelation of Residuals')
    
    # PACF plot of residuals
    plot_pacf(residuals, ax=axes[2, 1], lags=40)
    axes[2, 1].set_title('Partial Autocorrelation of Residuals')
    
    plt.tight_layout()
    return fig

def plot_metrics_over_time(actual, predicted, dates, window=24):
    """Plot rolling metrics over time."""
    # Calculate rolling metrics using numpy arrays
    rolling_rmse = []
    rolling_mae = []
    rolling_mape = []
    
    for i in range(len(actual)):
        start_idx = max(0, i - window + 1)
        act_window = actual[start_idx:i+1]
        pred_window = predicted[start_idx:i+1]
        
        if len(act_window) > 0:
            mse = mean_squared_error(act_window, pred_window)
            mae = mean_absolute_error(act_window, pred_window)
            mape = np.mean(np.abs((act_window - pred_window) / act_window)) * 100
            rolling_rmse.append(np.sqrt(mse))
            rolling_mae.append(mae)
            rolling_mape.append(mape)
        else:
            rolling_rmse.append(np.nan)
            rolling_mae.append(np.nan)
            rolling_mape.append(np.nan)
    
    fig, axes = plt.subplots(3, 1, figsize=(15, 15))
    
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
    
    # Plot rolling MAPE
    axes[2].plot(dates, rolling_mape, label='Rolling MAPE', color='green')
    axes[2].set_title(f'Rolling MAPE (window={window})')
    axes[2].set_xlabel('Date')
    axes[2].set_ylabel('MAPE (%)')
    axes[2].grid(True)
    axes[2].legend()
    
    plt.tight_layout()
    return fig

def plot_seasonal_decomposition(data, dates, period=24):
    """Plot seasonal decomposition with additional analysis."""
    # Create time series
    ts = pd.Series(data, index=dates)
    
    # Perform seasonal decomposition
    decomposition = seasonal_decompose(ts, period=period)
    
    # Plot decomposition
    fig, axes = plt.subplots(5, 1, figsize=(15, 15))
    
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
    
    # Seasonal Pattern
    seasonal_pattern = pd.DataFrame(decomposition.seasonal).iloc[:period]
    axes[4].plot(range(len(seasonal_pattern)), seasonal_pattern.values)
    axes[4].set_title('Seasonal Pattern (One Period)')
    axes[4].set_xlabel('Hour of Day')
    axes[4].grid(True)
    
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
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100
    
    # Calculate seasonal metrics
    seasonal_rmse = []
    seasonal_mae = []
    for hour in range(24):
        hour_mask = pd.to_datetime(dates).hour == hour
        hour_actual = actual[hour_mask]
        hour_predicted = predicted[hour_mask]
        seasonal_rmse.append(np.sqrt(mean_squared_error(hour_actual, hour_predicted)))
        seasonal_mae.append(mean_absolute_error(hour_actual, hour_predicted))
    
    # Save metrics summary
    metrics_summary = {
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'mse': mse,
        'mape': mape,
        'avg_seasonal_rmse': np.mean(seasonal_rmse),
        'avg_seasonal_mae': np.mean(seasonal_mae),
        'max_seasonal_rmse': np.max(seasonal_rmse),
        'min_seasonal_rmse': np.min(seasonal_rmse)
    }
    
    metrics_df = pd.DataFrame([metrics_summary])
    metrics_df.to_csv(os.path.join(analysis_dir, 'metrics_summary.csv'), index=False)
    logger.info(f"Saved metrics summary to {analysis_dir}/metrics_summary.csv")
    
    # Save hourly metrics
    hourly_metrics = pd.DataFrame({
        'hour': range(24),
        'rmse': seasonal_rmse,
        'mae': seasonal_mae
    })
    hourly_metrics.to_csv(os.path.join(analysis_dir, 'hourly_metrics.csv'), index=False)
    logger.info(f"Saved hourly metrics to {analysis_dir}/hourly_metrics.csv")
    
    return metrics_summary

def main():
    """Main function to train and optimize SARIMA model."""
    try:
        # Initialize logger and MLflow tracker
        logger.info("Initializing MLflow tracker for experiment: sarima_optimization")
        tracker = ARIMATracker(experiment_name="sarima_optimization")
        
        # Load and preprocess data
        logger.info("Loading data from data/examples/consumption_data_france.csv")
        df = pd.read_csv("data/examples/consumption_data_france.csv")
        df['date'] = pd.to_datetime(df['date'])
        
        # Initialize preprocessor with configuration
        config = {
            'handle_missing_values': True,
            'missing_values_method': 'interpolate',
            'add_time_features': True,
            'add_lag_features': True,
            'lag_features': [1, 2, 3, 24, 168],  # Include weekly lag
            'add_rolling_features': True,
            'rolling_windows': [24, 168],  # Daily and weekly windows
            'scale_features': True,
            'scaling_method': 'standard',
            'columns_to_scale': ['value', 'temperature', 'humidity']
        }
        
        preprocessor = DataPreprocessor(config=config)
        df = preprocessor.prepare_data(df)
        
        # Rename columns to match model expectations
        df = df.rename(columns={'date': 'timestamp', 'value': 'consumption'})
        
        # Split data into training and test sets (80-20 split)
        train_size = int(len(df) * 0.8)
        train_data = df[:train_size]
        test_data = df[train_size:]
        logger.info(f"Split data into {len(train_data)} training and {len(test_data)} test samples")
        
        # Define parameter grid for SARIMA
        param_grid = {
            'p': [1, 2, 3],
            'd': [1],
            'q': [1, 2],
            'P': [0, 1],
            'D': [1],
            'Q': [0, 1],
            's': [24]  # Daily seasonality
        }
        
        # Initialize model with MLflow tracking
        logger.info("Starting grid search with MLflow tracking")
        best_model = None
        best_aic = float('inf')
        
        for p in param_grid['p']:
            for d in param_grid['d']:
                for q in param_grid['q']:
                    for P in param_grid['P']:
                        for D in param_grid['D']:
                            for Q in param_grid['Q']:
                                for s in param_grid['s']:
                                    config = {
                                        'model_type': 'sarima',
                                        'p': p, 'd': d, 'q': q,
                                        'P': P, 'D': D, 'Q': Q,
                                        's': s
                                    }
                                    
                                    logger.info(f"Initializing TimeSeriesModel with config: {config}")
                                    model = TimeSeriesModel(config=config)
                                    
                                    try:
                                        results = model.train(train_data)
                                        aic = results['aic']
                                        
                                        if aic < best_aic:
                                            best_aic = aic
                                            best_model = model
                                            
                                        # Log metrics
                                        metrics = {
                                            'aic': aic,
                                            'bic': results['bic'],
                                            'hqic': results['hqic']
                                        }
                                        tracker.log_training_metrics(metrics)
                                        
                                    except Exception as e:
                                        logger.error(f"Error training model: {str(e)}")
                                        continue
        
        if best_model is not None:
            # Generate predictions
            predictions, (lower, upper) = best_model.predict(test_data)
            
            # Calculate and log performance metrics
            metrics = best_model.evaluate(test_data['consumption'].values, predictions)
            tracker.log_training_metrics(metrics)
            
            # Create directories for results
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_name = f"sarima_p{best_model.config['p']}_d{best_model.config['d']}_q{best_model.config['q']}_P{best_model.config['P']}_D{best_model.config['D']}_Q{best_model.config['Q']}_s{best_model.config['s']}"
            base_dir = os.path.join('data', 'models', model_name, timestamp)
            
            # Create subdirectories
            model_dir = os.path.join(base_dir, 'model')
            results_dir = os.path.join(base_dir, 'results')
            analysis_dir = os.path.join(base_dir, 'analysis')
            experiment_dir = os.path.join(base_dir, 'experiment')
            
            # Create all directories
            for directory in [model_dir, results_dir, analysis_dir, experiment_dir]:
                os.makedirs(directory, exist_ok=True)
            
            # Save predictions and confidence intervals
            results_df = pd.DataFrame({
                'timestamp': test_data['timestamp'],
                'actual': test_data['consumption'],
                'predicted': predictions,
                'lower_bound': lower,
                'upper_bound': upper
            })
            predictions_path = os.path.join(results_dir, 'predictions.csv')
            results_df.to_csv(predictions_path, index=False)
            
            # Save the best model
            model_path = os.path.join(model_dir, 'sarima_model.joblib')
            best_model.save_model(model_path)
            
            # Save configuration
            config_df = pd.DataFrame([{
                'experiment_name': 'sarima_optimization',
                'timestamp': timestamp,
                'model_type': 'SARIMA',
                **best_model.config
            }])
            config_path = os.path.join(experiment_dir, 'config.csv')
            config_df.to_csv(config_path, index=False)
            
            # Save metrics
            metrics_df = pd.DataFrame([metrics])
            metrics_path = os.path.join(results_dir, 'metrics.csv')
            metrics_df.to_csv(metrics_path, index=False)
            
            # Analyze best model results
            analysis_metrics = analyze_best_model_results(
                actual=test_data['consumption'].values,
                predicted=predictions,
                dates=test_data['timestamp'].values,
                confidence_intervals=(lower, upper),
                output_dir=analysis_dir
            )
            
            # Save analysis metrics
            analysis_metrics_df = pd.DataFrame([analysis_metrics])
            analysis_metrics_path = os.path.join(analysis_dir, 'analysis_metrics.csv')
            analysis_metrics_df.to_csv(analysis_metrics_path, index=False)
            
            # Create a summary file
            summary = {
                'model_name': model_name,
                'timestamp': timestamp,
                'best_params': best_model.config,
                'metrics': metrics,
                'analysis_metrics': analysis_metrics,
                'data_info': {
                    'train_samples': len(train_data),
                    'test_samples': len(test_data),
                    'features': list(train_data.columns),
                    'seasonal_period': best_model.config['s']
                }
            }
            
            with open(os.path.join(base_dir, 'model_summary.json'), 'w') as f:
                json.dump(summary, f, indent=4)
            
            logger.info(f"Model and all related files saved to {base_dir}/")
            logger.info(f"MLflow tracking UI available at: http://localhost:5001")
            
        else:
            logger.error("No successful model found during grid search")
            
    except Exception as e:
        logger.error(f"Error in SARIMA optimization: {str(e)}")
        raise

if __name__ == "__main__":
    main() 