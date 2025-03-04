"""Utility functions for analyzing and visualizing forecasting model results."""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from backend.app.utils.logger import Logger

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

def analyze_model_results(actual, predicted, dates, confidence_intervals, output_dir):
    """Analyze and visualize model results."""
    logger.info("Analyzing model results")
    
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
    
    # Calculate metrics
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