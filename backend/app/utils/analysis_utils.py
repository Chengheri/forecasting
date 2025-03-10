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
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.stattools import adfuller, kpss
from typing import Dict, Tuple, Union, Any, Optional
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

def plot_residuals_analysis(actual, predicted, dates, maxlags=40):
    """
    Create comprehensive residuals analysis plots.
    
    Args:
        actual (np.array): Actual values
        predicted (np.array): Predicted values
        dates (np.array): Timestamps for the data
        maxlags (int): Maximum number of lags for autocorrelation plot
    
    Returns:
        matplotlib.figure.Figure: Figure containing all residuals analysis plots
    """
    residuals = actual - predicted
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
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
    stats.probplot(residuals, dist="norm", plot=axes[0, 2])
    axes[0, 2].set_title('Q-Q Plot')
    
    # Residuals vs Predicted
    axes[1, 0].scatter(predicted, residuals, alpha=0.5)
    axes[1, 0].axhline(y=0, color='r', linestyle='--')
    axes[1, 0].set_title('Residuals vs Predicted')
    axes[1, 0].set_xlabel('Predicted Values')
    axes[1, 0].set_ylabel('Residuals')
    axes[1, 0].grid(True)
    
    # Residuals autocorrelation
    axes[1, 1].acorr(residuals, maxlags=maxlags)
    axes[1, 1].set_title('Residuals Autocorrelation')
    axes[1, 1].set_xlabel('Lag')
    axes[1, 1].set_ylabel('Correlation')
    axes[1, 1].grid(True)
    
    # Hide the last subplot
    axes[1, 2].set_visible(False)
    
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

def analyze_time_series(data, save_path):
    """
    Analyze time series using ACF, PACF plots, and stationarity tests.
    
    Args:
        data (pd.Series): Time series data
        save_path (str): Path to save the plots
    """
    logger.info("Performing comprehensive time series analysis")
    
    # Create figure with subplots
    fig = plt.figure(figsize=(15, 10))
    gs = fig.add_gridspec(3, 2)
    
    # Plot ACF
    ax1 = fig.add_subplot(gs[0, 0])
    plot_acf(data, ax=ax1, lags=48)  # 48 lags for 2 days of hourly data
    ax1.set_title('Autocorrelation Function (ACF)')
    ax1.set_xlabel('Lag')
    ax1.set_ylabel('Correlation')
    
    # Plot PACF
    ax2 = fig.add_subplot(gs[0, 1])
    plot_pacf(data, ax=ax2, lags=48)
    ax2.set_title('Partial Autocorrelation Function (PACF)')
    ax2.set_xlabel('Lag')
    ax2.set_ylabel('Correlation')
    
    # Plot original time series
    ax3 = fig.add_subplot(gs[1, :])
    ax3.plot(data.index, data.values)
    ax3.set_title('Original Time Series')
    ax3.set_xlabel('Time')
    ax3.set_ylabel('Value')
    ax3.grid(True)
    
    # Perform stationarity tests
    stationarity_results = check_stationarity(data)
    
    # Add stationarity test results as text
    ax4 = fig.add_subplot(gs[2, :])
    ax4.axis('off')
    stationarity_text = (
        f"Stationarity Analysis:\n"
        f"ADF Test p-value: {stationarity_results['adf_test']['pvalue']:.4f}\n"
        f"KPSS Test p-value: {stationarity_results['kpss_test']['pvalue']:.4f}\n"
        f"Overall Assessment: {'Stationary' if stationarity_results['overall_assessment']['is_stationary'] else 'Non-stationary'}\n"
        f"Recommendation: {stationarity_results['overall_assessment']['recommendation']}"
    )
    ax4.text(0.1, 0.5, stationarity_text, fontsize=10, va='center')
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'acf_pacf_analysis.png'))
    plt.close()
    
    # Save stationarity results to a separate JSON file
    with open(os.path.join(save_path, 'stationarity_analysis.json'), 'w') as f:
        json.dump(stationarity_results, f, indent=4)
    
    logger.info(f"Saved comprehensive time series analysis to {save_path}")
    return stationarity_results

def suggest_sarima_parameters(acf_data, pacf_data, stationarity_results, seasonal_period=24):
    """
    Suggest SARIMA parameters based on ACF, PACF, and stationarity analysis.
    
    Args:
        acf_data (np.array): ACF values
        pacf_data (np.array): PACF values
        stationarity_results (dict): Results from stationarity tests
        seasonal_period (int): Seasonal period (default: 24 for hourly data)
    
    Returns:
        dict: Suggested SARIMA parameters
    """
    logger.info("Analyzing ACF and PACF to suggest SARIMA parameters")
    
    # Determine non-seasonal parameters based on ACF and PACF
    p = np.sum(np.abs(pacf_data) > 1.96/np.sqrt(len(pacf_data)))
    q = np.sum(np.abs(acf_data) > 1.96/np.sqrt(len(acf_data)))
    
    # Determine seasonal parameters
    P = np.sum(np.abs(pacf_data[seasonal_period::seasonal_period]) > 1.96/np.sqrt(len(pacf_data)))
    Q = np.sum(np.abs(acf_data[seasonal_period::seasonal_period]) > 1.96/np.sqrt(len(acf_data)))
    
    # Determine differencing based on stationarity results
    if stationarity_results['overall_assessment']['is_stationary']:
        d = 0
        D = 0
    else:
        d = 1
        D = 1
    
    # Adjust parameters based on stationarity recommendations
    if stationarity_results['overall_assessment']['recommendation'] == 'Consider seasonal differencing':
        D = 1
    elif stationarity_results['overall_assessment']['recommendation'] == 'Consider first differencing':
        d = 1
    
    # Ensure parameters are within reasonable bounds
    p = min(max(p, 0), 3)
    q = min(max(q, 0), 3)
    P = min(max(P, 0), 2)
    Q = min(max(Q, 0), 2)
    d = min(max(d, 0), 2)
    D = min(max(D, 0), 2)
    
    params = {
        'p': p,
        'd': d,
        'q': q,
        'P': P,
        'D': D,
        'Q': Q,
        's': seasonal_period
    }
    
    logger.info(f"Suggested SARIMA parameters: {params}")
    return params

def analyze_model_results(actual_values, predicted_values, timestamps, confidence_intervals, save_path):
    """
    Analyze and visualize model results using existing plotting functions.
    
    Args:
        actual_values (np.array): Actual values
        predicted_values (np.array): Predicted values
        timestamps (np.array): Timestamps for the data
        confidence_intervals (tuple): Lower and upper confidence intervals
        save_path (str): Path to save the analysis plots
    """    
    # 1. Actual vs Predicted plot
    fig_pred = plot_actual_vs_predicted(actual_values, predicted_values, timestamps, confidence_intervals)
    pred_path = os.path.join(save_path, 'actual_vs_predicted.png')
    fig_pred.savefig(pred_path)
    plt.close(fig_pred)
    logger.info(f"Saved actual vs predicted plot to {pred_path}")
    
    # 2. Residuals analysis
    fig_resid = plot_residuals_analysis(actual_values, predicted_values, timestamps)
    resid_path = os.path.join(save_path, 'residuals_analysis.png')
    fig_resid.savefig(resid_path)
    plt.close(fig_resid)
    logger.info(f"Saved residuals analysis to {resid_path}")
    
    # 3. Metrics over time
    fig_metrics = plot_metrics_over_time(actual_values, predicted_values, timestamps)
    metrics_path = os.path.join(save_path, 'metrics_over_time.png')
    fig_metrics.savefig(metrics_path)
    plt.close(fig_metrics)
    logger.info(f"Saved metrics over time plot to {metrics_path}")
    
    # 4. Seasonal decomposition
    fig_seasonal = plot_seasonal_decomposition(actual_values, timestamps)
    seasonal_path = os.path.join(save_path, 'seasonal_decomposition.png')
    fig_seasonal.savefig(seasonal_path)
    plt.close(fig_seasonal)
    logger.info(f"Saved seasonal decomposition to {seasonal_path}")
    
    # Calculate and return metrics
    residuals = actual_values - predicted_values
    metrics = {
        'rmse': np.sqrt(np.mean(residuals**2)),
        'mae': np.mean(np.abs(residuals)),
        'mape': np.mean(np.abs(residuals / actual_values)) * 100,
        'r2': 1 - np.sum(residuals**2) / np.sum((actual_values - np.mean(actual_values))**2),
        'directional_accuracy': np.mean(np.sign(np.diff(actual_values)) == np.sign(np.diff(predicted_values))) * 100
    }
    
    return metrics 

def is_stationary(data: Union[np.ndarray, pd.Series], alpha: float = 0.05) -> Dict[str, Any]:
    """Check if a time series is stationary using ADF and KPSS tests.
    
    Args:
        data: Time series data to analyze
        alpha: Significance level for statistical tests
        
    Returns:
        Dict containing test results and overall stationarity assessment
    """
    try:
        # Convert data to numpy array if needed
        if isinstance(data, pd.Series):
            data = data.values
            
        # Perform ADF test (null hypothesis: non-stationary)
        adf_result = adfuller(data)
        adf_pvalue = adf_result[1]
        
        # Perform KPSS test (null hypothesis: stationary)
        kpss_result = kpss(data)
        kpss_pvalue = kpss_result[1]
        
        # Determine stationarity based on both tests
        is_stationary_bool = bool(adf_pvalue < alpha and kpss_pvalue >= alpha)
        
        return {
            'is_stationary': is_stationary_bool,
            'adf_pvalue': adf_pvalue,
            'kpss_pvalue': kpss_pvalue,
            'adf_result': adf_result,
            'kpss_result': kpss_result
        }
    except Exception as e:
        logger.warning(f"Error in stationarity test: {str(e)}")
        return {
            'is_stationary': False,
            'error': str(e)
        }

def check_stationarity(data: Union[np.ndarray, pd.Series], 
                      regression: str = 'c',
                      nlags: str = 'aic',
                      kpss_regression: str = 'c',
                      kpss_nlags: str = 'auto') -> Dict[str, Dict[str, Union[float, bool]]]:
    """Check if a time series is stationary using ADF and KPSS tests."""
    logger.info("Performing stationarity tests on the time series")
    
    try:
        # Get stationarity test results
        stat_results = is_stationary(data)
        
        if 'error' in stat_results:
            raise ValueError(f"Stationarity test failed: {stat_results['error']}")
        
        # Create detailed results dictionary
        results = {
            'adf_test': {
                'test_statistic': float(stat_results['adf_result'][0]),
                'pvalue': float(stat_results['adf_pvalue']),
                'critical_values': {k: float(v) for k, v in stat_results['adf_result'][4].items()},
                'is_stationary': bool(stat_results['adf_pvalue'] < 0.05)
            },
            'kpss_test': {
                'test_statistic': float(stat_results['kpss_result'][0]),
                'pvalue': float(stat_results['kpss_pvalue']),
                'critical_values': {k: float(v) for k, v in stat_results['kpss_result'][3].items()},
                'is_stationary': bool(stat_results['kpss_pvalue'] > 0.05)
            },
            'overall_assessment': {
                'is_stationary': stat_results['is_stationary'],
                'confidence': 'high' if stat_results['is_stationary'] else 'low',
                'recommendation': 'No differencing needed' if stat_results['is_stationary'] else 'Consider differencing'
            }
        }
        
        # Log the results
        logger.info(f"Stationarity test results: {'Stationary' if stat_results['is_stationary'] else 'Non-stationary'}")
        logger.info(f"ADF test p-value: {stat_results['adf_pvalue']:.4f}")
        logger.info(f"KPSS test p-value: {stat_results['kpss_pvalue']:.4f}")
        
        return results
        
    except Exception as e:
        logger.error(f"Error in check_stationarity: {str(e)}")
        raise

def get_suggested_parameters(data: pd.Series, config: dict) -> Dict[str, Any]:
    """Get suggested SARIMA parameters based on time series analysis.
    
    Args:
        data (pd.Series): Time series data
        config (dict): Configuration dictionary
        
    Returns:
        Tuple[dict, dict]: A tuple containing (suggested_params, stationarity_results)
    """
    # Create directories for analysis
    analysis_dir = "data/analysis/stationarity_analysis"
    os.makedirs(analysis_dir, exist_ok=True)
    
    # Perform comprehensive time series analysis
    stationarity_results = analyze_time_series(
        data,
        save_path=analysis_dir
    )
    
    # Analyze ACF and PACF to suggest SARIMA parameters
    acf_data = acf(data, nlags=48)
    pacf_data = pacf(data, nlags=48)
    suggested_params = suggest_sarima_parameters(acf_data, pacf_data, stationarity_results)
    
    return suggested_params, stationarity_results 

def remove_non_stationarity(data: pd.Series, max_diff: int = 2, seasonal_diff: bool = True, 
                          seasonal_period: Optional[int] = None) -> Tuple[pd.Series, Dict[str, int]]:
    """Remove non-stationarity through differencing."""
    def infer_seasonal_period(x: pd.Series) -> int:
        """Infer seasonal period from data."""
        if isinstance(x.index, pd.DatetimeIndex):
            if x.index.freq == 'H' or x.index.inferred_freq == 'H':
                return 24  # Hourly data
            elif x.index.freq == 'D' or x.index.inferred_freq == 'D':
                return 7   # Daily data
            elif x.index.freq == 'M' or x.index.inferred_freq == 'M':
                return 12  # Monthly data
        return 1  # No seasonality
    
    result = data.copy()
    params = {'d': 0, 'D': 0}
    
    # Apply seasonal differencing first if requested
    if seasonal_diff:
        if seasonal_period is None:
            seasonal_period = infer_seasonal_period(data)
        
        if seasonal_period > 1:
            seasonal_diff_data = result.diff(seasonal_period).dropna()
            if is_stationary(seasonal_diff_data)['is_stationary']:
                result = seasonal_diff_data
                params['D'] = 1
                logger.info(f"Applied seasonal differencing with period {seasonal_period}")
    
    # Apply regular differencing until stationary or max_diff reached
    for d in range(max_diff + 1):
        stat_results = is_stationary(result)
        if stat_results['is_stationary']:
            break
        
        if d < max_diff:
            result = result.diff().dropna()
            params['d'] = d + 1
            logger.info(f"Applied regular differencing (order {d + 1})")
    
    return result, params 