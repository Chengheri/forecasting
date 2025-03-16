"""Utility class for analyzing and visualizing forecasting model results."""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import acf, pacf, adfuller, kpss
from typing import Dict, Tuple, Union, Any, Optional, List
import json
from ..utils.logger import Logger

logger = Logger()

class Analyzer:
    # Default configurations
    DEFAULT_ANALYSIS_CONFIG = {
        'rolling_window': 24,
        'maxlags': 48,
        'seasonal_period': 24,
        'significance_level': 0.05,
        'save_path': "data/analysis"
    }
    
    # Plot configurations
    PLOT_CONFIGS = {
        'figsize_large': (15, 7),
        'figsize_medium': (12, 6),
        'figsize_small': (8, 6),
        'style': {
            'actual': {'color': 'blue', 'alpha': 0.7, 'label': 'Actual'},
            'predicted': {'color': 'red', 'linestyle': '--', 'label': 'Predicted'},
            'confidence': {'color': 'red', 'alpha': 0.1, 'label': '95% CI'}
        }
    }
    
    # Stationarity test configurations
    STATIONARITY_CONFIG = {
        'adf_regression': 'c',
        'adf_autolag': 'aic',
        'kpss_regression': 'c',
        'kpss_nlags': 'auto'
    }

    def __init__(self, config: Dict[str, Any], save_path: Optional[str] = None):
        """Initialize the Analyzer with configuration, save path, and tracker.
        
        Args:
            config (Dict[str, Any]): Configuration dictionary
            save_path (Optional[str]): Path to save analysis outputs
            tracker (Optional[Any]): Tracker instance for logging metrics and parameters
        """
        self.config = self.DEFAULT_ANALYSIS_CONFIG.copy()
        if config:
            self.config.update(config.get('analysis', {}))
        
        self.save_path = save_path if save_path else self.config['save_path']
        os.makedirs(self.save_path, exist_ok=True)
        
        self._initialize_parameters()
        self._log_initialization()

    def _initialize_parameters(self) -> None:
        """Initialize analysis parameters from config."""
        self.default_window = self.config.get('rolling_window', self.DEFAULT_ANALYSIS_CONFIG['rolling_window'])
        self.default_maxlags = self.config.get('maxlags', self.DEFAULT_ANALYSIS_CONFIG['maxlags'])
        self.default_seasonal_period = self.config.get('seasonal_period', self.DEFAULT_ANALYSIS_CONFIG['seasonal_period'])
        self.alpha = self.config.get('significance_level', self.DEFAULT_ANALYSIS_CONFIG['significance_level'])

    def _log_initialization(self) -> None:
        """Log initialization parameters."""
        logger.info("Initializing Analyzer")
        logger.debug(f"Initialized Analyzer with config: {self.config}")
        logger.info(f"Using save path for analysis: {self.save_path}")
        logger.debug(f"Default parameters - window: {self.default_window}, maxlags: {self.default_maxlags}, seasonal_period: {self.default_seasonal_period}")

    def _create_figure(self, figsize: Tuple[int, int] = None) -> Tuple[plt.Figure, plt.Axes]:
        """Create a new figure with specified size."""
        figsize = figsize or self.PLOT_CONFIGS['figsize_large']
        return plt.subplots(figsize=figsize)

    def _save_figure(self, fig: plt.Figure, filename: str, save_path:str='') -> None:
        """Save figure to specified path."""
        if not save_path:
            save_path = self.save_path
        filepath = os.path.join(save_path, filename)
        fig.savefig(filepath)
        plt.close(fig)
        logger.debug(f"Saved figure to {filepath}")

    def _save_json(self, data: Dict[str, Any], filename: str, save_path:str='') -> None:
        """Save dictionary as JSON file."""
        if not save_path:
            save_path = self.save_path
        filepath = os.path.join(save_path, filename)
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=4)
        logger.debug(f"Saved JSON to {filepath}")

    def plot_actual_vs_predicted(self, actual: np.ndarray, predicted: np.ndarray, 
                               dates: np.ndarray, confidence_intervals: Optional[Tuple] = None) -> plt.Figure:
        """Plot actual vs predicted values with confidence intervals."""
        logger.info("Plotting actual vs predicted values...")
        fig, ax = self._create_figure()
        
        style = self.PLOT_CONFIGS['style']
        ax.plot(dates, actual, **style['actual'])
        ax.plot(dates, predicted, **style['predicted'])
        
        if confidence_intervals is not None:
            lower, upper = confidence_intervals
            ax.fill_between(dates, lower, upper, **style['confidence'])
        
        ax.set_title('Actual vs Predicted Values')
        ax.set_xlabel('Date')
        ax.set_ylabel('Value')
        ax.legend()
        ax.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()
        return fig

    def plot_residuals_analysis(self, actual: np.ndarray, predicted: np.ndarray, 
                              dates: np.ndarray, maxlags: Optional[int] = None) -> plt.Figure:
        """Create comprehensive residuals analysis plots."""
        maxlags = maxlags if maxlags is not None else self.default_maxlags
        residuals = actual - predicted
        
        logger.info("Plotting residuals analysis...")
        fig = plt.figure(figsize=self.PLOT_CONFIGS['figsize_large'])
        gs = fig.add_gridspec(2, 3)
        
        # Time series of residuals
        ax_ts = fig.add_subplot(gs[0, 0])
        ax_ts.plot(dates, residuals)
        ax_ts.set_title('Residuals over Time')
        ax_ts.set_xlabel('Date')
        ax_ts.set_ylabel('Residual')
        ax_ts.grid(True)
        
        # Histogram of residuals
        ax_hist = fig.add_subplot(gs[0, 1])
        sns.histplot(residuals, kde=True, ax=ax_hist)
        ax_hist.set_title('Residuals Distribution')
        ax_hist.set_xlabel('Residual')
        
        # Q-Q plot
        ax_qq = fig.add_subplot(gs[0, 2])
        stats.probplot(residuals, dist="norm", plot=ax_qq)
        ax_qq.set_title('Q-Q Plot')
        
        # Residuals vs Predicted
        ax_scatter = fig.add_subplot(gs[1, 0])
        ax_scatter.scatter(predicted, residuals, alpha=0.5)
        ax_scatter.axhline(y=0, color='r', linestyle='--')
        ax_scatter.set_title('Residuals vs Predicted')
        ax_scatter.set_xlabel('Predicted Values')
        ax_scatter.set_ylabel('Residuals')
        ax_scatter.grid(True)
        
        # Residuals autocorrelation
        ax_acorr = fig.add_subplot(gs[1, 1])
        ax_acorr.acorr(residuals, maxlags=maxlags)
        ax_acorr.set_title('Residuals Autocorrelation')
        ax_acorr.set_xlabel('Lag')
        ax_acorr.set_ylabel('Correlation')
        ax_acorr.grid(True)
        
        # Hide the last subplot
        fig.add_subplot(gs[1, 2]).set_visible(False)
        
        plt.tight_layout()
        return fig

    def plot_metrics_over_time(self, actual: np.ndarray, predicted: np.ndarray, 
                             dates: np.ndarray, window: Optional[int] = None) -> plt.Figure:
        """Plot rolling metrics over time."""
        window = window if window is not None else self.default_window
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
        
        logger.info("Plotting metrics over time...")
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=self.PLOT_CONFIGS['figsize_large'])
        
        ax1.plot(dates, rolling_rmse, label='Rolling RMSE')
        ax1.set_title(f'Rolling RMSE (window={window})')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('RMSE')
        ax1.grid(True)
        ax1.legend()
        
        ax2.plot(dates, rolling_mae, label='Rolling MAE', color='orange')
        ax2.set_title(f'Rolling MAE (window={window})')
        ax2.set_xlabel('Date')
        ax2.set_ylabel('MAE')
        ax2.grid(True)
        ax2.legend()
        
        plt.tight_layout()
        return fig

    def plot_seasonal_decomposition(self, data: np.ndarray, dates: np.ndarray, 
                                  period: Optional[int] = None) -> plt.Figure:
        """Plot seasonal decomposition of the time series."""
        period = period if period is not None else self.default_seasonal_period
        ts = pd.Series(data, index=dates)
        decomposition = seasonal_decompose(ts, period=period)
        
        logger.info("Plotting seasonal decomposition...")
        fig, axes = plt.subplots(4, 1, figsize=(15, 12))
        
        axes[0].plot(dates, decomposition.observed)
        axes[0].set_title('Original Time Series')
        axes[0].grid(True)
        
        axes[1].plot(dates, decomposition.trend)
        axes[1].set_title('Trend')
        axes[1].grid(True)
        
        axes[2].plot(dates, decomposition.seasonal)
        axes[2].set_title('Seasonal')
        axes[2].grid(True)
        
        axes[3].plot(dates, decomposition.resid)
        axes[3].set_title('Residual')
        axes[3].grid(True)
        
        plt.tight_layout()
        return fig

    def analyze_time_series(self, data: pd.Series) -> Dict[str, Any]:
        """Analyze time series using ACF, PACF plots, and stationarity tests."""
        logger.info("Performing comprehensive time series analysis")
        
        # Create figure with subplots
        fig = plt.figure(figsize=(15, 10))
        gs = fig.add_gridspec(3, 2)
        
        # Plot ACF
        logger.info("Plotting ACF...")
        ax1 = fig.add_subplot(gs[0, 0])
        plot_acf(data, ax=ax1, lags=48)
        ax1.set_title('Autocorrelation Function (ACF)')
        
        # Plot PACF
        logger.info("Plotting PACF...")
        ax2 = fig.add_subplot(gs[0, 1])
        plot_pacf(data, ax=ax2, lags=48)
        ax2.set_title('Partial Autocorrelation Function (PACF)')
        
        # Plot original time series
        logger.info("Plotting original time series...")
        ax3 = fig.add_subplot(gs[1, :])
        ax3.plot(data.index, data.values)
        ax3.set_title('Original Time Series')
        ax3.grid(True)
        
        # Perform stationarity tests
        stationarity_results = self.check_stationarity(data)
        
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
        
        plt.tight_layout()
        self._save_figure(fig, 'acf_pacf_analysis.png', save_path=self.save_path)
        self._save_json(stationarity_results, 'stationarity_analysis.json', save_path=self.save_path)
        
        logger.info(f"Saved comprehensive time series analysis to {self.save_path}")
        return stationarity_results

    def analyze_model_results(self, actual_values: np.ndarray, predicted_values: np.ndarray,
                            timestamps: np.ndarray, confidence_intervals: Optional[Tuple] = None, save_path:str='') -> Dict[str, float]:
        """Analyze and visualize model results."""
        # Generate and save plots
        fig_pred = self.plot_actual_vs_predicted(actual_values, predicted_values, timestamps, confidence_intervals)
        self._save_figure(fig_pred, 'actual_vs_predicted.png', save_path=save_path)
        
        fig_resid = self.plot_residuals_analysis(actual_values, predicted_values, timestamps)
        self._save_figure(fig_resid, 'residuals_analysis.png', save_path=save_path)
        
        fig_metrics = self.plot_metrics_over_time(actual_values, predicted_values, timestamps)
        self._save_figure(fig_metrics, 'metrics_over_time.png', save_path=save_path)
        
        fig_seasonal = self.plot_seasonal_decomposition(actual_values, timestamps)
        self._save_figure(fig_seasonal, 'seasonal_decomposition.png', save_path=save_path)
        
        metrics = self.evaluate(actual_values, predicted_values)
        return metrics

    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Evaluate model performance."""
        logger.info("Starting model evaluation")
        logger.debug(f"True values shape: {y_true.shape}, Predicted values shape: {y_pred.shape}")
        
        try:
            logger.info("Calculating performance metrics...")
            # Calculate metrics
            residuals = y_true - y_pred
            metrics = {
                'rmse': np.sqrt(np.mean(residuals**2)),
                'mae': np.mean(np.abs(residuals)),
                'mape': np.mean(np.abs(residuals / y_true)) * 100,
                'r2': 1 - np.sum(residuals**2) / np.sum((y_true - np.mean(y_true))**2),
                'directional_accuracy': np.mean(np.sign(np.diff(y_true)) == np.sign(np.diff(y_pred))) * 100,
                'residuals_mean': np.mean(residuals),
                'residuals_std': np.std(residuals),
                'residuals_skewness': stats.skew(residuals),
                'residuals_kurtosis': stats.kurtosis(residuals),
                'residuals_normal': stats.normaltest(residuals)[1] > 0.05,
                'residuals_autocorrelation': acf(residuals)[1] if len(residuals) > 1 else 0,
                'residuals_independent': np.abs(acf(residuals)[1]) < 1.96/np.sqrt(len(residuals)) if len(residuals) > 1 else True
            }
            logger.debug(f"Calculated metrics: {metrics}")   
            logger.info("Model evaluation completed successfully")
            return metrics
        except Exception as e:
            logger.error(f"Error in model evaluation: {str(e)}")
            raise

    def _perform_stationarity_tests(self, data: Union[np.ndarray, pd.Series]) -> Dict[str, Any]:
        """Perform ADF and KPSS tests on the data."""
        try:
            if isinstance(data, pd.Series):
                data = data.values
            
            adf_result = adfuller(
                data,
                regression=self.STATIONARITY_CONFIG['adf_regression'],
                autolag=self.STATIONARITY_CONFIG['adf_autolag']
            )
            
            kpss_result = kpss(
                data,
                regression=self.STATIONARITY_CONFIG['kpss_regression'],
                nlags=self.STATIONARITY_CONFIG['kpss_nlags']
            )
            
            return {
                'adf_test': {
                    'test_statistic': float(adf_result[0]),
                    'pvalue': float(adf_result[1]),
                    'critical_values': {k: float(v) for k, v in adf_result[4].items()},
                    'is_stationary': bool(adf_result[1] < self.alpha)
                },
                'kpss_test': {
                    'test_statistic': float(kpss_result[0]),
                    'pvalue': float(kpss_result[1]),
                    'critical_values': {k: float(v) for k, v in kpss_result[3].items()},
                    'is_stationary': bool(kpss_result[1] >= self.alpha)
                }
            }
        except Exception as e:
            logger.error(f"Error in stationarity tests: {str(e)}")
            raise

    def _evaluate_stationarity(self, test_results: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate stationarity test results and provide recommendations."""
        adf_stationary = test_results['adf_test']['is_stationary']
        kpss_stationary = test_results['kpss_test']['is_stationary']
        
        is_stationary = adf_stationary and kpss_stationary
        confidence = 'high' if adf_stationary == kpss_stationary else 'medium'
        
        if is_stationary:
            recommendation = 'No differencing needed'
        else:
            recommendations = []
            if not adf_stationary:
                recommendations.append('regular differencing')
            if not kpss_stationary:
                recommendations.append('seasonal differencing')
            recommendation = f"Consider {' and '.join(recommendations)}"
        
        return {
            'is_stationary': is_stationary,
            'confidence': confidence,
            'recommendation': recommendation
        }

    def check_stationarity(self, data: Union[np.ndarray, pd.Series], 
                          alpha: Optional[float] = None) -> Dict[str, Dict[str, Union[float, bool]]]:
        """Check if a time series is stationary using ADF and KPSS tests."""
        alpha = alpha if alpha is not None else self.alpha
        logger.info("Performing stationarity tests on the time series")
        
        try:
            test_results = self._perform_stationarity_tests(data)
            assessment = self._evaluate_stationarity(test_results)
            
            results = {
                'adf_test': test_results['adf_test'],
                'kpss_test': test_results['kpss_test'],
                'overall_assessment': assessment,
                'is_stationary': assessment['is_stationary']
            }
            
            logger.info(f"Stationarity test results: {'Stationary' if assessment['is_stationary'] else 'Non-stationary'}")
            logger.info(f"ADF test p-value: {test_results['adf_test']['pvalue']:.4f}")
            logger.info(f"KPSS test p-value: {test_results['kpss_test']['pvalue']:.4f}")
            logger.info("Stationarity analysis completed")

            return results
            
        except Exception as e:
            logger.error(f"Error in check_stationarity: {str(e)}")
            raise

    def _calculate_acf_pacf(self, data: pd.Series, nlags: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate ACF and PACF values for the time series."""
        nlags = nlags if nlags is not None else self.default_maxlags
        try:
            acf_values = acf(data, nlags=nlags)
            pacf_values = pacf(data, nlags=nlags)
            return acf_values, pacf_values
        except Exception as e:
            logger.error(f"Error calculating ACF/PACF: {str(e)}")
            raise

    def _suggest_order_parameters(self, acf_values: np.ndarray, pacf_values: np.ndarray,
                                seasonal_period: int) -> Dict[str, int]:
        """Suggest order parameters based on ACF and PACF analysis."""
        try:
            sig_threshold = 1.96 / np.sqrt(len(acf_values))
            
            p = np.sum(np.abs(pacf_values) > sig_threshold)
            q = np.sum(np.abs(acf_values) > sig_threshold)
            
            P = Q = 0
            if seasonal_period > 1:
                P = np.sum(np.abs(pacf_values[seasonal_period::seasonal_period]) > sig_threshold)
                Q = np.sum(np.abs(acf_values[seasonal_period::seasonal_period]) > sig_threshold)
            
            params = {
                'p': min(max(p, 0), 3),
                'q': min(max(q, 0), 3),
                'P': min(max(P, 0), 2),
                'Q': min(max(Q, 0), 2)
            }
            
            logger.debug(f"Suggested order parameters: {params}")
            return params
            
        except Exception as e:
            logger.error(f"Error suggesting order parameters: {str(e)}")
            raise

    def _infer_seasonal_period(self, data: pd.Series) -> int:
        """Infer seasonal period from data."""
        if isinstance(data.index, pd.DatetimeIndex):
            freq = data.index.freq or data.index.inferred_freq
            if freq in ['H', 'h']:
                return 24  # Hourly data
            elif freq in ['D', 'd']:
                return 7   # Daily data
            elif freq in ['M', 'm']:
                return 12  # Monthly data
        return 1  # No seasonality

    def remove_non_stationarity(self, data: Union[pd.DataFrame, pd.Series, np.ndarray]) -> Tuple[pd.Series, Dict[str, Any]]:
        """Format input data into a pandas Series and handle non-stationarity if needed.
        
        Args:
            data: Input data in various formats (DataFrame, Series, or numpy array)
            
        Returns:
            Tuple containing:
                - Formatted and (optionally) differenced time series
                - Dictionary with stationarity results and differencing parameters
        """
        logger.info("Starting data formatting and stationarity analysis")
        logger.debug(f"Input data type: {type(data)}")
        
        try:
            # Convert input to pandas Series
            ts = self._convert_to_series(data)
            
            # Initialize parameters
            params = {'d': 0, 'D': 0}
            seasonal = self.config.get('model_type', '') == 'sarima'
            seasonal_period = self.config.get('s', None) if seasonal else None
            max_diff = self.config.get('max_diff', 2)
            
            # Check initial stationarity
            logger.info("Starting stationarity analysis before differencing...")
            stationarity_check = self.check_stationarity(ts)
            
            if not stationarity_check['is_stationary']:
                logger.info("Data is non-stationary. Applying differencing...")
                
                # Handle seasonal differencing first if applicable
                if seasonal and (seasonal_period or self._infer_seasonal_period(ts)) > 1:
                    seasonal_period = seasonal_period or self._infer_seasonal_period(ts)
                    logger.debug(f"Applying seasonal differencing with period {seasonal_period}")
                    
                    seasonal_diff = ts.diff(seasonal_period).dropna()
                    seasonal_check = self.check_stationarity(seasonal_diff)
                    
                    if seasonal_check['is_stationary']:
                        ts = seasonal_diff
                        params['D'] = 1
                        logger.info("Seasonal differencing improved stationarity")
                    else:
                        logger.info("Seasonal differencing did not improve stationarity")
                
                # Apply regular differencing if still needed
                for d in range(max_diff + 1):
                    current_check = self.check_stationarity(ts)
                    if current_check['is_stationary']:
                        stationarity_check = current_check
                        break
                    
                    if d < max_diff:
                        ts = ts.diff().dropna()
                        params['d'] = d + 1
                        logger.info(f"Applied regular differencing (order {d + 1})")
            
            results = {
                'stationarity_check': stationarity_check,
                'differencing_params': params,
                'seasonal_period': seasonal_period if seasonal else None
            }
            
            logger.info("Data formatting and stationarity analysis completed")
            logger.debug(f"Final differencing parameters: d={params['d']}, D={params['D']}")
            
            return ts, results
                
        except Exception as e:
            logger.error(f"Error in remove_non_stationarity: {str(e)}")
            raise

    def _convert_to_series(self, data: Union[pd.DataFrame, pd.Series, np.ndarray]) -> pd.Series:
        """Convert input data to pandas Series."""
        if isinstance(data, np.ndarray):
            logger.debug("Converting numpy array to pandas Series")
            if data.ndim > 1:
                logger.debug("Flattening multi-dimensional array")
                data = data.ravel()
            return pd.Series(data)
        elif isinstance(data, pd.DataFrame):
            logger.debug("Processing DataFrame")
            if 'timestamp' in data.columns and 'consumption' in data.columns:
                logger.debug("Using timestamp and consumption columns")
                return data.set_index('timestamp')['consumption']
            elif len(data.columns) == 1:
                logger.debug("Converting single-column DataFrame to Series")
                return data[data.columns[0]]
            else:
                logger.error("Invalid DataFrame format")
                raise ValueError("DataFrame must contain 'timestamp' and 'consumption' columns or be single column")
        elif isinstance(data, pd.Series):
            logger.debug("Input is already a pandas Series")
            return data
        else:
            logger.error(f"Unsupported data type: {type(data)}")
            raise ValueError("Data must be pandas DataFrame, Series, or numpy array")

    def suggest_model_parameters(self, data: pd.Series, config: dict) -> Dict[str, Any]:
        """Get suggested model parameters based on time series analysis."""
        logger.info("Starting parameter suggestion analysis")
        
        try:
            analysis_dir = os.path.join(self.save_path, "stationarity_analysis")
            os.makedirs(analysis_dir, exist_ok=True)
            
            # Format data and check stationarity
            formatted_data, format_results = self.remove_non_stationarity(data)
            stationarity_results = format_results['stationarity_check']
            
            # Calculate ACF and PACF on the formatted (and possibly differenced) data
            acf_values, pacf_values = self._calculate_acf_pacf(formatted_data)
            
            # Get seasonal period
            seasonal_period = format_results['seasonal_period'] or self._infer_seasonal_period(data)
            
            # Suggest parameters based on ACF/PACF analysis
            params = self._suggest_order_parameters(acf_values, pacf_values, seasonal_period)
            
            # Use the differencing parameters from format_results
            params['d'] = format_results['differencing_params']['d']
            params['D'] = format_results['differencing_params']['D']
            params['s'] = seasonal_period
            
            results = {
                'suggested_parameters': params,
                'analysis_results': stationarity_results,
                'acf_values': acf_values,
                'pacf_values': pacf_values
            }
            
            logger.info(f"Parameter suggestion completed: {params}")
            return results
            
        except Exception as e:
            logger.error(f"Error in suggest_model_parameters: {str(e)}")
            raise 