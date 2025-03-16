import os
from datetime import datetime
from typing import Dict, Any, Tuple, Optional, Union, Callable, TypeVar
from functools import wraps

import numpy as np
import pandas as pd
import mlflow
import joblib
from statsmodels.tsa.statespace.sarimax import SARIMAX

from backend.app.models.arima_model import TimeSeriesModel
from backend.app.utils.trackers import ARIMATracker
from backend.app.utils.preprocessing import DataPreprocessor
from backend.app.utils.logger import Logger
from backend.app.utils.analyzer import Analyzer
from backend.app.utils.data_loader import DataLoader, convert_to_native_types

logger = Logger()

T = TypeVar('T')

def handle_pipeline_errors(func: Callable[..., T]) -> Callable[..., T]:
    """Decorator to handle common error patterns in pipeline methods."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Failed in {func.__name__}: {str(e)}")
            raise
    return wrapper

def filter_model_params(params: Dict[str, Any], model_type: str) -> Dict[str, Any]:
    """Filter model parameters based on model type."""
    valid_model_params = {'p', 'd', 'q', 'method', 'trend',
                         'enforce_stationarity', 'enforce_invertibility', 'concentrate_scale'}
    if model_type == 'sarima':
        valid_model_params.update({'P', 'D', 'Q', 's', 'maxiter'})
    return {k: v for k, v in params.items() if k in valid_model_params}

class SARIMAPipeline:
    """Pipeline for training and evaluating SARIMA models."""
    
    def __init__(self, config: Dict[str, Any], tracker: ARIMATracker):
        """Initialize the SARIMA pipeline.
        
        Args:
            config: Configuration dictionary
            tracker: MLflow tracker instance
        """
        self.config = config
        self.tracker = tracker
        self.model_type = config['model'].get('model_type', 'sarima').lower()
        self.analyzer = Analyzer(config=config)
        self.preprocessor = DataPreprocessor(config=config['preprocessing'], tracker=tracker)
    
    @handle_pipeline_errors
    def load_data(self) -> pd.DataFrame:
        """Load data using DataLoader based on configuration.
        
        Returns:
            pd.DataFrame: Loaded data
        """
        logger.info("Initializing DataLoader")
        data_loader = DataLoader(config=self.config)
        
        data = data_loader.load_csv()
        return data
    
    @handle_pipeline_errors
    def prepare_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare data using DataPreprocessor based on configuration.
        
        Args:
            data: Raw input data
            
        Returns:
            pd.DataFrame: Prepared data
        """
        prepared_data = self.preprocessor.preprocess_data(
            data, 
            target_column=self.config['data']['target_column']
        )
        return prepared_data
    
    @handle_pipeline_errors
    def remove_non_stationarity(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Remove non-stationarity from time series data based on configuration.
        
        Args:
            data: Input time series data
            
        Returns:
            Tuple containing:
            - Processed time series data as pd.DataFrame
            - Dictionary with stationarity analysis results and differencing parameters
            
        The method uses the pipeline's configuration to determine:
        - Whether to apply seasonal differencing (based on model_type)
        - The seasonal period to use (from config['model']['s'])
        - Maximum differencing order (from config or default)
        """
        logger.info("Removing non-stationarity from time series data")
        
        try:
            # Create a copy of the input DataFrame
            processed_data = data.copy()
            
            # Extract target series
            target_col = self.config['data']['target_column']
            ts = data[target_col]

            if self.config['preprocessing'].get('remove_non_stationarity', False): 
                # Use analyzer to handle non-stationarity
                processed_ts, _ = self.analyzer.remove_non_stationarity(ts)
                logger.info(f"Starting stationarity analysis after differencing...")
                results = self.analyzer.check_stationarity(processed_ts)
                # Update the target column with processed values
                processed_data[target_col] = processed_ts
                logger.info("Non-stationarity removal completed")
            else:
                logger.info("Non-stationarity removal not enabled")             
                results = {}
                
            return processed_data, results
            
        except Exception as e:
            logger.error(f"Error removing non-stationarity: {str(e)}")
            raise
    
    def split_data(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Split data into training and test sets.
        
        Args:
            data: Input data to split
            
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: Training and test datasets
            
        Raises:
            Exception: If data splitting fails
        """
        logger.info(f"Splitting time series data with train_ratio={self.config['preprocessing']['train_ratio']}, "
                   f"validation_ratio={self.config['preprocessing'].get('validation_ratio', 0.0)}, "
                   f"gap={self.config['preprocessing'].get('gap', 0)}")
        
        try:
            # Convert data to DataFrame if it's a Series
            if isinstance(data, pd.Series):
                data = data.to_frame()
            
            # Get date column name, ensuring it's a string or None
            date_column = str(data.index.name) if data.index.name else None
            
            # Split data using preprocessor's method
            train_data, test_data = self.preprocessor.train_test_split_timeseries(
                data,
                train_ratio=self.config['preprocessing']['train_ratio'],
                validation_ratio=self.config['preprocessing'].get('validation_ratio', 0.0),
                target_column=self.config['data']['target_column'],
                date_column=date_column,
                gap=self.config['preprocessing'].get('gap', 0)
            )
            
            # Log data split information
            logger.info(f"Split data into {len(train_data)} training samples, 0 validation samples, and {len(test_data)} test samples")
            logger.info(f"Training data date range: {train_data.index[0]} to {train_data.index[-1]}")
            logger.info(f"Test data date range: {test_data.index[0]} to {test_data.index[-1]}")
            
            # Log data statistics
            target_col = self.config['data']['target_column']
            logger.info(f"Training target stats - Mean: {train_data[target_col].mean():.4f}, "
                       f"Std: {train_data[target_col].std():.4f}, "
                       f"Min: {train_data[target_col].min():.4f}, "
                       f"Max: {train_data[target_col].max():.4f}")
            logger.info(f"Test target stats - Mean: {test_data[target_col].mean():.4f}, "
                       f"Std: {test_data[target_col].std():.4f}, "
                       f"Min: {test_data[target_col].min():.4f}, "
                       f"Max: {test_data[target_col].max():.4f}")
            
            return train_data, test_data
            
        except Exception as e:
            logger.error(f"Failed to split data: {str(e)}")
            raise
    
    @handle_pipeline_errors
    def train_model(self, train_data: pd.DataFrame, 
                   suggested_params: Dict[str, Any],
        ) -> Tuple[TimeSeriesModel, Dict[str, Any], Dict[str, Any], Optional[Dict[str, Any]]]:
        """Train the ARIMA model.
        
        Args:
            train_data: Training dataset
            test_data: Test dataset
            suggested_params: Model parameters suggested by analyzer
            stationarity_results: Results from stationarity analysis
            preprocessor: Data preprocessor instance
            
        Returns:
            Tuple containing:
                - Trained model instance
                - Model parameters used
                - Training metrics
                - Grid search results (if performed)
        """        
        # Extract training data as Series
        train_series = train_data[self.config['data']['target_column']]
        if isinstance(train_series, pd.DataFrame):
            train_series = train_series.squeeze()
        
        # Initialize model
        self.config['model'].update(suggested_params)
        model = TimeSeriesModel(
            config=self.config,
            tracker=self.tracker,
            data=train_series
        )
        
        if self.config['model'].get('use_grid_search'):
            # Perform grid search
            grid_search_results = model.grid_search(
                data=train_series,
                param_grid=self.config['grid_search']['param_grid'],
                max_iterations=self.config['grid_search'].get('max_iterations', 50),
                early_stopping=self.config['grid_search'].get('early_stopping', True),
                cv=self.config['grid_search'].get('cv', 3)
            )
            model.fitted_model = grid_search_results['best_model']
            model_params = grid_search_results['best_params']
            metrics = grid_search_results['best_metrics']
        elif self.config['model'].get('optimize_hyperparameters'):
            # Perform hyperparameter optimization
            hyperparameter_optimization_results = model.fit_with_optuna(
                data=train_series,
                n_trials=self.config['model']['optimization']['n_trials'],
                timeout=self.config['model']['optimization']['timeout']
            )
            model_params = suggested_params.update(hyperparameter_optimization_results['best_params'])
            metrics = hyperparameter_optimization_results['best_metrics']
            grid_search_results = None
        else:
            # Train model with suggested parameters
            logger.info("Starting model training with suggested parameters...")
            metrics = model.fit(data=train_series)
            model_params = suggested_params
            grid_search_results = None
        
        return model, model_params, metrics, grid_search_results
    
    @handle_pipeline_errors
    def test_model(self, model: Union[TimeSeriesModel, SARIMAX], test_data: pd.DataFrame) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, np.ndarray]]]:
        """Generate predictions for test data.
        
        Args:
            model: Trained model instance
            test_data: Test dataset to generate predictions for
            
        Returns:
            Tuple containing:
                - Mean forecast as numpy array
                - Optional tuple of confidence intervals (lower, upper) as numpy arrays
        """
        logger.info(f"Generating predictions for test data with {len(test_data)} steps")
        
        # Extract test data as Series if needed
        test_series = test_data[self.config['data']['target_column']]
        if isinstance(test_series, pd.DataFrame):
            test_series = test_series.squeeze()
            
        # Generate predictions
        mean_forecast, confidence_intervals = model.predict(test_series)
        
        # Convert predictions to numpy arrays if they aren't already
        mean_forecast = mean_forecast.to_numpy() if isinstance(mean_forecast, (pd.Series, pd.DataFrame)) else mean_forecast
        
        if confidence_intervals is not None:
            lower = confidence_intervals[0].to_numpy() if isinstance(confidence_intervals[0], (pd.Series, pd.DataFrame)) else confidence_intervals[0]
            upper = confidence_intervals[1].to_numpy() if isinstance(confidence_intervals[1], (pd.Series, pd.DataFrame)) else confidence_intervals[1]
            confidence_intervals = (lower, upper)
        
        return mean_forecast, confidence_intervals
    
    @handle_pipeline_errors
    def track_results(self, model_params: Dict[str, Any], metrics: Dict[str, Any], 
                     analysis_results: Dict[str, Any], stationarity_results: Dict[str, Any],
                     stationarity_results_after_differencing: Dict[str, Any],
                     analysis_path: str, model_file_path: str) -> None:
        """Track model results using MLflow."""
        try:
            # Log model parameters
            model_params_with_prefix = {
                'model.arima.p': model_params['p'],
                'model.arima.d': model_params['d'],
                'model.arima.q': model_params['q'],
                'model.arima.seasonal_p': model_params['P'],
                'model.arima.seasonal_d': model_params['D'],
                'model.arima.seasonal_q': model_params['Q'],
                'model.arima.seasonal_s': model_params['s'],
                'model.arima.trend': model_params.get('trend', self.config['model']['trend']),
                'model.arima.enforce_stationarity': model_params.get('enforce_stationarity', self.config['model']['enforce_stationarity']),
                'model.arima.enforce_invertibility': model_params.get('enforce_invertibility', self.config['model']['enforce_invertibility']),
                'model.arima.concentrate_scale': model_params.get('concentrate_scale', self.config['model']['concentrate_scale']),
                'model.arima.method': model_params.get('method', self.config['model']['method']),
                'model.arima.maxiter': model_params.get('maxiter', self.config['model']['maxiter'])
            }
            self.tracker.log_params_safely(model_params_with_prefix)
            
            # Log training metrics
            training_metrics = {
                'metrics.training.aic': metrics['aic'],
                'metrics.training.bic': metrics.get('bic'),
                'metrics.training.hqic': metrics.get('hqic'),
            }
            self.tracker.log_metrics_safely(training_metrics)
            # Log test metrics
            test_metrics = {
                'metrics.test.rmse': analysis_results['rmse'],
                'metrics.test.mae': analysis_results['mae'],
                'metrics.test.mape': analysis_results['mape'],
                'metrics.test.r2': analysis_results.get('r2'),
                'metrics.test.directional_accuracy': analysis_results.get('directional_accuracy'),
                'metrics.test.residuals_mean': analysis_results.get('residuals_mean'),
                'metrics.test.residuals_std': analysis_results.get('residuals_std'),
                'metrics.test.residuals_skewness': analysis_results.get('residuals_skewness'),
                'metrics.test.residuals_kurtosis': analysis_results.get('residuals_kurtosis'),
                'metrics.test.residuals_autocorrelation': analysis_results.get('residuals_autocorrelation'),
                'metrics.test.residuals_normal': analysis_results.get('residuals_normal'),
                'metrics.test.residuals_independent': analysis_results.get('residuals_independent')
            }
            self.tracker.log_metrics_safely(test_metrics)
            
            # Log stationarity analysis
            stationarity_metrics = {
                'stationarity.adf_test.statistic': stationarity_results['adf_test']['test_statistic'],
                'stationarity.adf_test.pvalue': stationarity_results['adf_test']['pvalue'],
                'stationarity.adf_test.is_stationary': stationarity_results['adf_test']['is_stationary'],
                'stationarity.adf_test.critical_values': stationarity_results['adf_test']['critical_values'],
                'stationarity.kpss_test.statistic': stationarity_results['kpss_test']['test_statistic'],
                'stationarity.kpss_test.pvalue': stationarity_results['kpss_test']['pvalue'],
                'stationarity.kpss_test.is_stationary': stationarity_results['kpss_test']['is_stationary'],
                'stationarity.kpss_test.critical_values': stationarity_results['kpss_test']['critical_values'],
                'stationarity.overall.is_stationary': stationarity_results['overall_assessment']['is_stationary'],
                'stationarity.overall.confidence': stationarity_results['overall_assessment']['confidence'],
                'stationarity.overall.recommendation': stationarity_results['overall_assessment']['recommendation']
            }
            self.tracker.log_params_safely(stationarity_metrics)
            
            if stationarity_results_after_differencing:
                # Log stationarity results after differencing
                stationarity_metrics_after_differencing = {
                    'stationarity.after_differencing.adf_test.statistic': stationarity_results_after_differencing['adf_test']['test_statistic'],
                    'stationarity.after_differencing.adf_test.pvalue': stationarity_results_after_differencing['adf_test']['pvalue'],
                    'stationarity.after_differencing.adf_test.is_stationary': stationarity_results_after_differencing['adf_test']['is_stationary'],
                    'stationarity.after_differencing.kpss_test.statistic': stationarity_results_after_differencing['kpss_test']['test_statistic'],
                    'stationarity.after_differencing.kpss_test.pvalue': stationarity_results_after_differencing['kpss_test']['pvalue'],
                    'stationarity.after_differencing.kpss_test.is_stationary': stationarity_results_after_differencing['kpss_test']['is_stationary'],
                    'stationarity.after_differencing.overall.is_stationary': stationarity_results_after_differencing['overall_assessment']['is_stationary'],
                    'stationarity.after_differencing.overall.confidence': stationarity_results_after_differencing['overall_assessment']['confidence'],
                    'stationarity.after_differencing.overall.recommendation': stationarity_results_after_differencing['overall_assessment']['recommendation']
                }
                self.tracker.log_params_safely(stationarity_metrics_after_differencing)
            # Log artifacts
            mlflow.log_artifacts(analysis_path, "analysis")
            mlflow.log_artifact(model_file_path, "model")
            
        except Exception as e:
            logger.warning(f"Failed to log results to MLflow: {str(e)}. Continuing without MLflow tracking.")
    
    @handle_pipeline_errors
    def create_model_summary(self, train_data: pd.DataFrame, test_data: pd.DataFrame,
                           model_params: Dict[str, Any], metrics: Dict[str, Any], 
                           analysis_results: Dict[str, Any], stationarity_results: Dict[str, Any],
                           model_path: str, analysis_path: str, grid_search_results: Optional[Dict] = None,
                           preprocessor: Optional[DataPreprocessor] = None) -> Dict[str, Any]:
        """Create a comprehensive model summary with consistent structure.
        
        Args:
            train_data: Training dataset
            test_data: Test dataset
            model_params: Model parameters used
            metrics: Training metrics
            analysis_results: Results from model analysis
            stationarity_results: Results from stationarity analysis
            model_path: Path where model is saved
            analysis_path: Path where analysis results are saved
            grid_search_results: Optional grid search results
            preprocessor: Optional data preprocessor instance
            
        Returns:
            Dict[str, Any]: Model summary with all values converted to JSON-serializable types
        """
        # Convert all numeric values to native Python types
        model_summary = {
            'initial_configuration': convert_to_native_types(self.config),
            'training_details': {
                'train_samples': int(len(train_data)),
                'test_samples': int(len(test_data)),
                'training_date': datetime.now().isoformat(),
                'parameter_selection_method': 'grid_search' if self.config['model'].get('use_grid_search') else 'suggested_parameters'
            },
            'best_model': {
                'parameters': convert_to_native_types(model_params),
                'performance': {
                    'aic': float(metrics.get('aic')),
                    'bic': float(metrics.get('bic')),
                    'hqic': float(metrics.get('hqic')),
                    'metrics': convert_to_native_types(analysis_results),
                    'residuals_analysis': {
                        'mean': float(analysis_results.get('residuals_mean')),
                        'std': float(analysis_results.get('residuals_std')),
                        'skewness': float(analysis_results.get('residuals_skewness')),
                        'kurtosis': float(analysis_results.get('residuals_kurtosis')),
                        'autocorrelation': float(analysis_results.get('residuals_autocorrelation')),
                        'normal_distribution': bool(analysis_results.get('residuals_normal')),
                        'independent': bool(analysis_results.get('residuals_independent'))
                    }
                },
                'artifacts': {
                    'model_path': str(model_path),
                    'analysis_path': str(analysis_path),
                    'plots': [
                        'actual_vs_predicted.png',
                        'residuals_analysis.png',
                        'metrics_over_time.png',
                        'seasonal_decomposition.png',
                        'acf_pacf_analysis.png'
                    ]
                }
            },
            'stationarity_analysis': convert_to_native_types(stationarity_results)
        }
        
        # Add grid search details if available
        if grid_search_results:
            model_summary['training_details'].update({
                'grid_search_space': convert_to_native_types(grid_search_results.get('param_grid', {})),
                'models_evaluated': int(grid_search_results.get('total_combinations_tested', 0)),
                'convergence_rate': float(grid_search_results.get('convergence_rate', 0)),
                'all_results': convert_to_native_types(grid_search_results.get('all_results', []))
            })
        
        # Add preprocessing pipeline if available
        if preprocessor:
            model_summary['preprocessing_pipeline'] = convert_to_native_types(preprocessor.pipeline_steps)
        
        return model_summary
    
    @handle_pipeline_errors
    def analyze_results(self, test_data: pd.DataFrame, mean_forecast: np.ndarray, 
                      confidence_intervals: Optional[Tuple[np.ndarray, np.ndarray]], 
                      analysis_path: str) -> Dict[str, Any]:
        """Analyze model results using the Analyzer.
        
        Args:
            test_data: Test dataset
            mean_forecast: Mean forecast values
            confidence_intervals: Optional tuple of lower and upper confidence bounds
            analysis_path: Path to save analysis results
            
        Returns:
            Dict[str, Any]: Analysis results
        """
        logger.info("Analyzing model results")
        
        # Get target values from test data and ensure numpy array format
        target_values = test_data[self.config['data']['target_column']]
        if isinstance(target_values, pd.Series):
            target_values = target_values.to_numpy()
        
        # Convert timestamps to numpy array if needed
        timestamps = test_data.index
        if isinstance(timestamps, pd.Index):
            timestamps = timestamps.to_numpy()
        
        # Analyze results
        analysis_results = self.analyzer.analyze_model_results(
            target_values,
            mean_forecast,
            timestamps,
            confidence_intervals,
            analysis_path
        )
        
        # Convert results to native types for JSON serialization
        analysis_results = convert_to_native_types(analysis_results)
        
        logger.info("Model results analysis completed successfully")
        return analysis_results
    
    @handle_pipeline_errors
    def get_model_parameters(self, data: pd.DataFrame) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Get suggested model parameters and adjust them based on model type."""        
        # Get suggested parameters from analyzer
        results = self.analyzer.suggest_model_parameters(
            data[self.config['data']['target_column']], 
            self.config
        )
        suggested_params = results['suggested_parameters']
        stationarity_results = results['analysis_results']
        
        # Adjust suggested parameters based on model type
        if self.model_type == 'arima':
            logger.info("Adjusting parameters for ARIMA model (removing seasonal components)")
            # Remove seasonal parameters for ARIMA
            suggested_params = filter_model_params(suggested_params, self.model_type)
            if 'grid_search' in self.config:
                self.config['grid_search']['param_grid'] = filter_model_params(
                    self.config['grid_search']['param_grid'],
                    self.model_type
                )
        
        logger.info(f"Suggested parameters: {suggested_params}")
        return suggested_params, stationarity_results 