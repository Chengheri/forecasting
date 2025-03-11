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
from backend.app.data.data_loader import DataLoader

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

def convert_to_native_types(obj: Any) -> Any:
    """Convert numpy types to native Python types."""
    if isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32,
                       np.int64, np.uint8, np.uint16, np.uint32, np.uint64)):
        return int(obj)
    elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, dict):
        return {str(key): convert_to_native_types(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_native_types(item) for item in obj]
    return obj

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
        self.use_grid_search = config['model'].get('use_grid_search', False)
        self.optimize_hyperparameters = config['model'].get('optimize_hyperparameters', False)
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
        
        logger.info(f"Loading data from {self.config['data']['path']}")
        data = data_loader.load_csv()
        logger.info(f"Successfully loaded data with shape: {data.shape}")
        return data
    
    @handle_pipeline_errors
    def prepare_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare data using DataPreprocessor based on configuration.
        
        Args:
            data: Raw input data
            
        Returns:
            pd.DataFrame: Prepared data
        """
        logger.info("Starting data preparation")
        prepared_data = self.preprocessor.preprocess_data(
            data, 
            target_column=self.config['data']['target_column']
        )
        logger.info("Data preparation completed successfully")
        return prepared_data
    
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
            # Split data using preprocessor's method
            train_data, test_data = self.preprocessor.train_test_split_timeseries(
                data,
                train_ratio=self.config['preprocessing']['train_ratio'],
                validation_ratio=self.config['preprocessing'].get('validation_ratio', 0.0),
                target_column=self.config['data']['target_column'],
                date_column=data.index.name,
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
    def train_model(self, train_data: pd.DataFrame, test_data: pd.DataFrame,
                   suggested_params: Dict[str, Any], stationarity_results: Dict[str, Any],
                   preprocessor: DataPreprocessor) -> Tuple[TimeSeriesModel, Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
        """Train the ARIMA/SARIMA model with the specified configuration."""
        # Initialize model with data
        filtered_config = filter_model_params(self.config['model'], self.model_type)
        filtered_config['model_type'] = self.model_type
        
        model = TimeSeriesModel(
            config=filtered_config,
            tracker=self.tracker,
            data=train_data[self.config['data']['target_column']].values.ravel()
        )
        
        if self.use_grid_search:
            # Train with grid search
            logger.info("Starting grid search")
            grid_search_results = model.grid_search(
                data=train_data[self.config['data']['target_column']].values.ravel(),
                param_grid=self.config['grid_search']['param_grid'],
                max_iterations=self.config['grid_search'].get('max_iterations', 50),
                early_stopping=self.config['grid_search'].get('early_stopping', True),
                early_stopping_patience=self.config['grid_search'].get('early_stopping_patience', 5),
                timeout=self.config['grid_search'].get('timeout', 60)
            )
            
            model = model.get_best_model()
            model_params = grid_search_results['best_params']
            metrics = {
                'aic': grid_search_results['best_aic'],
                'bic': grid_search_results['best_bic'],
                'hqic': grid_search_results['best_hqic']
            }
            
            logger.info(f"Grid search completed. Best parameters: {model_params}")
            logger.info(f"Best AIC: {metrics['aic']}")
            
        else:
            # Train with suggested parameters
            logger.info(f"Training {self.model_type.upper()} model with suggested parameters: {suggested_params}")
            
            # Update model configuration with suggested parameters
            filtered_suggested = filter_model_params(suggested_params, self.model_type)
            model.config.update(filtered_suggested)
            
            # Train model
            logger.info("Starting model training")
            metrics = model.fit()
            model_params = suggested_params
            grid_search_results = None
        
        # Optimize non-structural hyperparameters if enabled
        if self.optimize_hyperparameters:
            logger.info("Starting hyperparameter optimization with Optuna")
            metrics = model.fit_with_optuna(
                n_trials=self.config['model']['optimization'].get('n_trials', 100),
                timeout=self.config['model']['optimization'].get('timeout', 600)
            )
            logger.info(f"Hyperparameter optimization completed. Best parameters: {model.config}")
        
        return model, model_params, metrics, grid_search_results
    
    @handle_pipeline_errors
    def test_model(self, model: Union[TimeSeriesModel, SARIMAX], test_data: pd.DataFrame) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, np.ndarray]]]:
        """Generate predictions for test data.
        
        Args:
            model: Trained model instance
            test_data: Test dataset
            
        Returns:
            Tuple containing predictions and confidence intervals
        """
        logger.info(f"Generating predictions for test data with {len(test_data)} steps")
        
        if isinstance(model, TimeSeriesModel):
            mean_forecast, confidence_intervals = model.predict(len(test_data))
            if isinstance(confidence_intervals, tuple) and len(confidence_intervals) == 2:
                lower_ci, upper_ci = confidence_intervals
            else:
                logger.warning("Unexpected confidence intervals format from TimeSeriesModel. Using None.")
                confidence_intervals = None
        else:
            forecast = model.get_forecast(steps=len(test_data))
            mean_forecast = forecast.predicted_mean
            conf_int = forecast.conf_int()
            try:
                lower_ci = conf_int.iloc[:, 0].values
                upper_ci = conf_int.iloc[:, 1].values
                confidence_intervals = (lower_ci, upper_ci)
                logger.info(f"Confidence intervals shape: lower {lower_ci.shape}, upper {upper_ci.shape}")
            except Exception as e:
                logger.warning(f"Failed to process confidence intervals: {str(e)}. Using None.")
                confidence_intervals = None
        
        return mean_forecast, confidence_intervals
    
    @handle_pipeline_errors
    def track_results(self, model_params: Dict[str, Any], metrics: Dict[str, Any], 
                     analysis_results: Dict[str, Any], stationarity_results: Dict[str, Any],
                     analysis_path: str, model_file_path: str) -> None:
        """Track model results using MLflow."""
        try:
            # Log model parameters
            model_params_with_prefix = {
                'model.arima.p': model_params['p'],
                'model.arima.d': model_params['d'],
                'model.arima.q': model_params['q'],
                'model.arima.P': model_params['P'],
                'model.arima.D': model_params['D'],
                'model.arima.Q': model_params['Q'],
                'model.arima.s': model_params['s'],
                'model.arima.trend': model_params.get('trend', 'c'),
                'model.arima.enforce_stationarity': model_params.get('enforce_stationarity', True),
                'model.arima.enforce_invertibility': model_params.get('enforce_invertibility', True),
                'model.arima.concentrate_scale': model_params.get('concentrate_scale', False),
                'model.arima.method': model_params.get('method', 'lbfgs'),
                'model.arima.maxiter': model_params.get('maxiter', 50)
            }
            self.tracker.log_params_safely(model_params_with_prefix)
            
            # Log training metrics
            training_metrics = {
                'metrics.training.aic': metrics['aic'],
                'metrics.training.bic': metrics.get('bic', 0),
                'metrics.training.hqic': metrics.get('hqic', 0),
                'metrics.training.rmse': analysis_results['rmse'],
                'metrics.training.mae': analysis_results['mae'],
                'metrics.training.mape': analysis_results['mape'],
                'metrics.training.r2': analysis_results.get('r2', 0),
                'metrics.training.directional_accuracy': analysis_results.get('directional_accuracy', 0)
            }
            self.tracker.log_metrics_safely(training_metrics)
            
            # Log model diagnostics
            diagnostic_metrics = {
                'metrics.diagnostics.residuals_mean': analysis_results.get('residuals_mean', 0),
                'metrics.diagnostics.residuals_std': analysis_results.get('residuals_std', 0),
                'metrics.diagnostics.residuals_skewness': analysis_results.get('residuals_skewness', 0),
                'metrics.diagnostics.residuals_kurtosis': analysis_results.get('residuals_kurtosis', 0),
                'metrics.diagnostics.residuals_autocorrelation': analysis_results.get('residuals_autocorrelation', 0),
                'metrics.diagnostics.residuals_normal': analysis_results.get('residuals_normal', False),
                'metrics.diagnostics.residuals_independent': analysis_results.get('residuals_independent', False)
            }
            self.tracker.log_metrics_safely(diagnostic_metrics)
            
            # Log forecast metrics
            forecast_metrics = {
                'metrics.forecast.rmse': analysis_results['rmse'],
                'metrics.forecast.mae': analysis_results['mae'],
                'metrics.forecast.mape': analysis_results['mape'],
                'metrics.forecast.r2': analysis_results.get('r2', 0)
            }
            self.tracker.log_metrics_safely(forecast_metrics)
            
            # Log stationarity analysis
            stationarity_metrics = {
                'stationarity.adf_test.statistic': stationarity_results['adf_test']['test_statistic'],
                'stationarity.adf_test.pvalue': stationarity_results['adf_test']['pvalue'],
                'stationarity.adf_test.is_stationary': stationarity_results['adf_test']['is_stationary'],
                'stationarity.kpss_test.statistic': stationarity_results['kpss_test']['test_statistic'],
                'stationarity.kpss_test.pvalue': stationarity_results['kpss_test']['pvalue'],
                'stationarity.kpss_test.is_stationary': stationarity_results['kpss_test']['is_stationary'],
                'stationarity.overall.is_stationary': stationarity_results['overall_assessment']['is_stationary']
            }
            self.tracker.log_metrics_safely(stationarity_metrics)
            
            # Log stationarity parameters
            stationarity_params = {
                'stationarity.adf_test.critical_values': stationarity_results['adf_test']['critical_values'],
                'stationarity.kpss_test.critical_values': stationarity_results['kpss_test']['critical_values'],
                'stationarity.overall.confidence': stationarity_results['overall_assessment']['confidence'],
                'stationarity.overall.recommendation': stationarity_results['overall_assessment']['recommendation']
            }
            self.tracker.log_params_safely(stationarity_params)
            
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
        """Create a comprehensive model summary with consistent structure."""
        model_summary = {
            'initial_configuration': convert_to_native_types(self.config),
            'training_details': {
                'train_samples': int(len(train_data)),
                'test_samples': int(len(test_data)),
                'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'parameter_selection_method': 'grid_search' if self.use_grid_search else 'suggested_parameters'
            },
            'best_model': {
                'parameters': convert_to_native_types(model_params),
                'performance': {
                    'aic': float(metrics.get('aic', 0)),
                    'bic': float(metrics.get('bic', 0)),
                    'hqic': float(metrics.get('hqic', 0)),
                    'metrics': convert_to_native_types(analysis_results),
                    'residuals_analysis': {
                        'mean': float(analysis_results.get('residuals_mean', 0)),
                        'std': float(analysis_results.get('residuals_std', 0)),
                        'skewness': float(analysis_results.get('residuals_skewness', 0)),
                        'kurtosis': float(analysis_results.get('residuals_kurtosis', 0)),
                        'autocorrelation': float(analysis_results.get('residuals_autocorrelation', 0)),
                        'normal_distribution': bool(analysis_results.get('residuals_normal', False)),
                        'independent': bool(analysis_results.get('residuals_independent', False))
                    }
                },
                'artifacts': {
                    'model_path': model_path,
                    'analysis_path': analysis_path,
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
        """Analyze model results using the Analyzer."""
        logger.info("Analyzing model results")
        
        # Initialize analyzer with analysis path
        analyzer = Analyzer(config=self.config, save_path=analysis_path)
        
        # Get target values from test data
        target_values = test_data[self.config['data']['target_column']].values if isinstance(
            test_data[self.config['data']['target_column']], pd.Series
        ) else test_data[self.config['data']['target_column']]
        
        # Analyze results
        analysis_results = analyzer.analyze_model_results(
            target_values,
            mean_forecast,
            test_data.index,
            confidence_intervals
        )
        
        logger.info("Model results analysis completed successfully")
        return analysis_results
    
    @handle_pipeline_errors
    def get_model_parameters(self, data: pd.DataFrame) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Get suggested model parameters and adjust them based on model type."""
        logger.info("Getting suggested model parameters")
        
        # Get suggested parameters from analyzer
        suggested_params, stationarity_results = self.analyzer.get_suggested_parameters(
            data[self.config['data']['target_column']], 
            self.config
        )
        
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