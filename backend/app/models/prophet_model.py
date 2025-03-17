from typing import Dict, Any, Optional, Tuple, List, Union
import pandas as pd
import numpy as np
from prophet import Prophet
import joblib
import warnings
from ..utils.logger import Logger
from ..utils.trackers import ProphetTracker
import optuna
import tempfile
import os
import uuid
from pandas.tseries.frequencies import infer_freq

warnings.filterwarnings('ignore')
logger = Logger()

class ProphetModel:
    """Prophet time series model with unified tracking."""
    
    def __init__(
        self,
        config: Dict[str, Any],
        tracker: Optional[ProphetTracker] = None,    
    ):
        """Initialize ProphetModel with parameters and optionally with data."""
        logger.debug(f"Configuration provided: {config}")

        self.config = config
        self.tracker = tracker
        self.model: Optional[Prophet] = None
        self.fitted_model: Optional[Prophet] = None
        self._training_data: Optional[pd.DataFrame] = None

        # Initialize model with data if provided
        self._log_initialization()
    
    def _log_initialization(self) -> None:
        """Log initialization parameters."""
        logger.info(f"Initialized {self.config['model']['model_type'].title()} model...")
        
        if self.tracker:
            logger.debug("Logging model parameters to tracker")
            self.tracker.log_params_safely({
                'model.type': self.config['model']['model_type']
            })
    
    def prepare_data(self, data: Union[pd.DataFrame, pd.Series, np.ndarray]) -> pd.DataFrame:
        """Prepare data for Prophet model."""
        if isinstance(data, pd.Series):
            df = pd.DataFrame({'ds': data.index, 'y': data.values})
        elif isinstance(data, pd.DataFrame):
            df = pd.DataFrame({'ds': data.index, 'y': data[self.config['data']['target_column']]})
        else:
            raise ValueError("Data must be pandas Series or DataFrame with datetime index")
        return df
    
    def initialize_model(self) -> None:
        """Initialize the Prophet model with data."""
        logger.info("Initializing model...")
        
        try:
            # Create Prophet model with configuration parameters
            self.model = Prophet(
                changepoint_prior_scale=self.config['model'].get('changepoint_prior_scale', 0.05),
                seasonality_prior_scale=self.config['model'].get('seasonality_prior_scale', 10.0),
                holidays_prior_scale=self.config['model'].get('holidays_prior_scale', 10.0),
                seasonality_mode=self.config['model'].get('seasonality_mode', 'additive'),
                changepoint_range=self.config['model'].get('changepoint_range', 0.8),
                growth=self.config['model'].get('growth', 'linear'),
                n_changepoints=self.config['model'].get('n_changepoints', 25),
                yearly_seasonality=self.config['model'].get('yearly_seasonality', 'auto'),
                weekly_seasonality=self.config['model'].get('weekly_seasonality', 'auto'),
                daily_seasonality=self.config['model'].get('daily_seasonality', 'auto')
            )
            
            logger.info("Prophet model initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing Prophet model: {str(e)}")
            raise
    
    @property
    def data(self) -> Optional[pd.DataFrame]:
        """Get the stored training data."""
        return self._training_data
    
    def fit(self, data: Union[pd.Series, pd.DataFrame, np.ndarray], params: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
        """Fit the Prophet model to the provided data."""
        try:
            # Update parameters if provided
            if params:
                self.config['model'].update(params)
            self.initialize_model()
            
            if self.model is None:
                raise ValueError("Model must be initialized before fitting")
            
            # Prepare data if needed
            if isinstance(data, (pd.Series, np.ndarray)) or (isinstance(data, pd.DataFrame) and ('y' not in data.columns or 'ds' not in data.columns)):
                prepared_data = self.prepare_data(data)
            else:
                prepared_data = data
            
            # Store training data for later use
            self._training_data = prepared_data.copy() if isinstance(prepared_data, pd.DataFrame) else prepared_data
            
            # Fit model
            logger.info("Fitting Prophet model")
            self.fitted_model = self.model.fit(prepared_data)
            
            # Calculate metrics on training data
            train_metrics = self._calculate_metrics(prepared_data)
            
            # Convert numpy types to Python native types
            train_metrics = {k: float(v) for k, v in train_metrics.items()}
            
            logger.info("Model training completed successfully")
            logger.info(f"Training metrics: {train_metrics}")
            
            return train_metrics
            
        except Exception as e:
            logger.error(f"Error in model training: {str(e)}")
            raise
    
    def predict(self, steps: int) -> Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Generate predictions using the fitted Prophet model."""
        if self.fitted_model is None:
            raise ValueError("Model must be trained before prediction")
        
        logger.info(f"Generating predictions for {steps} steps")
        
        try:
            # Create future dataframe
            future = self.fitted_model.make_future_dataframe(periods=steps, freq=self._infer_frequency())
            
            # Generate forecast
            forecast = self.fitted_model.predict(future)
            
            # Extract predictions and confidence intervals, ensuring numpy array types
            predictions = np.asarray(forecast['yhat'].values[-steps:], dtype=np.float64)
            lower_bound = np.asarray(forecast['yhat_lower'].values[-steps:], dtype=np.float64)
            upper_bound = np.asarray(forecast['yhat_upper'].values[-steps:], dtype=np.float64)
            
            logger.info("Predictions generated successfully")
            return predictions, (lower_bound, upper_bound)
            
        except Exception as e:
            logger.error(f"Error generating predictions: {str(e)}")
            raise
    
    def _infer_frequency(self) -> str:
        """Infer the frequency of the time series data."""
        if self._training_data is None:
            raise ValueError("No training data available")
        
        # Convert to DatetimeIndex for proper frequency inference
        dates = pd.DatetimeIndex(self._training_data['ds'])
        freq = pd.infer_freq(dates)
        return freq if freq is not None else 'D'
    
    def _calculate_metrics(self, data: pd.DataFrame) -> Dict[str, float]:
        """Calculate training metrics."""
        if self.fitted_model is None:
            raise ValueError("Model must be trained before calculating metrics")
            
        # Use the same data length for predictions
        forecast = self.fitted_model.predict(data)
        
        # Convert to numpy arrays with explicit type
        actual = np.asarray(data['y'].values, dtype=np.float64)
        predicted = np.asarray(forecast['yhat'].values, dtype=np.float64)
        
        # Avoid division by zero in MAPE calculation
        non_zero_mask = np.asarray(actual != 0, dtype=bool)
        if bool(non_zero_mask.any()):  # Convert numpy.bool_ to Python bool
            mape = float(np.mean(np.abs((actual[non_zero_mask] - predicted[non_zero_mask]) / actual[non_zero_mask])) * 100)
        else:
            mape = 0.0
        
        # Calculate metrics and convert to Python float
        mse = float(np.mean((actual - predicted) ** 2))
        rmse = float(np.sqrt(mse))
        mae = float(np.mean(np.abs(actual - predicted)))
        r2 = float(1 - np.sum((actual - predicted) ** 2) / (np.sum((actual - np.mean(actual)) ** 2) + 1e-10))
        
        metrics = {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'mape': mape,
            'r2': r2
        }
        
        # Ensure no nan values in metrics and convert to Python float
        metrics = {k: float(v) if not np.isnan(v) else 0.0 for k, v in metrics.items()}
        return metrics
    
    def grid_search(self, data: Union[pd.Series, pd.DataFrame, np.ndarray],
                   param_grid: Optional[Dict[str, List[Any]]] = None,
                   **kwargs) -> Dict[str, Any]:
        """Perform grid search to find optimal Prophet parameters."""
        if data is None:
            raise ValueError("Data cannot be None")
            
        if param_grid is None:
            if 'grid_search' not in self.config or 'param_grid' not in self.config['grid_search']:
                raise ValueError("No parameter grid provided and none found in config")
            param_grid = self.config['grid_search']['param_grid']
            if not isinstance(param_grid, dict):
                raise ValueError("Parameter grid must be a dictionary")
        
        logger.info("Starting grid search for best parameters...")
        logger.debug(f"Parameter grid: {param_grid}")
        
        try:
            best_rmse = float('inf')
            best_params = None
            best_metrics = None
            results = []
            no_improvement_count = 0
            
            # Generate parameter combinations
            from itertools import product
            param_names = list(param_grid.keys())
            param_values = list(param_grid.values())
            param_combinations = list(product(*param_values))
            
            if kwargs.get('max_iterations'):
                param_combinations = param_combinations[:kwargs['max_iterations']]
            
            total_combinations = len(param_combinations)
            logger.info(f"Total parameter combinations to evaluate: {total_combinations}")
            
            for i, values in enumerate(param_combinations, 1):
                params = dict(zip(param_names, values))
                logger.info(f"Testing combination {i}/{total_combinations}: {params}")
                
                try:
                    # Create and fit model with current parameters
                    self.config['model'].update(params)
                    self.initialize_model()
                    metrics = self.fit(data)
                    
                    # Convert numpy types to Python types
                    metrics = {k: float(v) for k, v in metrics.items()}
                    
                    result = {
                        'params': params,
                        'metrics': metrics
                    }
                    results.append(result)
                    
                    if metrics['rmse'] < best_rmse:
                        logger.info(f"New best model found with RMSE: {metrics['rmse']}")
                        best_rmse = float(metrics['rmse'])  # Convert to Python float
                        best_params = params.copy()  # Make a copy to avoid reference issues
                        best_metrics = metrics.copy()  # Make a copy to avoid reference issues
                        no_improvement_count = 0
                    else:
                        no_improvement_count += 1
                    
                    if kwargs.get('early_stopping', True) and no_improvement_count >= kwargs.get('early_stopping_patience', 5):
                        logger.info(f"Early stopping triggered after {i} iterations")
                        break
                        
                except Exception as e:
                    logger.warning(f"Failed to evaluate combination {params}: {str(e)}")
                    continue
            
            if not results:
                raise ValueError("No valid parameter combinations found during grid search")
            
            grid_search_results = {
                'best_params': best_params,
                'best_metrics': best_metrics,
                'all_results': sorted(results, key=lambda x: float(x['metrics']['rmse']))[:10],
                'total_combinations_tested': len(results)
            }
            
            # Update model with best parameters
            if best_params:
                self.config['model'].update(best_params)
                self.initialize_model()
                self.fit(data)
            
            logger.info(f"Grid search completed successfully. Best parameters: {best_params}")
            return grid_search_results
            
        except Exception as e:
            logger.error(f"Error in grid search: {str(e)}")
            raise
    
    def fit_with_optuna(self, data: Union[pd.Series, pd.DataFrame, np.ndarray],
                       n_trials: int = 100, timeout: int = 600) -> Dict[str, Any]:
        """Optimize hyperparameters using Optuna."""
        if data is None:
            raise ValueError("Data cannot be None")
            
        logger.info(f"Starting Optuna optimization with {n_trials} trials and {timeout}s timeout")
        
        def objective(trial):
            """Optimization objective function."""
            params = {
                'changepoint_prior_scale': trial.suggest_loguniform('changepoint_prior_scale', 0.001, 0.5),
                'seasonality_prior_scale': trial.suggest_loguniform('seasonality_prior_scale', 0.01, 10),
                'holidays_prior_scale': trial.suggest_loguniform('holidays_prior_scale', 0.01, 10),
                'seasonality_mode': trial.suggest_categorical('seasonality_mode', ['additive', 'multiplicative']),
                'changepoint_range': trial.suggest_uniform('changepoint_range', 0.8, 0.95),
            }
            
            try:
                # Create and fit model with trial parameters
                self.config['model'].update(params)
                self.initialize_model()
                metrics = self.fit(data)
                return float(metrics['rmse'])  # Convert to Python float
            except Exception as e:
                logger.warning(f"Trial failed: {str(e)}")
                return float('inf')
        
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=n_trials, timeout=timeout)
        
        best_params = study.best_params
        logger.info(f"Best parameters found: {best_params}")
        
        # Update model with best parameters and fit
        self.config['model'].update(best_params)
        self.initialize_model()
        final_metrics = self.fit(data)
        
        # Convert all metrics to Python native types
        final_metrics = {k: float(v) for k, v in final_metrics.items()}
        
        hyperparameter_optimization_results = {
            'best_params': best_params,
            'best_metrics': final_metrics
        }
        
        if self.tracker:
            logger.debug("Logging optimization results to tracker")
            self.tracker.log_params_safely({
                'optimization.n_trials': n_trials,
                'optimization.timeout': timeout,
                'optimization.best_value': float(study.best_value),  # Convert to Python float
                **{f'optimization.best_params.{k}': float(v) if isinstance(v, (np.floating, np.integer)) else v 
                   for k, v in best_params.items()}
            })
        
        logger.info("Optuna optimization completed successfully")
        return hyperparameter_optimization_results
    
    def save(self, path: str) -> None:
        """Save the model to disk."""
        logger.info(f"Saving Prophet model to {path}")
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, 'wb') as f:
                joblib.dump(self.fitted_model, f)
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            raise
    
    def load(self, path: str) -> None:
        """Load the model from disk."""
        logger.info(f"Loading Prophet model from {path}")
        try:
            with open(path, 'rb') as f:
                self.fitted_model = joblib.load(f)
            self.model = self.fitted_model
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise 