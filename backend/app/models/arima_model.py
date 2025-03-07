import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, List, Union, Optional
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import warnings
from ..utils.logger import Logger
from ..utils.metrics import ForecastingMetrics
from ..utils.trackers import ARIMATracker
import multiprocessing as mp
from multiprocessing import Process, Queue
import signal
import time
import os
import tempfile
import uuid
import optuna

warnings.filterwarnings('ignore')
logger = Logger()

class TimeSeriesModel:
    """ARIMA/SARIMA time series model with unified tracking."""
    
    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        tracker: Optional[ARIMATracker] = None,
        data: Optional[Union[pd.DataFrame, pd.Series, np.ndarray]] = None,
        force_preparation: bool = False
    ):
        """Initialize TimeSeriesModel with parameters and optionally with data.
        
        Args:
            config: Model configuration dictionary with parameters:
                   - model_type: str = 'sarima'
                   - p: int = 1  # AR order
                   - d: int = 1  # Difference order
                   - q: int = 1  # MA order
                   - P: int = 0  # Seasonal AR order (SARIMA only)
                   - D: int = 0  # Seasonal difference order (SARIMA only)
                   - Q: int = 0  # Seasonal MA order (SARIMA only)
                   - s: int = 24  # Seasonal period (SARIMA only)
                   - maxiter: int = 50  # Maximum iterations for fitting
                   - method: str = 'lbfgs'  # Optimization method
                   - trend: Optional[str] = None  # Trend component
                   - enforce_stationarity: bool = True
                   - enforce_invertibility: bool = True
                   - concentrate_scale: bool = False
            tracker: MLflow tracker instance (optional)
            data: Input time series data (optional)
            force_preparation: If True, always process the data. If False, use pd.Series as is.
        """
        # Default configuration
        self.config = {
            'model_type': 'sarima',
            'p': 1,
            'd': 1,
            'q': 1,
            'P': 0,
            'D': 0,
            'Q': 0,
            's': 24,
            'maxiter': 50,
            'method': 'lbfgs',
            'trend': None,
            'enforce_stationarity': True,
            'enforce_invertibility': True,
            'concentrate_scale': False
        }
        
        # Update with provided configuration
        if config:
            self.config.update(config)
        
        self.tracker = tracker if tracker else ARIMATracker()
        self.model = None
        self.fitted_model = None
        self._training_data = data  # Store the training data
        
        logger.info(f"Initialized {self.config['model_type'].upper()} model")
        if self.tracker:
            self.tracker.log_model_params(self.config)
            
        # Initialize model with data if provided
        if data is not None:
            ts = self.prepare_data(data, force_preparation=force_preparation)
            self.initialize_model(ts)
            logger.info("Model initialized with provided data")
    
    @staticmethod
    def prepare_data(data: Union[pd.DataFrame, pd.Series, np.ndarray], force_preparation: bool = False) -> pd.Series:
        """Prepare time series data for modeling.
        
        Args:
            data: Input data in various formats
            force_preparation: If True, always process the data. If False, return pd.Series as is.
        
        Returns:
            pd.Series: Prepared time series data
        """
        try:
            # If data is already a Series and force_preparation is False, return as is
            if isinstance(data, pd.Series) and not force_preparation:
                return data
                
            if isinstance(data, np.ndarray):
                # Ensure array is 1D
                if data.ndim > 1:
                    data = data.ravel()
                return pd.Series(data)
            elif isinstance(data, pd.DataFrame):
                if 'timestamp' in data.columns and 'consumption' in data.columns:
                    return data.set_index('timestamp')['consumption']
                # If single column DataFrame, convert to Series
                if len(data.columns) == 1:
                    return data[data.columns[0]]
                raise ValueError("DataFrame must contain 'timestamp' and 'consumption' columns or be single column")
            elif isinstance(data, pd.Series):
                return data
            raise ValueError("Data must be pandas DataFrame, Series, or numpy array")
        except Exception as e:
            logger.error(f"Error preparing data: {str(e)}")
            raise
    
    def initialize_model(self, ts: pd.Series) -> None:
        """Initialize ARIMA or SARIMA model with configured parameters."""
        try:
            if self.config['model_type'] == 'arima':
                logger.info(f"Initializing ARIMA(p={self.config['p']}, d={self.config['d']}, q={self.config['q']})")
                self.model = ARIMA(ts, order=(self.config['p'], self.config['d'], self.config['q']))
            else:
                logger.info(
                    f"Initializing SARIMA(p={self.config['p']}, d={self.config['d']}, q={self.config['q']}) "
                    f"x (P={self.config['P']}, D={self.config['D']}, Q={self.config['Q']}, s={self.config['s']})"
                )
                self.model = SARIMAX(
                    ts,
                    order=(self.config['p'], self.config['d'], self.config['q']),
                    seasonal_order=(self.config['P'], self.config['D'], self.config['Q'], self.config['s']),
                    enforce_stationarity=self.config['enforce_stationarity'],
                    enforce_invertibility=self.config['enforce_invertibility'],
                    concentrate_scale=self.config['concentrate_scale'],
                    trend=self.config['trend']
                )
            self._training_data = ts  # Update training data
        except Exception as e:
            logger.error(f"Error initializing model: {str(e)}")
            raise
    
    @property
    def data(self):
        """Get the stored training data."""
        return self._training_data
    
    def fit(self, data: Optional[Union[pd.DataFrame, pd.Series, np.ndarray]] = None, force_preparation: bool = False) -> Dict[str, Any]:
        """Train ARIMA or SARIMA model with tracking.
        
        Args:
            data: Input time series data (optional if model was initialized with data)
            force_preparation: If True, always process the data. If False, use pd.Series as is.
        """
        logger.info("Starting model training")
        try:
            # Use existing model if initialized with data and no new data provided
            if data is None and self.model is not None:
                logger.info("Using model initialized during instantiation")
            else:
                if data is None:
                    raise ValueError("Data must be provided either during initialization or fit")
                # Prepare data if needed
                ts = self.prepare_data(data, force_preparation=force_preparation)
                # Initialize model
                self.initialize_model(ts)
            
            # Fit model
            self.fitted_model = self.model.fit(
                maxiter=self.config['maxiter'],
                method=self.config['method'],
                disp=False
            )
            
            # Calculate metrics
            metrics = {
                "aic": self.fitted_model.aic,
                "bic": self.fitted_model.bic,
                "hqic": self.fitted_model.hqic,
                "llf": self.fitted_model.llf
            }
            
            # Log training metrics
            if self.tracker:
                self.tracker.log_training_metrics(metrics)
            
            logger.info(f"Model training completed. AIC: {metrics['aic']:.2f}, BIC: {metrics['bic']:.2f}")
            return metrics
            
        except Exception as e:
            logger.error(f"Error training model: {str(e)}")
            raise
    
    def predict(self, steps: int) -> Tuple[pd.Series, pd.DataFrame]:
        """Generate forecasts and confidence intervals."""
        try:
            if self.fitted_model is None:
                raise ValueError("Model must be trained before prediction")
            
            logger.info(f"Generating predictions for {steps} steps")
            
            # Get forecast with confidence intervals
            forecast = self.fitted_model.get_forecast(steps=steps)
            predictions = forecast.predicted_mean
            conf_int = forecast.conf_int()
            
            # Add confidence intervals to predictions
            forecast_df = pd.DataFrame({
                'prediction': predictions,
                'lower': conf_int.iloc[:, 0],
                'upper': conf_int.iloc[:, 1]
            })
            
            return predictions, forecast_df
            
        except Exception as e:
            logger.error(f"Error generating predictions: {str(e)}")
            raise
    
    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Evaluate model performance."""
        try:
            metrics = {
                'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
                'mae': mean_absolute_error(y_true, y_pred),
                'mape': np.mean(np.abs((y_true - y_pred) / y_true)) * 100,
                'r2': r2_score(y_true, y_pred)
            }
            
            if self.tracker:
                self.tracker.log_forecast_metrics(metrics)
            
            return metrics
        except Exception as e:
            logger.error(f"Error evaluating model: {str(e)}")
            raise
    
    def save_model(self, path: str) -> None:
        """Save the trained model and configuration."""
        logger.info(f"Saving model to {path}")
        try:
            if self.fitted_model is None:
                raise ValueError("Model must be trained before saving")
            
            model_data = {
                'fitted_model': self.fitted_model,
                'config': self.config
            }
            joblib.dump(model_data, path)
            logger.info("Model saved successfully")
            
            if self.tracker:
                self.tracker.log_artifact(path)
                
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            raise
    
    def load_model(self, path: str) -> None:
        """Load a trained model and configuration."""
        logger.info(f"Loading model from {path}")
        try:
            model_data = joblib.load(path)
            self.fitted_model = model_data['fitted_model']
            self.config = model_data['config']
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
    
    @staticmethod
    def _fit_process_worker(model, config: Dict[str, Any], temp_path: str, result_queue: Queue) -> None:
        """Worker process for fitting a model with timeout.
        
        Args:
            model: ARIMA/SARIMA model instance
            config: Model configuration dictionary
            temp_path: Path to save fitted model temporarily
            result_queue: Queue to communicate results back to parent process
        """
        try:
            fitted = model.fit(
                maxiter=config['maxiter'],
                method=config['method'],
                disp=False
            )
            
            # Save fitted model to temporary file
            joblib.dump(fitted, temp_path)
            
            result_queue.put({
                'success': True,
                'aic': fitted.aic,
                'bic': fitted.bic,
                'hqic': fitted.hqic,
                'converged': fitted.mle_retvals['converged'],
                'iterations': fitted.mle_retvals['iterations']
            })
        except Exception as e:
            result_queue.put({
                'success': False,
                'error': str(e)
            })

    def _evaluate_combination(self, ts: pd.Series, params: Dict[str, Any], timeout: int = 60) -> Optional[Dict[str, Any]]:
        """Evaluate a single parameter combination with process-based timeout."""
        temp_path = None
        process = None
        
        try:
            # Update config with new parameters
            temp_config = self.config.copy()
            temp_config.update(params)
            
            # Ensure data is 1D
            if isinstance(ts, pd.Series):
                ts_data = ts.values
            elif isinstance(ts, np.ndarray):
                ts_data = ts.ravel() if ts.ndim > 1 else ts
            else:
                ts_data = np.array(ts).ravel()
            
            # Create temporary file path
            temp_dir = tempfile.gettempdir()
            temp_path = os.path.join(temp_dir, f'sarima_model_{uuid.uuid4()}.joblib')
            
            # Initialize model using TimeSeriesModel
            temp_model = TimeSeriesModel(
                config=temp_config,
                tracker=self.tracker,
                data=ts_data,
                force_preparation=True
            )
            
            # Set up multiprocessing
            result_queue = Queue()
            process = Process(
                target=TimeSeriesModel._fit_process_worker,
                args=(temp_model.model, temp_config, temp_path, result_queue)
            )
            
            # Start process and wait with timeout
            process.start()
            start_time = time.time()
            
            while time.time() - start_time < timeout:
                if not result_queue.empty():
                    result = result_queue.get()
                    process.terminate()
                    process.join()
                    process = None
                    
                    if result['success'] and os.path.exists(temp_path):
                        # Load fitted model from temporary file
                        fitted_model = joblib.load(temp_path)
                        os.remove(temp_path)
                        temp_path = None
                        
                        return {
                            **params,
                            'aic': result['aic'],
                            'bic': result['bic'],
                            'hqic': result['hqic'],
                            'converged': result['converged'],
                            'iterations': result['iterations'],
                            'model': fitted_model
                        }
                    else:
                        if temp_path and os.path.exists(temp_path):
                            os.remove(temp_path)
                            temp_path = None
                        logger.warning(f"Failed combination {params}: {result.get('error', 'Unknown error')}")
                        return None
                
                time.sleep(0.1)  # Short sleep to prevent busy waiting
            
            # Timeout reached
            if process:
                process.terminate()
                process.join()
            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)
            logger.warning(f"Timeout reached for combination {params}")
            return None
            
        except Exception as e:
            # Clean up
            if process:
                process.terminate()
                process.join()
            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)
            logger.warning(f"Failed combination {params}: {str(e)}")
            return None

    def _to_serializable(self, obj: Any) -> Any:
        """Convert objects to JSON serializable format.
        
        Args:
            obj: Object to convert
            
        Returns:
            JSON serializable version of the object
        """
        if hasattr(obj, 'to_dict'):
            return obj.to_dict()
        elif hasattr(obj, '__dict__'):
            return {k: self._to_serializable(v) for k, v in obj.__dict__.items()
                   if not k.startswith('_')}
        elif isinstance(obj, (list, tuple)):
            return [self._to_serializable(item) for item in obj]
        elif isinstance(obj, dict):
            return {k: self._to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (int, float, str, bool, type(None))):
            return obj
        elif hasattr(obj, 'summary'):
            # For statsmodels results objects
            return str(obj.summary())
        elif hasattr(obj, 'tolist'):
            # For numpy arrays
            return obj.tolist()
        else:
            return str(obj)

    def grid_search(
        self,
        data: Optional[Union[pd.DataFrame, pd.Series, np.ndarray]] = None,
        param_grid: Dict[str, List[Any]] = None,
        max_iterations: Optional[int] = None,
        early_stopping: bool = True,
        early_stopping_patience: int = 5,
        force_preparation: bool = False,
        timeout: int = 60
    ) -> Dict[str, Any]:
        """Perform grid search for best parameters with tracking."""
        logger.info("Starting grid search for best parameters")
        try:
            # Use existing model's data if no new data provided
            if data is None and self.model is not None:
                ts = self.model.endog
                logger.info("Using data from initialized model")
            else:
                if data is None:
                    raise ValueError("Data must be provided either during initialization or grid search")
                ts = self.prepare_data(data, force_preparation=force_preparation)
            
            best_aic = float('inf')
            best_bic = float('inf')
            best_hqic = float('inf')
            best_params = None
            self.best_model = None  # Store the actual model object
            results = []
            no_improvement_count = 0
            
            # Calculate total combinations
            total_combinations = np.prod([len(values) for values in param_grid.values()])
            if max_iterations:
                total_combinations = min(total_combinations, max_iterations)
            
            # Generate parameter combinations
            param_combinations = self._generate_param_combinations(param_grid)
            if max_iterations:
                param_combinations = param_combinations[:max_iterations]
            
            # Track grid search parameters
            if self.tracker:
                self.tracker.log_params_safely({
                    "grid_search.param_grid": param_grid,
                    "grid_search.max_iterations": max_iterations,
                    "grid_search.early_stopping": early_stopping,
                    "grid_search.patience": early_stopping_patience,
                    "grid_search.timeout": timeout
                })
            
            # Evaluate combinations
            for i, params in enumerate(param_combinations, 1):
                logger.info(f"Testing combination {i}/{len(param_combinations)}: {params}")
                
                # Calculate dynamic timeout based on model complexity
                dynamic_timeout = timeout
                if params.get('P', 0) > 0 or params.get('Q', 0) > 0:
                    # Increase timeout for seasonal models
                    dynamic_timeout = timeout * 2
                if params.get('p', 1) > 2 or params.get('q', 1) > 2:
                    # Increase timeout for higher-order models
                    dynamic_timeout = int(dynamic_timeout * 1.5)
                
                # Evaluate combination with dynamic timeout
                result = self._evaluate_combination(ts, params, timeout=dynamic_timeout)
                if result is None:
                    continue
                
                # Store the actual model object before serializing for results
                model_obj = result['model']
                result['model'] = self._to_serializable(result['model'])
                results.append(result)
                
                # Update best model
                if result['aic'] < best_aic:
                    logger.info(f"New best model found with AIC: {result['aic']}")
                    best_aic = result['aic']
                    best_bic = result['bic']
                    best_hqic = result['hqic']
                    best_params = params
                    self.best_model = model_obj  # Store the actual model object
                    no_improvement_count = 0
                else:
                    no_improvement_count += 1
                
                # Early stopping check
                if early_stopping and no_improvement_count >= early_stopping_patience:
                    logger.info(f"Early stopping triggered after {i} iterations")
                    break
            
            if not results:
                raise ValueError("No valid parameter combinations found during grid search")
            
            # Prepare results
            grid_search_results = {
                'best_params': best_params,
                'best_aic': best_aic,
                'best_bic': best_bic,
                'best_hqic': best_hqic,
                'best_model': self._to_serializable(self.best_model),  # Serialize for results
                'all_results': sorted(results, key=lambda x: x['aic'])[:10],
                'total_combinations_tested': len(results),
                'convergence_rate': sum(1 for r in results if r.get('converged', False)) / len(results) if results else 0
            }
            
            # Log results
            if self.tracker:
                self.tracker.log_params_safely({
                    "grid_search.best_params": best_params,
                    "grid_search.best_aic": best_aic,
                    "grid_search.best_bic": best_bic,
                    "grid_search.best_hqic": best_hqic,
                    "grid_search.total_combinations": len(results),
                    "grid_search.convergence_rate": grid_search_results['convergence_rate']
                })
            
            return grid_search_results
            
        except Exception as e:
            logger.error(f"Error in grid search: {str(e)}")
            raise

    def get_best_model(self):
        """Get the best model from grid search."""
        if self.best_model is None:
            raise ValueError("No best model available. Run grid search first.")
        return self.best_model

    def _generate_param_combinations(self, param_grid: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
        """Generate all parameter combinations for grid search."""
        combinations = []
        
        if self.config['model_type'] == 'sarima':
            # Get suggested parameters
            p = param_grid.get('p', [1])
            d = param_grid.get('d', [1])
            q = param_grid.get('q', [1])
            P = param_grid.get('P', [0])
            D = param_grid.get('D', [0])
            Q = param_grid.get('Q', [0])
            s = param_grid.get('s', [24])
            
            # Generate combinations with reduced search space
            for p_val in p:
                for d_val in d:
                    for q_val in q:
                        # Only test seasonal parameters if they're explicitly provided
                        if len(P) > 1 or len(D) > 1 or len(Q) > 1:
                            for P_val in P:
                                for D_val in D:
                                    for Q_val in Q:
                                        for s_val in s:
                                            combinations.append({
                                                'p': p_val, 'd': d_val, 'q': q_val,
                                                'P': P_val, 'D': D_val, 'Q': Q_val, 's': s_val
                                            })
                        else:
                            # Otherwise, use default seasonal parameters
                            combinations.append({
                                'p': p_val, 'd': d_val, 'q': q_val,
                                'P': P[0], 'D': D[0], 'Q': Q[0], 's': s[0]
                            })
        else:
            for p in param_grid.get('p', [1]):
                for d in param_grid.get('d', [1]):
                    for q in param_grid.get('q', [1]):
                        combinations.append({'p': p, 'd': d, 'q': q})
        
        return combinations 

    def optimize_hyperparameters(self, data: np.ndarray, fixed_params: Dict[str, Any],
                               n_trials: int = 100, timeout: int = 600) -> Dict[str, Any]:
        """Optimize non-structural hyperparameters using Optuna.
        
        Args:
            data: Time series data for model fitting
            fixed_params: Dictionary containing fixed structural parameters (p, d, q, P, D, Q, s)
            n_trials: Number of optimization trials
            timeout: Timeout in seconds
            
        Returns:
            Dictionary containing the best hyperparameters
        """
        def objective(trial):
            # Define hyperparameter search space
            params = {
                'method': trial.suggest_categorical('method', ['lbfgs', 'bfgs', 'cg', 'newton', 'nm']),
                'maxiter': trial.suggest_int('maxiter', 50, 500),
                'enforce_stationarity': trial.suggest_categorical('enforce_stationarity', [True, False]),
                'enforce_invertibility': trial.suggest_categorical('enforce_invertibility', [True, False]),
                'concentrate_scale': trial.suggest_categorical('concentrate_scale', [True, False]),
                'trend': trial.suggest_categorical('trend', ['n', 'c', 't', 'ct']),
                'disp': False  # Always set to False during optimization
            }
            
            # Combine with fixed structural parameters
            params.update(fixed_params)
            
            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore')
                    # Initialize and fit model
                    model = SARIMAX(data, **params)
                    results = model.fit()
                    
                    # Calculate objective value (AIC)
                    return results.aic
            except:
                # Return a large value if model fitting fails
                return float('inf')
        
        # Create Optuna study
        study = optuna.create_study(direction='minimize')
        
        # Run optimization
        study.optimize(objective, n_trials=n_trials, timeout=timeout)
        
        # Get best parameters
        best_params = study.best_params
        
        # Add fixed parameters
        best_params.update(fixed_params)
        
        # Log optimization results
        if self.tracker:
            try:
                self.tracker.log_params_safely({
                    'optuna.best_value': study.best_value,
                    'optuna.n_trials': len(study.trials),
                    'optuna.optimization_time': study.trials[-1].datetime_complete - study.trials[0].datetime_start
                })
            except Exception as e:
                self.logger.warning(f"Failed to log Optuna results: {str(e)}")
        
        return best_params

    def fit_with_optuna(self, n_trials=100, timeout=600):
        """Optimize hyperparameters using Optuna and fit the model.
        
        Args:
            n_trials (int): Number of optimization trials
            timeout (int): Timeout in seconds
            
        Returns:
            dict: Metrics from the best model
        """
        if self._training_data is None:
            raise ValueError("No training data available. Please provide data before optimization.")
            
        logger.info(f"Starting Optuna optimization with {n_trials} trials and {timeout}s timeout")
        
        def objective(trial):
            """Optimization objective function."""
            # Define the hyperparameter search space
            params = {
                'method': trial.suggest_categorical('method', ['lbfgs', 'bfgs', 'newton', 'cg']),
                'maxiter': trial.suggest_int('maxiter', 50, 500),
                'trend': trial.suggest_categorical('trend', ['n', 'c', 't', 'ct']),
                'enforce_stationarity': trial.suggest_categorical('enforce_stationarity', [True, False]),
                'enforce_invertibility': trial.suggest_categorical('enforce_invertibility', [True, False]),
                'concentrate_scale': trial.suggest_categorical('concentrate_scale', [True, False])
            }
            
            # Update model configuration with trial parameters
            self.config.update(params)
            
            try:
                # Fit model with current parameters
                metrics = self.fit()
                return metrics['aic']  # Optimize for AIC
            except Exception as e:
                logger.warning(f"Trial failed with parameters {params}: {str(e)}")
                return float('inf')  # Return infinity for failed trials
        
        # Create and run Optuna study
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=n_trials, timeout=timeout)
        
        # Get best parameters and update model configuration
        best_params = study.best_params
        self.config.update(best_params)
        logger.info(f"Best parameters found: {best_params}")
        
        # Fit final model with best parameters
        final_metrics = self.fit()
        
        # Log optimization results if tracker is available
        if self.tracker:
            try:
                self.tracker.log_params_safely({
                    'optimization.n_trials': n_trials,
                    'optimization.timeout': timeout,
                    'optimization.best_value': study.best_value,
                    **{f'optimization.best_params.{k}': v for k, v in best_params.items()}
                })
            except Exception as e:
                logger.warning(f"Failed to log optimization results: {str(e)}")
        
        return final_metrics 