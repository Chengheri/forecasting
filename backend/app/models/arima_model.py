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
from ..utils.analyzer import Analyzer
import multiprocessing as mp
from multiprocessing import Process, Queue
import signal
import time
import os
import tempfile
import uuid
import optuna
from statsmodels.tsa.stattools import adfuller, kpss

warnings.filterwarnings('ignore')
logger = Logger()

class TimeSeriesModel:
    """ARIMA/SARIMA time series model with unified tracking."""
    
    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        tracker: Optional[ARIMATracker] = None,
        data: Optional[Union[pd.DataFrame, pd.Series, np.ndarray]] = None,
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
        """
        logger.info("Initializing TimeSeriesModel")
        logger.debug(f"Configuration provided: {config}")

        self.config = config
        self.tracker = tracker
        self.model = None
        self.fitted_model = None
        self._training_data = None  

        # Initialize model with data if provided
        if data is not None:
            self.initialize_model(data, params=None)

        self._log_initialization()

        # Initialize analyzer
        self.analyzer = Analyzer(config=self.config) 
    
    def _log_initialization(self) -> None:
        """Log initialization parameters."""
        logger.info(f"Initialized {self.config['model']['model_type'].upper()} model with parameters:")
        logger.info(f"Order parameters: p={self.config['model']['p']}, d={self.config['model']['d']}, q={self.config['model']['q']}")
        if self.config['model']['model_type'] == 'sarima':
            logger.info(f"Seasonal parameters: P={self.config['model']['P']}, D={self.config['model']['D']}, Q={self.config['model']['Q']}, s={self.config['model']['s']}")
        
        if self.tracker:
            logger.debug("Logging model parameters to tracker")
            self.tracker.log_params_safely({
                'model.type': self.config['model']['model_type']
            })
    
    def _get_fit_params(self) -> Dict[str, Any]:
        """Get fit parameters based on model type."""
        if self.config['model']['model_type'] == 'sarima':
            return {
                'maxiter': self.config['model'].get('maxiter', 50),
                'method': self.config['model'].get('method', 'lbfgs'),
                'disp': False
            }
        return {
            'method': self.config['model'].get('method', 'statespace')
        }
    
    def _create_model_instance(self, ts: pd.Series) -> Union[ARIMA, SARIMAX]:
        """Create ARIMA or SARIMA model instance based on configuration."""
        if self.config['model']['model_type'] == 'arima':
            return ARIMA(
                ts,
                order=(self.config['model']['p'], self.config['model']['d'], self.config['model']['q']),
                trend=self.config['model'].get('trend'),
                enforce_stationarity=self.config['model'].get('enforce_stationarity', True),
                enforce_invertibility=self.config['model'].get('enforce_invertibility', True),
                concentrate_scale=self.config['model'].get('concentrate_scale', False),
            )
        else:  # sarima
            return SARIMAX(
                ts,
                order=(self.config['model']['p'], self.config['model']['d'], self.config['model']['q']),
                seasonal_order=(self.config['model']['P'], self.config['model']['D'], self.config['model']['Q'], self.config['model']['s']),
                trend=self.config['model'].get('trend'),
                enforce_stationarity=self.config['model'].get('enforce_stationarity', True),
                enforce_invertibility=self.config['model'].get('enforce_invertibility', True),
                concentrate_scale=self.config['model'].get('concentrate_scale', False),
            )
    
    def initialize_model(self, data: Optional[Union[pd.DataFrame, pd.Series, np.ndarray]] = None, params: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the ARIMA model with data."""
        if params:
            self.config['model'].update(params)
        
        logger.info("Initializing model with provided data")
        if data is not None:
            self._training_data = data if isinstance(data, pd.Series) else pd.Series(data)
        else:
            if self._training_data is None:
                raise ValueError("No data provided for training")
        
        try:
            self.model = self._create_model_instance(self._training_data)
            logger.info(f"Model initialized successfully: {self.config['model']['model_type'].upper()}")
        except Exception as e:
            logger.error(f"Error initializing model: {str(e)}")
            raise
    
    @property
    def data(self):
        """Get the stored training data."""
        return self._training_data
    
    def fit(self, data: Union[pd.Series, pd.DataFrame, np.ndarray], params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Fit the ARIMA model to the provided data.
        """
        try:
            # Convert input data to Series if needed
            ts = data if isinstance(data, pd.Series) else pd.Series(data)
            self._training_data = ts
            
            # Fit model with appropriate parameters
            logger.info("Fitting model")
            self.initialize_model(ts, params=params)
            fit_params = self._get_fit_params()
            self.fitted_model = self.model.fit(**fit_params)
            
            # Calculate metrics
            logger.info("Calculating model training metrics...")
            metrics = {
                "aic": self.fitted_model.aic,
                "bic": self.fitted_model.bic,
                "hqic": self.fitted_model.hqic,
                "llf": self.fitted_model.llf
            }
            logger.debug(f"Model training metrics: {metrics}")
            logger.info(f"Model training completed successfully")
            logger.info(f"Final metrics - AIC: {metrics['aic']:.2f}, BIC: {metrics['bic']:.2f}")
            return metrics
            
        except Exception as e:
            logger.error(f"Error in model training: {str(e)}")
            raise
    
    def predict(self, data: Union[pd.Series, pd.DataFrame, np.ndarray], **kwargs) -> Tuple[pd.Series, Tuple[pd.Series, pd.Series]]:
        """
        Generate predictions for test data using the fitted ARIMA model.
        
        Args:
            data: Test data to generate predictions for
            **kwargs: Additional keyword arguments for prediction
            
        Returns:
            Tuple containing:
            - predictions: Predicted values as pd.Series
            - confidence_intervals: Tuple of (lower_bound, upper_bound) as pd.Series
            
        Raises:
            ValueError: If model is not trained or no test data provided
        """
        if data is None:
            raise ValueError("Test data must be provided for prediction")
            
        # Convert input data to Series if needed
        ts = data if isinstance(data, pd.Series) else pd.Series(data)
        
        logger.info(f"Generating predictions for {len(ts)} steps...")
        
        try:
            if self.fitted_model is None:
                logger.error("Model not trained")
                raise ValueError("Model must be trained before prediction")
            
            forecast = self.fitted_model.get_forecast(steps=len(ts))
            logger.info("Calculating confidence intervals...")
            predictions = forecast.predicted_mean
            conf_int = forecast.conf_int()
            
            logger.debug(f"Predictions shape: {predictions.shape}")
            logger.debug(f"Confidence intervals shape: {conf_int.shape}")
            logger.info("Predictions generated successfully")
            return predictions, (conf_int.iloc[:, 0], conf_int.iloc[:, 1])
            
        except Exception as e:
            logger.error(f"Error generating predictions: {str(e)}")
            raise
    
    def save_model(self, path: str) -> None:
        """Save the trained model and configuration."""
        logger.info(f"Saving model to {path}")
        
        try:
            if self.fitted_model is None:
                logger.error("No fitted model available to save")
                raise ValueError("Model must be trained before saving")
            
            logger.debug("Preparing model data for saving")
            model_data = {
                'fitted_model': self.fitted_model,
                'config': self.config['model']
            }
            
            logger.info("Writing model to disk")
            joblib.dump(model_data, path)
            logger.info("Model saved successfully")               
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            raise
    
    def load_model(self, path: str) -> None:
        """Load a trained model and configuration."""
        logger.info(f"Loading model from {path}")
        
        try:
            logger.debug("Reading model from disk")
            model_data = joblib.load(path)
            
            logger.info("Updating model attributes")
            self.fitted_model = model_data['fitted_model']
            self.config['model'] = model_data['config']
            
            logger.debug(f"Loaded model configuration: {self.config['model']}")
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
    
    @staticmethod
    def _fit_process_worker(model, config: Dict[str, Any], temp_path: str, result_queue: Queue) -> None:
        """Worker process for fitting a model with timeout."""
        logger.debug("Starting worker process for model fitting")
        try:
            logger.info("Fitting model in worker process")
            fitted = model.fit(
                maxiter=config['maxiter'],
                method=config['method'],
                disp=False
            )
            
            logger.debug("Saving fitted model to temporary file")
            joblib.dump(fitted, temp_path)
            
            logger.debug("Preparing result metrics")
            result_queue.put({
                'success': True,
                'aic': fitted.aic,
                'bic': fitted.bic,
                'hqic': fitted.hqic,
                'converged': fitted.mle_retvals['converged'],
                'iterations': fitted.mle_retvals['iterations']
            })
            logger.info("Worker process completed successfully")
        except Exception as e:
            logger.error(f"Error in worker process: {str(e)}")
            result_queue.put({
                'success': False,
                'error': str(e)
            })

    def _generate_param_combinations(self, param_grid: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
        """Generate all parameter combinations for grid search."""
        combinations = []
        
        if self.config['model']['model_type'] == 'sarima':
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
                                                'model_type': 'sarima',
                                                'p': p_val, 'd': d_val, 'q': q_val,
                                                'P': P_val, 'D': D_val, 'Q': Q_val, 's': s_val
                                            })
                        else:
                            # Otherwise, use default seasonal parameters
                            combinations.append({
                                'model_type': 'sarima',
                                'p': p_val, 'd': d_val, 'q': q_val,
                                'P': P[0], 'D': D[0], 'Q': Q[0], 's': s[0]
                            })
        else:
            for p in param_grid.get('p', [1]):
                for d in param_grid.get('d', [1]):
                    for q in param_grid.get('q', [1]):
                        combinations.append({
                            'model_type': 'arima',
                            'p': p, 'd': d, 'q': q
                        })
        
        return combinations

    def _evaluate_combination(self, ts: pd.Series, params: Dict[str, Any], timeout: int = 60) -> Optional[Dict[str, Any]]:
        """Evaluate a single parameter combination with process-based timeout."""
        logger.info(f"Evaluating parameter combination: {params}")
        temp_path = None
        process = None
        
        try:
            logger.debug("Updating configuration for evaluation")
            temp_config = self.config['model'].copy()
            # Ensure model_type is included in parameters
            params['model_type'] = self.config['model']['model_type']
            temp_config.update(params)
            
            logger.debug("Preparing data for evaluation")
            if isinstance(ts, pd.Series):
                ts_data = ts.values
            elif isinstance(ts, np.ndarray):
                ts_data = ts.ravel() if ts.ndim > 1 else ts
            else:
                ts_data = np.array(ts).ravel()
            
            logger.debug("Creating temporary file for model")
            temp_dir = tempfile.gettempdir()
            temp_path = os.path.join(temp_dir, f'sarima_model_{uuid.uuid4()}.joblib')
            
            logger.info("Initializing temporary model")
            temp_model = TimeSeriesModel(
                config=temp_config,
                tracker=self.tracker,
                data=ts_data,
            )
            
            logger.debug("Setting up multiprocessing")
            result_queue = Queue()
            process = Process(
                target=TimeSeriesModel._fit_process_worker,
                args=(temp_model.model, temp_config, temp_path, result_queue)
            )
            
            logger.info(f"Starting evaluation process with {timeout}s timeout")
            process.start()
            start_time = time.time()
            
            while time.time() - start_time < timeout:
                if not result_queue.empty():
                    result = result_queue.get()
                    process.terminate()
                    process.join()
                    process = None
                    
                    if result['success'] and os.path.exists(temp_path):
                        logger.info("Loading successful model fit")
                        fitted_model = joblib.load(temp_path)
                        os.remove(temp_path)
                        temp_path = None
                        
                        logger.debug(f"Evaluation metrics: {result}")
                        return {
                            **params,  # This now includes model_type
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
                
                time.sleep(0.1)
            
            logger.warning(f"Timeout reached for combination {params}")
            if process:
                process.terminate()
                process.join()
            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)
            return None
            
        except Exception as e:
            logger.error(f"Error evaluating combination: {str(e)}")
            if process:
                process.terminate()
                process.join()
            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)
            return None

    def grid_search(self, data: Optional[Union[pd.Series, pd.DataFrame, np.ndarray]] = None,
                   param_grid: Optional[Dict[str, List[int]]] = None,
                   **kwargs) -> Dict[str, Any]:
        """
        Perform grid search to find optimal ARIMA parameters.
        """
        if data is not None:
            ts = data if isinstance(data, pd.Series) else pd.Series(data)
        else:
            if self._training_data is None:
                raise ValueError("No data provided and no training data available")
            ts = self._training_data
        
        if param_grid is None:
            param_grid = {}
        
        logger.info("Starting grid search for best parameters...")
        logger.debug(f"Parameter grid: {param_grid}")
        logger.debug(f"Max iterations: {kwargs.get('max_iterations', None)}, Early stopping: {kwargs.get('early_stopping', True)}, Patience: {kwargs.get('early_stopping_patience', 5)}")
        
        try:
            best_aic = float('inf')
            best_bic = float('inf')
            best_hqic = float('inf')
            best_params = None
            self.best_model = None
            results = []
            no_improvement_count = 0
            
            # Ensure model_type is included in param_grid
            if 'model_type' not in param_grid:
                param_grid['model_type'] = [self.config['model']['model_type']]
            
            total_combinations = np.prod([len(values) for values in param_grid.values()])
            if kwargs.get('max_iterations', None):
                total_combinations = min(total_combinations, kwargs['max_iterations'])
            logger.info(f"Total parameter combinations to evaluate: {total_combinations}")
            
            logger.debug("Generating parameter combinations")
            param_combinations = self._generate_param_combinations(param_grid)
            if kwargs.get('max_iterations', None):
                param_combinations = param_combinations[:kwargs['max_iterations']]
            
            if self.tracker:
                logger.debug("Logging grid search parameters to tracker")
                self.tracker.log_params_safely({
                    "grid_search.param_grid": param_grid,
                    "grid_search.max_iterations": kwargs.get('max_iterations', None),
                    "grid_search.early_stopping": kwargs.get('early_stopping', True),
                    "grid_search.patience": kwargs.get('early_stopping_patience', 5),
                    "grid_search.timeout": kwargs.get('timeout', 60),
                })
            
            for i, params in enumerate(param_combinations, 1):
                logger.info(f"Testing combination {i}/{len(param_combinations)}: {params}")
                
                dynamic_timeout = kwargs.get('timeout', 60)
                if params.get('P', 0) > 0 or params.get('Q', 0) > 0:
                    logger.debug("Increasing timeout for seasonal model")
                    dynamic_timeout = dynamic_timeout * 2
                if params.get('p', 1) > 2 or params.get('q', 1) > 2:
                    logger.debug("Increasing timeout for higher-order model")
                    dynamic_timeout = int(dynamic_timeout * 1.5)
                
                result = self._evaluate_combination(ts, params, timeout=dynamic_timeout)
                if result is None:
                    continue
                
                model_obj = result['model']
                result['model'] = self._to_serializable(result['model'])
                results.append(result)
                
                if result['aic'] < best_aic:
                    logger.info(f"New best model found with AIC: {result['aic']}")
                    logger.debug(f"Best parameters: {params}")
                    best_aic = result['aic']
                    best_bic = result['bic']
                    best_hqic = result['hqic']
                    best_params = params
                    self.best_model = model_obj
                    no_improvement_count = 0
                else:
                    no_improvement_count += 1
                    logger.debug(f"No improvement for {no_improvement_count} iterations")
                
                if kwargs.get('early_stopping', True) and no_improvement_count >= kwargs.get('early_stopping_patience', 5):
                    logger.info(f"Early stopping triggered after {i} iterations")
                    break
            
            if not results:
                logger.error("No valid parameter combinations found")
                raise ValueError("No valid parameter combinations found during grid search")
            
            grid_search_results = {
                'best_params': best_params,
                'best_aic': best_aic,
                'best_bic': best_bic,
                'best_hqic': best_hqic,
                'best_model': self._to_serializable(self.best_model),
                'all_results': sorted(results, key=lambda x: x['aic'])[:10],
                'total_combinations_tested': len(results),
                'convergence_rate': sum(1 for r in results if r.get('converged', False)) / len(results) if results else 0
            }
            
            if self.tracker:
                logger.debug("Logging grid search results to tracker")
                self.tracker.log_params_safely({
                    "grid_search.best_params": best_params,
                    "grid_search.best_aic": best_aic,
                    "grid_search.best_bic": best_bic,
                    "grid_search.best_hqic": best_hqic,
                    "grid_search.total_combinations": len(results),
                    "grid_search.convergence_rate": grid_search_results['convergence_rate']
                })
            
            logger.info(f"Grid search completed successfully. Best parameters: {best_params}")
            return grid_search_results
            
        except Exception as e:
            logger.error(f"Error in grid search: {str(e)}")
            raise

    def get_best_model(self):
        """Get the best model from grid search."""
        if self.best_model is None:
            raise ValueError("No best model available. Run grid search first.")
        return self.best_model

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

    def fit_with_optuna(self, data: Optional[Union[pd.Series, pd.DataFrame, np.ndarray]] = None, n_trials=100, timeout=600):
        """Optimize hyperparameters using Optuna and fit the model."""
        logger.info(f"Starting Optuna optimization with {n_trials} trials and {timeout}s timeout")
        
        if data is not None:
            ts = data if isinstance(data, pd.Series) else pd.Series(data)
        else:
            if self._training_data is None:
                raise ValueError("No data provided and no training data available")
            ts = self._training_data
        
        def objective(trial):
            """Optimization objective function."""
            logger.debug(f"Starting trial {trial.number}")
            
            # Define common hyperparameters
            params = {
                'trend': trial.suggest_categorical('trend', ['n', 'c', 't', 'ct']),
                'enforce_stationarity': trial.suggest_categorical('enforce_stationarity', [True, False]),
                'enforce_invertibility': trial.suggest_categorical('enforce_invertibility', [True, False]),
                'concentrate_scale': trial.suggest_categorical('concentrate_scale', [True, False])
            }
            
            # Add model-specific parameters
            if self.config['model']['model_type'] == 'sarima':
                params['method'] = trial.suggest_categorical('method', ['lbfgs', 'bfgs', 'newton', 'cg', 'nm', 'powell'])
                params['maxiter'] = trial.suggest_int('maxiter', 50, 500)
            else:  # ARIMA
                params['method'] = trial.suggest_categorical('method', ['statespace', 'innovations_mle', 'hannan_rissanen'])
            
            logger.debug(f"Trial parameters: {params}")
            
            # Keep structural parameters unchanged
            logger.info(f"Keep structural parameters unchanged")
            structural_params = {
                'p': self.config['model']['p'],
                'd': self.config['model']['d'],
                'q': self.config['model']['q']
            }
            
            if self.config['model']['model_type'] == 'sarima':
                structural_params.update({
                    'P': self.config['model']['P'],
                    'D': self.config['model']['D'],
                    'Q': self.config['model']['Q'],
                    's': self.config['model']['s']
                })
            logger.debug(f"Structural parameters: {structural_params}")
            
            # Update configuration
            trial_config = self.config.copy()
            trial_config['model'].update(params)
            trial_config['model'].update(structural_params)
            
            try:
                # Create a temporary model for this trial
                temp_model = TimeSeriesModel(
                    config=trial_config,
                    tracker=None,  # Don't track individual trials
                    data=ts,
                )
                
                logger.info(f"Fitting model for trial {trial.number}")
                metrics = temp_model.fit(data=ts)
                logger.debug(f"Trial {trial.number} completed with AIC: {metrics['aic']}")
                return metrics['aic']
            except Exception as e:
                logger.warning(f"Trial {trial.number} failed: {str(e)}")
                return float('inf')
        
        logger.info("Creating and running Optuna study...")
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=n_trials, timeout=timeout)
        
        best_params = study.best_params
        logger.info(f"Best parameters found: {best_params}")
        logger.info(f"Best value (AIC): {study.best_value}")
        
        # Update model with best parameters
        self.config['model'].update(best_params)
        
        logger.info("Fitting final model with best parameters...")
        final_metrics = self.fit(data=ts, params=best_params)
        hyperparameter_optimization_results = {
            'best_params': best_params,
            'best_metrics': final_metrics
        }

        
        if self.tracker:
            logger.debug("Logging optimization results to tracker")
            try:
                self.tracker.log_params_safely({
                    'optimization.n_trials': n_trials,
                    'optimization.timeout': timeout,
                    'optimization.best_value': study.best_value,
                    **{f'optimization.best_params.{k}': v for k, v in best_params.items()}
                })
            except Exception as e:
                logger.warning(f"Failed to log optimization results: {str(e)}")
        
        logger.info("Optuna optimization completed successfully")
        return hyperparameter_optimization_results