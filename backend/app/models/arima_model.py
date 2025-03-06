import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, List, Union
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib
import warnings
from ..utils.logger import Logger
from ..utils.metrics import ForecastingMetrics

warnings.filterwarnings('ignore')
logger = Logger()

class TimeSeriesModel:
    def __init__(self, model_type: str = 'sarima', **kwargs):
        """Initialize TimeSeriesModel with parameters.
        
        Args:
            model_type: Type of model ('arima' or 'sarima')
            **kwargs: Model parameters
        """
        logger.info(f"Initializing TimeSeriesModel with config: {kwargs}")
        self.model_type = model_type
        self.params = kwargs
        self.model = None
        self.fitted_model = None
        logger.info(f"Model type: {self.model_type}")
    
    def prepare_data(self, data: pd.DataFrame) -> pd.Series:
        """Prepare time series data."""
        logger.info("Preparing time series data")
        try:
            if isinstance(data, pd.DataFrame):
                if 'timestamp' in data.columns and 'consumption' in data.columns:
                    ts = data.set_index('timestamp')['consumption']
                    logger.info("Data prepared from DataFrame with timestamp and consumption columns")
                else:
                    logger.error("DataFrame must contain 'timestamp' and 'consumption' columns")
                    raise ValueError("DataFrame must contain 'timestamp' and 'consumption' columns")
            elif isinstance(data, pd.Series):
                ts = data
                logger.info("Data prepared from Series")
            else:
                logger.error("Data must be pandas DataFrame or Series")
                raise ValueError("Data must be pandas DataFrame or Series")
            
            return ts
        except Exception as e:
            logger.error(f"Error preparing data: {str(e)}")
            raise
    
    def initialize_model(self, ts: pd.Series) -> None:
        """Initialize ARIMA or SARIMA model with configured parameters.
        
        Args:
            ts: Time series data to initialize the model with
        """
        if self.model_type == 'arima':
            # ARIMA parameters
            p = self.params.get('p', 1)  # AR order
            d = self.params.get('d', 1)  # Difference order
            q = self.params.get('q', 1)  # MA order
            
            logger.info(f"Initializing ARIMA model with parameters: p={p}, d={d}, q={q}")
            self.model = ARIMA(ts, order=(p, d, q))
            
        else:  # SARIMA
            # SARIMA parameters
            p = self.params.get('p', 1)  # AR order
            d = self.params.get('d', 1)  # Difference order
            q = self.params.get('q', 1)  # MA order
            P = self.params.get('P', 1)  # Seasonal AR order
            D = self.params.get('D', 1)  # Seasonal difference order
            Q = self.params.get('Q', 1)  # Seasonal MA order
            s = self.params.get('s', 24)  # Seasonal period
            
            logger.info(f"Initializing SARIMA model with parameters: p={p}, d={d}, q={q}, P={P}, D={D}, Q={Q}, s={s}")
            self.model = SARIMAX(ts,
                               order=(p, d, q),
                               seasonal_order=(P, D, Q, s))
    
    def fit(self, data: Union[pd.DataFrame, pd.Series, np.ndarray]) -> Dict[str, Any]:
        """Train ARIMA or SARIMA model."""
        logger.info("Starting model training")
        try:
            # Convert numpy array to pandas Series if needed
            if isinstance(data, np.ndarray):
                data = pd.Series(data)
            
            ts = self.prepare_data(data) if isinstance(data, pd.DataFrame) else data
            
            # Initialize the model
            self.initialize_model(ts)
            
            # Fit the model with maxiter parameter
            self.fitted_model = self.model.fit(maxiter=self.params.get('maxiter', 50))
            
            # Get model metrics
            metrics = {
                "aic": self.fitted_model.aic,
                "bic": self.fitted_model.bic,
                "hqic": self.fitted_model.hqic,
                "model_type": self.model_type
            }
            
            logger.info(f"Model training completed successfully. AIC: {metrics['aic']}, BIC: {metrics['bic']}, HQIC: {metrics['hqic']}")
            return metrics
            
        except Exception as e:
            logger.error(f"Error training model: {str(e)}")
            raise ValueError(f"Error fitting model: {str(e)}")
    
    def predict(self, steps: int) -> Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Generate forecasts and confidence intervals.
        
        Args:
            steps: Number of steps to forecast
            
        Returns:
            Tuple of predicted mean and confidence intervals
        """
        try:
            if self.fitted_model is None:
                logger.error("Model must be trained before prediction")
                raise ValueError("Model must be trained before prediction")
            
            logger.info(f"Generating predictions for {steps} steps")
            
            # Get forecast
            forecast = self.fitted_model.get_forecast(steps=steps)
            mean_forecast = forecast.predicted_mean
            
            # Get confidence intervals
            conf_int = forecast.conf_int()
            lower = conf_int.iloc[:, 0]
            upper = conf_int.iloc[:, 1]
            
            return mean_forecast, (lower, upper)
            
        except Exception as e:
            logger.error(f"Error generating predictions: {str(e)}")
            raise ValueError(f"Error generating predictions: {str(e)}")
    
    def save_model(self, path: str) -> None:
        """Save the trained model."""
        logger.info(f"Saving model to {path}")
        try:
            if self.fitted_model is None:
                logger.error("Model must be trained before saving")
                raise ValueError("Model must be trained before saving")
            
            model_data = {
                'fitted_model': self.fitted_model,
                'params': self.params,
                'model_type': self.model_type
            }
            joblib.dump(model_data, path)
            logger.info("Model saved successfully")
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            raise
    
    def load_model(self, path: str) -> None:
        """Load a trained model."""
        logger.info(f"Loading model from {path}")
        try:
            model_data = joblib.load(path)
            self.fitted_model = model_data['fitted_model']
            self.params = model_data['params']
            self.model_type = model_data['model_type']
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
    
    def grid_search(self, data: pd.DataFrame, param_grid: Dict[str, List[Any]], 
                   max_iterations: int = None, early_stopping: bool = True,
                   early_stopping_patience: int = 5) -> Dict[str, Any]:
        """Perform grid search for best parameters.
        
        Args:
            data (pd.DataFrame): Training data
            param_grid (Dict[str, List[Any]]): Dictionary of parameter ranges to search
            max_iterations (int, optional): Maximum number of iterations to perform
            early_stopping (bool): Whether to use early stopping
            early_stopping_patience (int): Number of iterations without improvement before stopping
            
        Returns:
            Dict containing best parameters, AIC, and model
        """
        logger.info("Starting grid search for best parameters")
        try:
            # Initialize tracking variables
            best_aic = float('inf')
            best_bic = float('inf')
            best_hqic = float('inf')
            best_params = None
            best_model = None
            results = []
            
            # Prepare data
            ts = self.prepare_data(data)
            
            # Calculate total combinations and limit if needed
            total_combinations = np.prod([len(values) for values in param_grid.values()])
            if max_iterations:
                total_combinations = min(total_combinations, max_iterations)
            
            # Helper function to evaluate a parameter combination
            def evaluate_combination(params):
                try:
                    # Create temporary config for this combination
                    temp_config = self.params.copy()
                    temp_config.update(params)
                    
                    # Create temporary model with this config
                    temp_model = TimeSeriesModel(model_type=self.model_type, **temp_config)
                    temp_model.initialize_model(ts)
                    fitted = temp_model.model.fit(maxiter=self.params.get('maxiter', 50), disp=False)
                    
                    # Create result dictionary
                    result = {
                        **params,
                        'aic': fitted.aic,
                        'bic': fitted.bic,
                        'hqic': fitted.hqic,
                        'converged': fitted.mle_retvals['converged'],
                        'iterations': fitted.mle_retvals['iterations']
                    }
                    
                    return result, fitted
                except Exception as e:
                    logger.warning(f"Failed combination {params}: {str(e)}")
                    return None, None
            
            # Helper function to check if we should continue searching
            def should_continue(current, no_improvement_count):
                if max_iterations and current >= max_iterations:
                    logger.info(f"Reached maximum iterations ({max_iterations})")
                    return False
                if early_stopping and no_improvement_count >= early_stopping_patience:
                    logger.info(f"Early stopping triggered after {no_improvement_count} iterations without improvement")
                    return False
                return True
            
            # Iterate through parameter combinations
            current = 0
            no_improvement_count = 0
            
            # Generate parameter combinations
            param_combinations = []
            
            if self.model_type == 'sarima':
                for p in param_grid.get('p', [1]):
                    for d in param_grid.get('d', [1]):
                        for q in param_grid.get('q', [1]):
                            for P in param_grid.get('P', [0]):
                                for D in param_grid.get('D', [0]):
                                    for Q in param_grid.get('Q', [0]):
                                        for s in param_grid.get('s', [1]):
                                            param_combinations.append({
                                                'p': p, 'd': d, 'q': q,
                                                'P': P, 'D': D, 'Q': Q, 's': s,
                                                'model_type': 'sarima'
                                            })
            else:
                for p in param_grid.get('p', [1]):
                    for d in param_grid.get('d', [1]):
                        for q in param_grid.get('q', [1]):
                            param_combinations.append({
                                'p': p, 'd': d, 'q': q,
                                'model_type': 'arima'
                            })
            
            # Limit combinations if needed
            if max_iterations and len(param_combinations) > max_iterations:
                param_combinations = param_combinations[:max_iterations]
            
            # Evaluate each combination
            for params in param_combinations:
                current += 1
                logger.info(f"Testing combination {current}/{min(len(param_combinations), total_combinations)}: {params}")
                
                result, fitted = evaluate_combination(params)
                if result is None:
                    continue
                
                results.append(result)
                
                # Update best model if current model has lower AIC
                if result['aic'] < best_aic:
                    logger.info(f"New best model found with AIC: {result['aic']}")
                    best_aic = result['aic']
                    best_bic = result['bic']
                    best_hqic = result['hqic']
                    best_params = {k: v for k, v in params.items() if k != 'model_type'}
                    best_model = fitted
                    no_improvement_count = 0
                else:
                    no_improvement_count += 1
                
                # Check if we should continue
                if not should_continue(current, no_improvement_count):
                    break
            
            # Sort results by AIC
            results.sort(key=lambda x: x['aic'])
            
            # Calculate convergence rate
            converged_count = sum(1 for r in results if r.get('converged', False))
            convergence_rate = converged_count / len(results) if results else 0
            
            return {
                'best_params': best_params,
                'best_aic': best_aic,
                'best_bic': best_bic,
                'best_hqic': best_hqic,
                'best_model': best_model,
                'all_results': results[:10],  # Return top 10 results
                'total_combinations_tested': current,
                'convergence_rate': convergence_rate,
                'param_grid': param_grid
            }
        except Exception as e:
            logger.error(f"Error in grid search: {str(e)}")
            raise 