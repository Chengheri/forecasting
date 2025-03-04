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
    def __init__(self, config: Dict[str, Any]):
        logger.info(f"Initializing TimeSeriesModel with config: {config}")
        self.config = config
        self.model = None
        self.model_type = config.get('model_type', 'arima')  # 'arima' or 'sarima'
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
    
    def train(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Train ARIMA or SARIMA model."""
        logger.info("Starting model training")
        try:
            ts = self.prepare_data(data)
            
            if self.model_type == 'arima':
                # ARIMA parameters
                p = self.config.get('p', 1)  # AR order
                d = self.config.get('d', 1)  # Difference order
                q = self.config.get('q', 1)  # MA order
                
                logger.info(f"Training ARIMA model with parameters: p={p}, d={d}, q={q}")
                self.model = ARIMA(ts, order=(p, d, q))
                
            else:  # SARIMA
                # SARIMA parameters
                p = self.config.get('p', 1)  # AR order
                d = self.config.get('d', 1)  # Difference order
                q = self.config.get('q', 1)  # MA order
                P = self.config.get('P', 1)  # Seasonal AR order
                D = self.config.get('D', 1)  # Seasonal difference order
                Q = self.config.get('Q', 1)  # Seasonal MA order
                s = self.config.get('s', 24)  # Seasonal period
                
                logger.info(f"Training SARIMA model with parameters: p={p}, d={d}, q={q}, P={P}, D={D}, Q={Q}, s={s}")
                self.model = SARIMAX(ts,
                                   order=(p, d, q),
                                   seasonal_order=(P, D, Q, s))
            
            # Fit the model
            self.fitted_model = self.model.fit()
            results = {
                "aic": self.fitted_model.aic,
                "bic": self.fitted_model.bic,
                "hqic": self.fitted_model.hqic,
                "model_type": self.model_type
            }
            logger.info(f"Model training completed successfully. AIC: {results['aic']}, BIC: {results['bic']}, HQIC: {results['hqic']}")
            return results
        except Exception as e:
            logger.error(f"Error training model: {str(e)}")
            raise ValueError(f"Error fitting model: {str(e)}")
    
    def predict(self, data: Union[int, pd.DataFrame], last_data: pd.DataFrame = None) -> Tuple[np.ndarray, np.ndarray]:
        """Generate forecasts and confidence intervals.
        
        Args:
            data: Either the number of steps to forecast or a test DataFrame
            last_data: Optional DataFrame containing the last known data points
            
        Returns:
            Tuple of predicted mean and confidence intervals
        """
        try:
            if self.fitted_model is None:
                logger.error("Model must be trained before prediction")
                raise ValueError("Model must be trained before prediction")
            
            # Determine number of steps
            if isinstance(data, pd.DataFrame):
                steps = len(data)
                logger.info(f"Generating predictions for test data with {steps} steps")
            else:
                steps = data
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
    
    def evaluate(self, actual: np.ndarray, predicted: np.ndarray) -> Dict[str, float]:
        """Evaluate model performance using comprehensive metrics.
        
        Args:
            actual: Array of actual values
            predicted: Array of predicted values
            
        Returns:
            Dictionary of evaluation metrics including:
            - Basic metrics: RMSE, MAE, MAPE, R², directional accuracy
            - Residuals analysis: mean, std dev, skewness, kurtosis, autocorrelation
            - Residuals properties: normality test, independence test
        """
        logger.info("Evaluating model performance")
        try:
            # Calculate all available metrics using ForecastingMetrics
            metrics = ForecastingMetrics.calculate_all_metrics(actual, predicted)
            
            # Add complete residuals analysis
            residuals_metrics = ForecastingMetrics.evaluate_residuals(actual, predicted)
            metrics.update({
                # Basic residuals statistics
                "residuals_mean": residuals_metrics["mean_residual"],
                "residuals_std": residuals_metrics["std_residual"],
                # Distribution characteristics
                "residuals_skewness": residuals_metrics["skewness"],
                "residuals_kurtosis": residuals_metrics["kurtosis"],
                # Time series properties
                "residuals_autocorrelation": residuals_metrics["autocorrelation"],
                # Statistical tests results
                "residuals_normal": residuals_metrics["is_normal"],
                "residuals_independent": residuals_metrics["is_independent"]
            })
            
            logger.info(f"Model evaluation completed. Metrics: {metrics}")
            return metrics
        except Exception as e:
            logger.error(f"Error evaluating model: {str(e)}")
            raise
    
    def save_model(self, path: str):
        """Save the trained model."""
        logger.info(f"Saving model to {path}")
        try:
            if self.fitted_model is None:
                logger.error("Model must be trained before saving")
                raise ValueError("Model must be trained before saving")
            
            model_data = {
                'fitted_model': self.fitted_model,
                'config': self.config,
                'model_type': self.model_type
            }
            joblib.dump(model_data, path)
            logger.info("Model saved successfully")
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            raise
    
    def load_model(self, path: str):
        """Load a trained model."""
        logger.info(f"Loading model from {path}")
        try:
            model_data = joblib.load(path)
            self.fitted_model = model_data['fitted_model']
            self.config = model_data['config']
            self.model_type = model_data['model_type']
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
    
    @staticmethod
    def grid_search(data: pd.DataFrame, param_grid: Dict[str, List[Any]]) -> Dict[str, Any]:
        """Perform grid search for best parameters."""
        logger.info("Starting grid search for best parameters")
        try:
            best_aic = float('inf')
            best_params = None
            best_model = None
            
            ts = data.set_index('timestamp')['consumption']
            
            total_combinations = np.prod([len(values) for values in param_grid.values()])
            current = 0
            
            for p in param_grid.get('p', [1]):
                for d in param_grid.get('d', [1]):
                    for q in param_grid.get('q', [1]):
                        if 'P' in param_grid:  # SARIMA
                            for P in param_grid['P']:
                                for D in param_grid['D']:
                                    for Q in param_grid['Q']:
                                        for s in param_grid['s']:
                                            current += 1
                                            logger.debug(f"Testing combination {current}/{total_combinations}")
                                            
                                            try:
                                                model = SARIMAX(ts,
                                                              order=(p, d, q),
                                                              seasonal_order=(P, D, Q, s))
                                                fitted = model.fit()
                                                
                                                if fitted.aic < best_aic:
                                                    best_aic = fitted.aic
                                                    best_params = {
                                                        'p': p, 'd': d, 'q': q,
                                                        'P': P, 'D': D, 'Q': Q, 's': s
                                                    }
                                                    best_model = fitted
                                                    logger.debug(f"New best model found with AIC: {best_aic}")
                                            except Exception as e:
                                                logger.debug(f"Failed combination: {str(e)}")
                                                continue
                        else:  # ARIMA
                            current += 1
                            logger.debug(f"Testing combination {current}/{total_combinations}")
                            
                            try:
                                model = ARIMA(ts, order=(p, d, q))
                                fitted = model.fit()
                                
                                if fitted.aic < best_aic:
                                    best_aic = fitted.aic
                                    best_params = {'p': p, 'd': d, 'q': q}
                                    best_model = fitted
                                    logger.debug(f"New best model found with AIC: {best_aic}")
                            except Exception as e:
                                logger.debug(f"Failed combination: {str(e)}")
                                continue
            
            logger.info(f"Grid search completed. Best AIC: {best_aic}")
            return {
                'best_params': best_params,
                'best_aic': best_aic,
                'best_model': best_model
            }
        except Exception as e:
            logger.error(f"Error in grid search: {str(e)}")
            raise 