import numpy as np
from typing import Dict, Union, List, Tuple
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from ..utils.logger import Logger

logger = Logger()

class ForecastingMetrics:
    @staticmethod
    def calculate_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate Mean Absolute Percentage Error."""
        logger.debug("Calculating MAPE")
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        mask = y_true != 0
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
        logger.debug(f"MAPE: {mape}%")
        return mape
    
    @staticmethod
    def calculate_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate Root Mean Square Error."""
        logger.debug("Calculating RMSE")
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        logger.debug(f"RMSE: {rmse}")
        return rmse
    
    @staticmethod
    def calculate_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate Mean Absolute Error."""
        logger.debug("Calculating MAE")
        mae = mean_absolute_error(y_true, y_pred)
        logger.debug(f"MAE: {mae}")
        return mae
    
    @staticmethod
    def calculate_mase(y_true: np.ndarray, y_pred: np.ndarray, y_train: np.ndarray, 
                      seasonal_period: int = 24) -> float:
        """Calculate Mean Absolute Scaled Error."""
        n = len(y_train)
        d = np.abs(np.diff(y_train, seasonal_period)).sum() / (n - seasonal_period)
        errors = np.abs(y_true - y_pred)
        return errors.mean() / d
    
    @staticmethod
    def calculate_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate R-squared score."""
        logger.debug("Calculating R²")
        r2 = r2_score(y_true, y_pred)
        logger.debug(f"R²: {r2}")
        return r2
    
    @staticmethod
    def calculate_directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate directional accuracy of the forecast."""
        logger.debug("Calculating direction accuracy")
        direction_true = np.diff(y_true) > 0
        direction_pred = np.diff(y_pred) > 0
        direction_accuracy = np.mean(direction_true == direction_pred) * 100
        logger.debug(f"Direction accuracy: {direction_accuracy}%")
        return direction_accuracy
    
    @staticmethod
    def calculate_all_metrics(y_true: np.ndarray, y_pred: np.ndarray, 
                            y_train: np.ndarray = None) -> Dict[str, float]:
        """Calculate all available metrics."""
        logger.info("Calculating performance metrics")
        try:
            metrics = {
                "mape": ForecastingMetrics.calculate_mape(y_true, y_pred),
                "rmse": ForecastingMetrics.calculate_rmse(y_true, y_pred),
                "mae": ForecastingMetrics.calculate_mae(y_true, y_pred),
                "r2": ForecastingMetrics.calculate_r2(y_true, y_pred),
                "directional_accuracy": ForecastingMetrics.calculate_directional_accuracy(y_true, y_pred)
            }
            
            if y_train is not None:
                metrics["mase"] = ForecastingMetrics.calculate_mase(y_true, y_pred, y_train)
            
            logger.info(f"Calculated metrics: {metrics}")
            return metrics
        except Exception as e:
            logger.error(f"Error calculating metrics: {str(e)}")
            raise
    
    @staticmethod
    def calculate_confidence_intervals(y_pred: np.ndarray, 
                                    std_dev: np.ndarray, 
                                    confidence_level: float = 0.95) -> Dict[str, np.ndarray]:
        """Calculate confidence intervals for the forecast."""
        logger.info(f"Calculating confidence intervals with {confidence_level*100}% confidence")
        try:
            z_score = {
                0.90: 1.645,
                0.95: 1.96,
                0.99: 2.576
            }.get(confidence_level, 1.96)
            
            lower_bound = y_pred - z_score * std_dev
            upper_bound = y_pred + z_score * std_dev
            
            logger.info("Confidence intervals calculated successfully")
            return {
                "lower_bound": lower_bound,
                "upper_bound": upper_bound
            }
        except Exception as e:
            logger.error(f"Error calculating confidence intervals: {str(e)}")
            raise
    
    @staticmethod
    def evaluate_residuals(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Union[float, bool]]:
        """Evaluate forecast residuals for various properties."""
        residuals = y_true - y_pred
        
        # Basic statistics
        mean_residual = np.mean(residuals)
        std_residual = np.std(residuals)
        
        # Normality test (using simple skewness and kurtosis test)
        skewness = np.mean((residuals - mean_residual) ** 3) / (std_residual ** 3)
        kurtosis = np.mean((residuals - mean_residual) ** 4) / (std_residual ** 4) - 3
        
        # Autocorrelation test (lag-1)
        autocorr = np.corrcoef(residuals[:-1], residuals[1:])[0, 1]
        
        return {
            "mean_residual": mean_residual,
            "std_residual": std_residual,
            "skewness": skewness,
            "kurtosis": kurtosis,
            "autocorrelation": autocorr,
            "is_normal": abs(skewness) < 0.5 and abs(kurtosis) < 0.5,
            "is_independent": abs(autocorr) < 0.2
        } 