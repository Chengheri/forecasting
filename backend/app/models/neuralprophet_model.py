from typing import Dict, Any, Optional, Tuple, Union, List, cast
import pandas as pd
import numpy as np
import torch
from neuralprophet import NeuralProphet
from prophet import Prophet
import optuna

from .prophet_model import ProphetModel
from ..utils.logger import Logger
from ..utils.trackers import ProphetTracker

logger = Logger()

class NeuralProphetModel(ProphetModel):
    """NeuralProphet time series model extending Prophet with neural network capabilities."""
    
    def __init__(
        self,
        config: Dict[str, Any],
        tracker: Optional[ProphetTracker] = None,    
    ):
        """Initialize NeuralProphetModel with parameters."""
        super().__init__(config, tracker)
        self.neural_model: Optional[NeuralProphet] = None
        
    def prepare_data(self, data: Union[pd.DataFrame, pd.Series, np.ndarray]) -> pd.DataFrame:
        """Prepare data for NeuralProphet model."""
        # Use the same data preparation as Prophet
        return super().prepare_data(data)
        
    def initialize_model(self) -> None:
        """Initialize the NeuralProphet model with data."""
        logger.info("Initializing model...")
        
        try:
            # Get NeuralProphet specific parameters
            np_params = {
                # Common Prophet parameters
                'seasonality_mode': self.config['model'].get('seasonality_mode'),
                'yearly_seasonality': self.config['model'].get('yearly_seasonality'),
                'weekly_seasonality': self.config['model'].get('weekly_seasonality'),
                'daily_seasonality': self.config['model'].get('daily_seasonality'),
                'growth': self.config['model'].get('growth'),
                'changepoints_range': self.config['model'].get('changepoint_range'),
                'n_changepoints': self.config['model'].get('n_changepoints'),
                
                # NeuralProphet specific parameters
                'learning_rate': self.config['model'].get('learning_rate'),
                'epochs': self.config['model'].get('epochs'),
                'batch_size': self.config['model'].get('batch_size'),
                'n_forecasts': self.config['model'].get('n_forecasts'),
                'n_lags': self.config['model'].get('n_lags'),
                'ar_reg': self.config['model'].get('ar_reg'),
                'normalize': self.config['model'].get('normalize'),
                'quantiles': self.config['model'].get('quantiles'),
                'future_regressors_model': self.config['model'].get('future_regressors_model'),
                'future_regressors_d_hidden': self.config['model'].get('future_regressors_d_hidden'),
                'future_regressors_num_hidden_layers': self.config['model'].get('future_regressors_num_hidden_layers'),
            }
            # Create NeuralProphet model
            self.neural_model = NeuralProphet(**np_params)
            # For compatibility with ProphetModel parent class
            self.model = cast(Optional[Prophet], self.neural_model)
            logger.info("NeuralProphet model initialized successfully")
            logger.debug(f"NeuralProphet parameters: {np_params}")
            
        except Exception as e:
            logger.error(f"Error initializing NeuralProphet model: {str(e)}")
            raise

    def predict(self, data: pd.DataFrame) -> Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Generate predictions using the fitted NeuralProphet model."""
        if self.neural_model is None:
            raise ValueError("Model must be trained before prediction")
        
        forecast = self.neural_model.predict(data)
        df_fc = self.neural_model.get_latest_forecast(forecast)
        print(df_fc)
        return df_fc

    def predict_into_future(self, data: pd.DataFrame, steps: int) -> Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Generate predictions using the fitted NeuralProphet model."""
        if self.neural_model is None:
            raise ValueError("Model must be trained before prediction")
        
        logger.info(f"Generating NeuralProphet predictions for {steps} steps")
        
        try:
            # NeuralProphet uses a different approach for creating future data
            # We need the last period of data to make predictions due to AR structure
            # Generate forecast
            future = self.neural_model.make_future_dataframe(
                df=data, 
                periods=steps,
                n_historic_predictions=False
            )
            
            logger.debug(f"Created future dataframe with shape: {future.shape}")
            forecast = self.neural_model.predict(future)
            
            # Extract predictions and confidence intervals, ensuring numpy array types
            predictions = np.asarray(forecast['yhat1'].values[-steps:], dtype=np.float64)
            
            logger.info("NeuralProphet predictions generated successfully")
            return predictions
            
        except Exception as e:
            logger.error(f"Error generating NeuralProphet predictions: {str(e)}")
            raise
    
    def fit(self, data: Union[pd.Series, pd.DataFrame, np.ndarray], params: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
        """Fit the NeuralProphet model to the provided data."""
        try:
            # Update parameters if provided
            if params:
                self.config['model'].update(params)
                self.initialize_model()
            
            if self.neural_model is None:
                raise ValueError("Model must be initialized before fitting")
            
            # Prepare data if needed
            if isinstance(data, (pd.Series, np.ndarray)) or (isinstance(data, pd.DataFrame) and ('y' not in data.columns or 'ds' not in data.columns)):
                prepared_data = self.prepare_data(data)
            else:
                prepared_data = data
            
            # Store training data for later use
            self._training_data = prepared_data.copy() if isinstance(prepared_data, pd.DataFrame) else prepared_data
            
            # Fit model
            logger.info("Fitting NeuralProphet model")
            metrics = self.neural_model.fit(prepared_data, **self.config['model'].get('fit_params', {}))
            train_metrics = {
                'mae': metrics['MAE'].min(),
                'rmse': metrics['RMSE'].min(),
            }
            self.fitted_model = self.neural_model
            # Convert numpy types to Python native types
            train_metrics = {k: float(v) for k, v in train_metrics.items()}
            
            logger.info("Model training completed successfully")
            logger.info(f"Training metrics: {train_metrics}")
            
            return train_metrics
            
        except Exception as e:
            logger.error(f"Error in model training: {str(e)}")
            raise
    
    def fit_with_optuna(self, data: Union[pd.Series, pd.DataFrame, np.ndarray], n_trials: int = 100, timeout: int = 600) -> Dict[str, Any]:
        """Optimize NeuralProphet hyperparameters using Optuna."""
        if data is None:
            raise ValueError("Data cannot be None")
            
        logger.info(f"Starting hyperparameter optimization with {n_trials} trials and {timeout}s timeout")
        
        def objective(trial):
            """Optimization objective function."""
            params = {
                # Prophet Parameters
                'seasonality_mode': trial.suggest_categorical('seasonality_mode', ['additive', 'multiplicative']),
                
                # NeuralProphet specific parameters
                'learning_rate': trial.suggest_loguniform('learning_rate', 0.0001, 0.1),
                'epochs': trial.suggest_int('epochs', 50, 200),
                'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64, 128, 256, 512]),
                'n_changepoints': trial.suggest_int('n_changepoints', 10, 50),
                'n_forecasts': trial.suggest_int('n_forecasts', 1, 5),
                'n_lags': trial.suggest_int('n_lags', 24, 72),
                'ar_reg': trial.suggest_float('ar_reg', 0.01, 1.0),
                'normalize': trial.suggest_categorical('normalize', ['auto', 'minmax', 'standardize']),
                'future_regressors_model': trial.suggest_categorical('future_regressors_model', ['linear', 'neural_nets']),
                'future_regressors_d_hidden': trial.suggest_int('future_regressors_d_hidden', 1, 10),
                'future_regressors_num_hidden_layers': trial.suggest_int('future_regressors_num_hidden_layers', 1, 3),
            }
            
            try:
                # Create and fit model with trial parameters
                self.config['model'].update(params)
                self.initialize_model()
                model_metrics = self.fit(data)
                return float(model_metrics['rmse'])  # Convert to Python float
            except Exception as e:
                logger.warning(f"Trial failed: {str(e)}")
                return float('inf')
        
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=n_trials, timeout=timeout)
        
        best_params = study.best_params
        logger.info(f"Best parameters found: {best_params}")
        
        # Update model with best parameters and fit
        logger.info("Updating model with best parameters and fitting...")
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
                'optimization.best_value': float(study.best_value),
                **{f'optimization.best_params.{k}': float(v) if isinstance(v, (np.floating, np.integer)) else v 
                   for k, v in best_params.items()}
            })
        
        logger.info("Optuna optimization completed successfully")
        return hyperparameter_optimization_results
    
    def save(self, path: str) -> None:
        """Save the model to disk."""
        logger.info(f"Saving NeuralProphet model to {path}")
        try:
            import os
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, 'wb') as f:
                torch.save(self.neural_model, f)
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            raise
    
    def load(self, path: str) -> None:
        """Load the model from disk."""
        logger.info(f"Loading NeuralProphet model from {path}")
        try:
            with open(path, 'rb') as f:
                self.neural_model = torch.load(f)
            self.model = cast(Optional[Prophet], self.neural_model)
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise 