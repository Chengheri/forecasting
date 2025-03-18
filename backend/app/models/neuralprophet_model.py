from typing import Dict, Any, Optional, Tuple, Union, List, cast, TYPE_CHECKING
import pandas as pd
import numpy as np
import torch
from neuralprophet import NeuralProphet
from prophet import Prophet
import optuna
import joblib
import os
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from pathlib import Path

from .prophet_model import ProphetModel
from ..utils.logger import Logger
from ..utils.trackers import ProphetTracker

logger = Logger()

# Type checking imports to avoid circular imports
if TYPE_CHECKING:
    from numpy.typing import NDArray
    
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
        
        future = self.neural_model.make_future_dataframe(data, periods=0, n_historic_predictions=len(data))
        forecast = self.neural_model.predict(future)
        lower_bound = forecast['yhat1 5.0%'].values[-len(data):]
        upper_bound = forecast['yhat1 95.0%'].values[-len(data):]
        predictions = forecast['yhat1'].values[-len(data):]

        logger.info("NeuralProphet predictions generated successfully")
        return predictions, (lower_bound, upper_bound)

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

    def cross_validation(
        self, 
        data: Union[pd.Series, pd.DataFrame, np.ndarray], 
        fold_pct: float = 0.1, 
        horizon_pct: float = 0.2,
        k_folds: int = 5,
        fold_overlap_pct: float = 0.0,
        save_plots: bool = False,
        plots_dir: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Perform time series cross-validation for NeuralProphet model.
        
        Args:
            data: The time series data to be used for cross-validation
            fold_pct: Percentage of data to use in each fold as validation
            horizon_pct: Percentage of data to use for prediction horizon
            k_folds: Number of folds to create
            fold_overlap_pct: Percentage of overlap between folds (0.0 = no overlap)
            save_plots: Whether to save the cross-validation plots
            plots_dir: Directory to save plots to (if save_plots is True)
            
        Returns:
            Dictionary containing cross-validation metrics and results summary
        """
        logger.info(f"Starting cross-validation with {k_folds} folds")
        
        try:
            # Prepare data if needed
            if isinstance(data, (pd.Series, np.ndarray)) or (isinstance(data, pd.DataFrame) and ('y' not in data.columns or 'ds' not in data.columns)):
                prepared_data = self.prepare_data(data)
            else:
                prepared_data = data
                
            # Store training data for later use
            self._training_data = prepared_data.copy() if isinstance(prepared_data, pd.DataFrame) else prepared_data
            
            # Initialize the model if not already initialized
            if self.neural_model is None:
                self.initialize_model()
                
            if self.neural_model is None:
                raise ValueError("Failed to initialize NeuralProphet model")
                
            # Calculate fold size and horizon size
            n_samples = len(prepared_data)
            fold_size = int(n_samples * fold_pct)
            horizon = int(n_samples * horizon_pct)
            
            # Ensure we have enough data for cross-validation
            if n_samples < (horizon + fold_size) * k_folds:
                logger.warning(f"Not enough data for {k_folds} folds. Reducing number of folds.")
                k_folds = max(1, n_samples // (horizon + fold_size))
                
            logger.info(f"Performing {k_folds} fold cross-validation with horizon of {horizon} steps")
            
            # Manual cross-validation if NeuralProphet doesn't have built-in cross_validation
            cv_results = []
            metrics_all = []
            
            # Split data into folds
            data_length = len(prepared_data)
            fold_indices = []
            
            # Create fold indices
            for i in range(k_folds):
                # Calculate cutoff indices
                cutoff = data_length - (i + 1) * horizon
                if cutoff < fold_size:  # Not enough data for another fold
                    break
                fold_indices.append(cutoff)
            
            # Reverse order to start with earliest fold
            fold_indices.reverse()
            
            # For each fold, train and predict
            for fold, cutoff in enumerate(fold_indices):
                # Split data
                train_df = prepared_data.iloc[:cutoff].copy()
                val_df = prepared_data.iloc[cutoff:cutoff + horizon].copy()
                
                # Initialize and fit model on training data
                if fold > 0:  # Re-initialize model for each fold
                    self.initialize_model()
                    
                try:
                    # Fit model on training data
                    self.neural_model.fit(train_df, **self.config['model'].get('fit_params', {}))
                    
                    # Create future dataframe for prediction
                    future = self.neural_model.make_future_dataframe(
                        df=train_df, 
                        periods=horizon,
                        n_historic_predictions=True
                    )
                    
                    # Predict
                    forecast = self.neural_model.predict(future)
                    
                    # Calculate metrics for this fold
                    fold_metrics = self._calculate_backtest_metrics(val_df, forecast.iloc[-horizon:])
                    
                    # Add fold information
                    fold_metrics['fold'] = fold
                    fold_metrics['cutoff'] = cutoff
                    fold_metrics['horizon'] = horizon
                    
                    metrics_all.append(fold_metrics)
                    
                    # Add to CV results
                    result = {
                        'fold': fold,
                        'cutoff': cutoff,
                        'train_df': train_df,
                        'val_df': val_df,
                        'forecast': forecast,
                        'metrics': fold_metrics
                    }
                    cv_results.append(result)
                    
                    logger.info(f"Fold {fold} complete with RMSE: {fold_metrics.get('rmse', 'N/A')}")
                    
                except Exception as e:
                    logger.warning(f"Error in fold {fold}: {str(e)}")
                    continue
            
            # Calculate summary metrics across all folds
            metrics_df = pd.DataFrame(metrics_all)
            
            # Convert metrics to dictionary format
            metrics_by_fold = {}
            for fold in range(len(fold_indices)):
                fold_metrics = metrics_df[metrics_df['fold'] == fold]
                
                # Skip empty folds
                if fold_metrics.empty:
                    continue
                    
                # Get metrics for this fold
                fold_summary = {}
                for metric in ['mse', 'rmse', 'mae', 'mape', 'r2']:
                    if metric in fold_metrics.columns:
                        fold_summary[metric] = float(fold_metrics[metric].iloc[0])
                
                metrics_by_fold[f"fold_{fold}"] = fold_summary
                
            # Calculate overall metrics
            summary_metrics = {}
            for metric in ['mse', 'rmse', 'mae', 'mape', 'r2']:
                if metric in metrics_df.columns:
                    summary_metrics[metric] = float(metrics_df[metric].mean())
            
            # Create plots if requested
            if save_plots and plots_dir is not None:
                plots_path = Path(plots_dir)
                plots_path.mkdir(parents=True, exist_ok=True)
                
                # Plot actual vs predicted for each fold
                for fold, result in enumerate(cv_results):
                    if 'forecast' not in result or 'val_df' not in result:
                        continue
                        
                    plt.figure(figsize=(12, 6))
                    
                    # Get actual and predicted data
                    val_df = result['val_df']
                    forecast = result['forecast']
                    
                    # Plot actual vs predicted
                    plt.plot(val_df['ds'], val_df['y'], 'b-', label='Actual')
                    plt.plot(forecast.iloc[-len(val_df):]['ds'], 
                             forecast.iloc[-len(val_df):]['yhat1'], 'r-', label='Predicted')
                    
                    plt.title(f'Cross-Validation Fold {fold}: Actual vs Predicted')
                    plt.xlabel('Date')
                    plt.ylabel('Value')
                    plt.legend()
                    plt.tight_layout()
                    plt.savefig(plots_path / f"cv_fold_{fold}.png")
                    plt.close()
                
                # Plot summary metrics
                plt.figure(figsize=(10, 6))
                metrics_df.plot(x='fold', y=['rmse', 'mae'], kind='bar', ax=plt.gca())
                plt.title('Cross-Validation Metrics by Fold')
                plt.tight_layout()
                plt.savefig(plots_path / "cv_metrics.png")
                plt.close()
                
                logger.info(f"Cross-validation plots saved to {plots_dir}")
            
            # Save the trained model - we consider the model from the last fold as final
            if cv_results:
                self._model_is_fitted = True
            
            # Log results to tracker if available
            if self.tracker and metrics_by_fold:
                logger.debug("Logging cross-validation results to tracker")
                self.tracker.log_params_safely({
                    'cross_validation.folds': len(fold_indices),
                    'cross_validation.horizon': horizon,
                    'cross_validation.fold_size': fold_size,
                    **{f'cross_validation.metrics.{k}': v for k, v in summary_metrics.items()}
                })
                
                # Also log the best fold metrics
                best_fold = min(metrics_by_fold.keys(), 
                                key=lambda f: metrics_by_fold[f].get('rmse', float('inf')) 
                                if metrics_by_fold[f] else float('inf'))
                
                if best_fold in metrics_by_fold and metrics_by_fold[best_fold]:
                    self.tracker.log_params_safely({
                        'cross_validation.best_fold': best_fold,
                        **{f'cross_validation.best_fold.{k}': v for k, v in metrics_by_fold[best_fold].items()}
                    })
            
            cv_results_dict = {
                'metrics_by_fold': metrics_by_fold,
                'summary_metrics': summary_metrics,
                'cv_results': cv_results,
                'metrics_df': metrics_df,
                'k_folds': len(fold_indices),
                'horizon': horizon,
                'fold_size': fold_size
            }
            
            logger.info(f"Cross-validation completed successfully with summary metrics: {summary_metrics}")
            return cv_results_dict
            
        except Exception as e:
            logger.error(f"Error in cross-validation: {str(e)}")
            raise
            
    def cross_validation_backtest(
        self,
        data: Union[pd.Series, pd.DataFrame, np.ndarray],
        test_percentage: float = 0.2,
        periods: Optional[int] = None,
        horizon: Optional[int] = None,
        initial: Optional[int] = None,
        period: Optional[int] = None,
        save_plots: bool = False,
        plots_dir: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Perform backtest cross-validation with expanding window approach.
        
        This method implements an expanding window approach where we:
        1. Start with a small initial training set
        2. Make predictions for the next horizon steps
        3. Expand the training window to include more data
        4. Repeat until we've covered the entire dataset
        
        Args:
            data: The time series data to use for backtesting
            test_percentage: Percentage of data to use as test set
            periods: Number of forecast steps for each window
            horizon: Alias for periods (for compatibility)
            initial: Initial number of samples in the first training window
            period: Number of samples to expand the window by in each iteration
            save_plots: Whether to save the backtest plots
            plots_dir: Directory to save plots to (if save_plots is True)
            
        Returns:
            Dictionary containing backtest metrics and results
        """
        logger.info("Starting cross-validation backtest")
        
        try:
            # Prepare data if needed
            if isinstance(data, (pd.Series, np.ndarray)) or (isinstance(data, pd.DataFrame) and ('y' not in data.columns or 'ds' not in data.columns)):
                prepared_data = self.prepare_data(data)
            else:
                prepared_data = data
                
            # Store training data for later use
            self._training_data = prepared_data.copy() if isinstance(prepared_data, pd.DataFrame) else prepared_data
            
            # Initialize the model if not already initialized
            if self.neural_model is None:
                self.initialize_model()
                
            if self.neural_model is None:
                raise ValueError("Failed to initialize NeuralProphet model")
                
            # Calculate test size
            n_samples = len(prepared_data)
            test_size = int(n_samples * test_percentage)
            train_size = n_samples - test_size
            
            # Set default values for backtest parameters if not provided
            if periods is None and horizon is None:
                periods = max(30, int(test_size * 0.2))  # Default to 20% of test data or 30 samples
            elif horizon is not None and periods is None:
                periods = horizon
                
            if initial is None:
                initial = max(30, int(train_size * 0.5))  # Default to 50% of training data or 30 samples
                
            if period is None:
                period = max(1, int(periods * 0.5))  # Default to 50% of the forecast horizon
                
            logger.info(f"Backtest configuration: initial={initial}, period={period}, horizon={periods}")
            
            # Perform backtesting using NeuralProphet's built-in functionality
            backtest_df = self.neural_model.predict(df=prepared_data)
            
            # When model is fitted during backtesting, we consider it trained
            self._model_is_fitted = True
            
            # Calculate metrics on test data
            test_df = prepared_data.iloc[-test_size:]
            backtest_metrics = self._calculate_backtest_metrics(test_df, backtest_df.iloc[-test_size:])
            
            # Create plots if requested
            if save_plots and plots_dir is not None:
                plots_path = Path(plots_dir)
                plots_path.mkdir(parents=True, exist_ok=True)
                
                plt.figure(figsize=(12, 6))
                
                # Plot actual vs predicted
                plt.plot(test_df['ds'], test_df['y'], 'b-', label='Actual')
                plt.plot(backtest_df.iloc[-test_size:]['ds'], 
                         backtest_df.iloc[-test_size:]['yhat1'], 'r-', label='Predicted')
                
                # Add confidence intervals if available
                if 'yhat1_lower' in backtest_df.columns and 'yhat1_upper' in backtest_df.columns:
                    plt.fill_between(
                        backtest_df.iloc[-test_size:]['ds'],
                        backtest_df.iloc[-test_size:]['yhat1_lower'],
                        backtest_df.iloc[-test_size:]['yhat1_upper'],
                        color='r', alpha=0.2, label='95% CI'
                    )
                
                plt.title('Backtest: Actual vs Predicted')
                plt.xlabel('Date')
                plt.ylabel('Value')
                plt.legend()
                plt.tight_layout()
                plt.savefig(plots_path / "backtest_prediction.png")
                plt.close()
                
                logger.info(f"Backtest plots saved to {plots_dir}")
            
            # Log results to tracker if available
            if self.tracker:
                logger.debug("Logging backtest results to tracker")
                self.tracker.log_params_safely({
                    'backtest.test_size': test_size,
                    'backtest.initial': initial,
                    'backtest.period': period,
                    'backtest.horizon': periods,
                    **{f'backtest.metrics.{k}': v for k, v in backtest_metrics.items()}
                })
            
            backtest_results = {
                'backtest_metrics': backtest_metrics,
                'backtest_df': backtest_df,
                'test_size': test_size,
                'initial': initial,
                'period': period,
                'horizon': periods
            }
            
            logger.info(f"Backtest completed successfully with metrics: {backtest_metrics}")
            return backtest_results
            
        except Exception as e:
            logger.error(f"Error in backtest: {str(e)}")
            raise

    def _calculate_backtest_metrics(self, actual_df: pd.DataFrame, forecast_df: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate metrics for backtest evaluation.
        
        Args:
            actual_df: DataFrame containing actual values
            forecast_df: DataFrame containing predicted values
            
        Returns:
            Dictionary of metrics
        """
        # Extract actual and predicted values
        actual = actual_df['y'].values
        predicted = forecast_df['yhat1'].values
        
        # Handle case where forecast_df might be a numpy array
        if isinstance(forecast_df, np.ndarray):
            # Assume it's already the predicted values
            predicted = forecast_df
            
            # Adjust length to match actual if needed
            min_length = min(len(actual), len(predicted))
            actual = actual[:min_length]
            predicted = predicted[:min_length]
        
        # Ensure same length for DataFrame case
        else:
            min_length = min(len(actual), len(predicted))
            actual = actual[:min_length]
            predicted = predicted[:min_length]
        
        metrics = self._calculate_metrics(actual, predicted)

        return metrics
            