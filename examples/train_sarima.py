import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

import pandas as pd
import numpy as np
from datetime import datetime
import mlflow
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import acf, pacf
from backend.app.models.arima_model import TimeSeriesModel
from backend.app.utils.trackers import ARIMATracker
from backend.app.utils.logger import Logger
from backend.app.utils.preprocessing import DataPreprocessor
from backend.app.utils.advanced_preprocessing import AdvancedPreprocessor
from backend.app.utils.analysis_utils import (
    analyze_time_series,
    get_suggested_parameters,
    analyze_model_results,
    plot_actual_vs_predicted,
    plot_residuals_analysis,
    plot_metrics_over_time,
    plot_seasonal_decomposition
)
from backend.app.utils.decorators import log_execution_time
import json
from statsmodels.tsa.statespace.sarimax import SARIMAX
import joblib
from typing import Dict, Any

# Initialize logger
logger = Logger()

def create_model_summary(initial_config: dict, train_data: pd.DataFrame, test_data: pd.DataFrame,
                      model_params: dict, metrics: dict, analysis_results: dict,
                      stationarity_results: dict, model_path: str, analysis_path: str,
                      grid_search_results: dict = None, preprocessor=None, use_grid_search: bool = False) -> dict:
    """Create a comprehensive model summary with consistent structure.
    
    Args:
        initial_config: Initial configuration dictionary
        train_data: Training data
        test_data: Test data
        model_params: Model parameters
        metrics: Model metrics
        analysis_results: Analysis results
        stationarity_results: Stationarity analysis results
        model_path: Path where model is saved
        analysis_path: Path where analysis results are saved
        grid_search_results: Optional grid search results
        preprocessor: Optional preprocessor object
        use_grid_search: Whether grid search was used for parameter selection
        
    Returns:
        Dictionary containing model summary
    """
    # Convert numpy numbers to native Python types
    def convert_to_native_types(obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32,
                          np.int64, np.uint8, np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, dict):
            return {key: convert_to_native_types(value) for key, value in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert_to_native_types(item) for item in obj]
        return obj

    model_summary = {
        'initial_configuration': initial_config,
        'training_details': {
            'train_samples': int(len(train_data)),
            'test_samples': int(len(test_data)),
            'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'parameter_selection_method': 'grid_search' if use_grid_search else 'suggested_parameters'
        },
        'best_model': {
            'parameters': convert_to_native_types(model_params),
            'performance': {
                'aic': float(metrics.get('aic', 0)),
                'bic': float(metrics.get('bic', 0)),
                'hqic': float(metrics.get('hqic', 0)),
                'metrics': {
                    'rmse': float(analysis_results.get('rmse', 0)),
                    'mae': float(analysis_results.get('mae', 0)),
                    'mape': float(analysis_results.get('mape', 0)),
                    'r2': float(analysis_results.get('r2', 0)),
                    'directional_accuracy': float(analysis_results.get('directional_accuracy', 0))
                },
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

def handle_post_training(model, model_params: dict, metrics: dict, train_data: pd.DataFrame, 
                       test_data: pd.DataFrame, initial_config: dict, stationarity_results: dict,
                       tracker: ARIMATracker, start_time: datetime,
                       grid_search_results: dict = None, preprocessor=None, use_grid_search: bool = False) -> None:
    """Handle all post-training steps including predictions, evaluation, and saving results."""
    # Generate predictions
    logger.info(f"Generating predictions for test data with {len(test_data)} steps")
    if isinstance(model, TimeSeriesModel):
        mean_forecast, confidence_intervals = model.predict(len(test_data))
    else:
        # For fitted SARIMAX model
        forecast = model.get_forecast(steps=len(test_data))
        mean_forecast = forecast.predicted_mean
        conf_int = forecast.conf_int()
        confidence_intervals = (conf_int.iloc[:, 0], conf_int.iloc[:, 1])
    
    # Create model-specific directory name
    model_name = f"sarima_p{model_params['p']}_d{model_params['d']}_q{model_params['q']}_P{model_params['P']}_D{model_params['D']}_Q{model_params['Q']}_s{model_params['s']}"
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Create model path and analysis paths
    model_path = os.path.join("data/models", model_name, timestamp)
    analysis_path = os.path.join(model_path, "analysis")
    
    # Create directories
    os.makedirs(os.path.join(model_path, "model"), exist_ok=True)
    os.makedirs(analysis_path, exist_ok=True)
    
    # Save model
    model_file_path = os.path.join(model_path, "model", "sarima_model.joblib")
    if isinstance(model, TimeSeriesModel):
        model.save_model(model_file_path)
    else:
        model_data = {
            'fitted_model': model,
            'config': model_params,
            'model_type': 'sarima'
        }
        joblib.dump(model_data, model_file_path)
        logger.info(f"Model saved successfully")
    
    # Generate and save analysis plots
    analysis_results = analyze_model_results(
        test_data[initial_config['model']['target_column']] if isinstance(test_data[initial_config['model']['target_column']], np.ndarray) else test_data[initial_config['model']['target_column']].values,
        mean_forecast,
        test_data.index,
        confidence_intervals,
        analysis_path
    )
    
    # Log model parameters and metrics using the tracker
    try:
        # Log model parameters with proper prefixes
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
        tracker.log_params_safely(model_params_with_prefix)
        
        # Log training metrics with proper prefixes
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
        tracker.log_metrics_safely(training_metrics)
        
        # Log model diagnostics with proper prefixes
        diagnostic_metrics = {
            'metrics.diagnostics.residuals_mean': analysis_results.get('residuals_mean', 0),
            'metrics.diagnostics.residuals_std': analysis_results.get('residuals_std', 0),
            'metrics.diagnostics.residuals_skewness': analysis_results.get('residuals_skewness', 0),
            'metrics.diagnostics.residuals_kurtosis': analysis_results.get('residuals_kurtosis', 0),
            'metrics.diagnostics.residuals_autocorrelation': analysis_results.get('residuals_autocorrelation', 0),
            'metrics.diagnostics.residuals_normal': analysis_results.get('residuals_normal', False),
            'metrics.diagnostics.residuals_independent': analysis_results.get('residuals_independent', False)
        }
        tracker.log_metrics_safely(diagnostic_metrics)
        
        # Log forecast metrics with proper prefixes
        forecast_metrics = {
            'metrics.forecast.rmse': analysis_results['rmse'],
            'metrics.forecast.mae': analysis_results['mae'],
            'metrics.forecast.mape': analysis_results['mape'],
            'metrics.forecast.r2': analysis_results.get('r2', 0)
        }
        tracker.log_metrics_safely(forecast_metrics)
        
        # Log stationarity analysis results
        stationarity_metrics = {
            'stationarity.adf_test.statistic': stationarity_results['adf_test']['test_statistic'],
            'stationarity.adf_test.pvalue': stationarity_results['adf_test']['pvalue'],
            'stationarity.adf_test.is_stationary': stationarity_results['adf_test']['is_stationary'],
            'stationarity.kpss_test.statistic': stationarity_results['kpss_test']['test_statistic'],
            'stationarity.kpss_test.pvalue': stationarity_results['kpss_test']['pvalue'],
            'stationarity.kpss_test.is_stationary': stationarity_results['kpss_test']['is_stationary'],
            'stationarity.overall.is_stationary': stationarity_results['overall_assessment']['is_stationary']
        }
        tracker.log_metrics_safely(stationarity_metrics)
        
        # Log critical values and other stationarity parameters
        stationarity_params = {
            'stationarity.adf_test.critical_values': stationarity_results['adf_test']['critical_values'],
            'stationarity.kpss_test.critical_values': stationarity_results['kpss_test']['critical_values'],
            'stationarity.overall.confidence': stationarity_results['overall_assessment']['confidence'],
            'stationarity.overall.recommendation': stationarity_results['overall_assessment']['recommendation']
        }
        tracker.log_params_safely(stationarity_params)
        
        # Log artifacts
        mlflow.log_artifacts(analysis_path, "analysis")
        mlflow.log_artifact(model_file_path, "model")
        
    except Exception as e:
        logger.warning(f"Failed to log results to MLflow: {str(e)}. Continuing without MLflow tracking.")
    
    # Create model summary
    model_summary = create_model_summary(
        initial_config=initial_config,
        train_data=train_data,
        test_data=test_data,
        model_params=model_params,
        metrics=metrics,
        analysis_results=analysis_results,
        stationarity_results=stationarity_results,
        model_path=model_path,
        analysis_path=analysis_path,
        grid_search_results=grid_search_results,
        preprocessor=preprocessor,
        use_grid_search=use_grid_search
    )
    
    # Save model summary
    summary_file = os.path.join(model_path, "model_summary.json")
    with open(summary_file, 'w') as f:
        json.dump(model_summary, f, indent=4)
    
    # Log execution time
    end_time = datetime.now()
    execution_time = (end_time - start_time).total_seconds()
    logger.info(f"Total training time: {execution_time:.2f} seconds")
    
    # Log the run URL
    if mlflow.active_run():
        run_id = mlflow.active_run().info.run_id
        experiment_id = mlflow.active_run().info.experiment_id
        print(f"🏃 View run {tracker.current_run.info.run_name} at: http://localhost:5001/#/experiments/{experiment_id}/runs/{run_id}")
        print(f"🧪 View experiment at: http://localhost:5001/#/experiments/{experiment_id}")
    
    return model_summary

@log_execution_time
def main():
    """Main function to train and evaluate SARIMA model."""
    # Load configuration
    with open('config/config.json', 'r') as f:
        config = json.load(f)
    
    # Get hyperparameter optimization setting from config
    use_grid_search = config['model'].get('optimize_hyperparameters', False)
    logger.info(f"Using {'grid search' if use_grid_search else 'suggested parameters'} for model training")
    
    # Initialize MLflow tracking
    run_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    tracker = ARIMATracker(experiment_name="sarima_optimization", run_name=f"run_{run_timestamp}")
    
    try:
        # Load and preprocess data
        logger.info("Loading data from data/examples/consumption_data_france.csv")
        data = pd.read_csv('data/examples/consumption_data_france.csv', parse_dates=['date'])
        data.set_index('date', inplace=True)
        
        # Initialize preprocessor with configuration
        preprocessor_config = {
            'add_time_features': config['preprocessing'].get('time_features', True),
            'add_lag_features': config['preprocessing'].get('lag_features', True),
            'add_rolling_features': config['preprocessing'].get('rolling_features', True),
            'handle_missing_values': config['preprocessing'].get('handle_missing_values', True),
            'handle_outliers': config['preprocessing'].get('handle_outliers', False),
            'outlier_config': config['preprocessing'].get('outlier_config', {
                'method': 'isolation_forest',
                'contamination': 0.1
            })
        }
        
        preprocessor = DataPreprocessor(
            config=preprocessor_config,
            experiment_name=tracker.experiment_name,
            run_name=tracker.current_run.info.run_name if tracker.current_run else None
        )
        
        # Prepare data
        logger.info("Starting data preparation")
        data = preprocessor.prepare_data(data, target_column=config['model']['target_column'])
        
        # Get suggested parameters and stationarity results
        suggested_params, stationarity_results = get_suggested_parameters(data[config['model']['target_column']], config)
        
        # Split data using preprocessor's method
        logger.info(f"Splitting time series data with train_ratio={config['preprocessing']['train_ratio']}, validation_ratio={config['preprocessing'].get('validation_ratio', 0.0)}, gap={config['preprocessing'].get('gap', 0)}")
        train_data, test_data = preprocessor.train_test_split_timeseries(
            data,
            train_ratio=config['preprocessing']['train_ratio'],
            validation_ratio=config['preprocessing'].get('validation_ratio', 0.0),
            target_column=config['model']['target_column'],
            date_column=data.index.name,
            gap=config['preprocessing'].get('gap', 0)
        )
        
        # Log data split information
        logger.info(f"Split data into {len(train_data)} training samples, 0 validation samples, and {len(test_data)} test samples")
        logger.info(f"Training data date range: {train_data.index[0]} to {train_data.index[-1]}")
        logger.info(f"Test data date range: {test_data.index[0]} to {test_data.index[-1]}")
        
        # Log data statistics
        target_col = config['model']['target_column']
        logger.info(f"Training target stats - Mean: {train_data[target_col].mean():.4f}, Std: {train_data[target_col].std():.4f}, Min: {train_data[target_col].min():.4f}, Max: {train_data[target_col].max():.4f}")
        logger.info(f"Test target stats - Mean: {test_data[target_col].mean():.4f}, Std: {test_data[target_col].std():.4f}, Min: {test_data[target_col].min():.4f}, Max: {test_data[target_col].max():.4f}")
        
        if use_grid_search:
            # Set up grid search parameters
            param_grid = {
                'p': list(range(max(0, suggested_params['p'] - 1), min(4, suggested_params['p'] + 2))),
                'd': [suggested_params['d']],
                'q': list(range(max(0, suggested_params['q'] - 1), min(4, suggested_params['q'] + 2))),
                'P': list(range(max(0, suggested_params['P'] - 1), min(3, suggested_params['P'] + 2))),
                'D': [suggested_params['D']],
                'Q': list(range(max(0, suggested_params['Q'] - 1), min(3, suggested_params['Q'] + 2))),
                's': [suggested_params['s']]
            }
            
            # Initialize model for grid search
            model = TimeSeriesModel(
                model_type='sarima',
                **config.get('model', {
                    'trend': 'c',
                    'maxiter': 50
                })
            )
            
            # Perform grid search
            logger.info("Starting grid search for best parameters")
            grid_search_results = model.grid_search(train_data[config['model']['target_column']].values, param_grid)
            
            # Get best model and parameters
            model = grid_search_results['best_model']
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
            logger.info(f"Training SARIMA model with parameters: {suggested_params}")
            model = TimeSeriesModel(
                model_type='sarima',
                **{**config.get('model', {
                    'trend': 'c',
                    'maxiter': 50
                }), **suggested_params}
            )
            
            # Train model
            logger.info("Starting model training")
            metrics = model.fit(train_data[config['model']['target_column']].values)
            model_params = suggested_params
            grid_search_results = None
        
        # Handle post-training tasks
        model_summary = handle_post_training(
            model=model,
            model_params=model_params,
            metrics=metrics,
            train_data=train_data,
            test_data=test_data,
            initial_config=config,
            stationarity_results=stationarity_results,
            tracker=tracker,
            start_time=datetime.strptime(run_timestamp, '%Y%m%d_%H%M%S'),
            grid_search_results=grid_search_results,
            preprocessor=preprocessor,
            use_grid_search=use_grid_search
        )
        
        return model_summary
        
    except Exception as e:
        logger.error(f"Error during model training: {str(e)}")
        raise
    
    finally:
        if tracker:
            tracker.end_run()

if __name__ == "__main__":
    main() 