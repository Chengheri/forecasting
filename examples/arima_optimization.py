import pandas as pd
import numpy as np
from datetime import datetime
import os
import sys
import itertools
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
import mlflow
from tqdm import tqdm

# Add the project root directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from backend.app.utils.logger import Logger

# Initialize logger
logger = Logger()

def load_and_prepare_data(filename: str, target_column: str = 'value') -> pd.DataFrame:
    """Load and prepare the data for ARIMA modeling."""
    logger.info(f"Loading and preparing data from {filename}")
    try:
        df = pd.read_csv(filename)
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date')
        df = df[[target_column]]  # Keep only the target column
        logger.info(f"Data loaded successfully. Shape: {df.shape}")
        return df
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        raise

def split_data(df: pd.DataFrame, train_size: float = 0.8) -> tuple:
    """Split data into training and testing sets."""
    logger.info(f"Splitting data with train_size: {train_size}")
    n = len(df)
    train_idx = int(n * train_size)
    train = df[:train_idx]
    test = df[train_idx:]
    return train, test

def evaluate_arima_model(train: pd.DataFrame, test: pd.DataFrame, order: tuple) -> dict:
    """Evaluate an ARIMA model with the given order."""
    try:
        # Fit the model
        model = SARIMAX(train, order=order)
        results = model.fit(disp=False)
        
        # Make predictions
        predictions = results.forecast(steps=len(test))
        
        # Calculate metrics
        mse = mean_squared_error(test, predictions)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(test, predictions)
        r2 = r2_score(test, predictions)
        aic = results.aic
        bic = results.bic
        
        return {
            'order': order,
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'aic': aic,
            'bic': bic,
            'predictions': predictions
        }
    except Exception as e:
        logger.warning(f"Failed to evaluate ARIMA{order}: {str(e)}")
        return None

def optimize_arima(train: pd.DataFrame, test: pd.DataFrame, 
                  p_values: list, d_values: list, q_values: list) -> list:
    """Perform grid search for ARIMA hyperparameters."""
    logger.info("Starting ARIMA hyperparameter optimization")
    results = []
    
    # Create all possible combinations of parameters
    pdq = list(itertools.product(p_values, d_values, q_values))
    
    # Set up MLflow tracking
    mlflow.set_experiment("arima_optimization")
    
    # Try each combination of parameters
    with tqdm(total=len(pdq)) as pbar:
        for order in pdq:
            with mlflow.start_run(nested=True):
                # Log parameters
                mlflow.log_params({
                    'p': order[0],
                    'd': order[1],
                    'q': order[2]
                })
                
                # Evaluate model
                eval_result = evaluate_arima_model(train, test, order)
                
                if eval_result is not None:
                    # Log metrics
                    mlflow.log_metrics({
                        'mse': eval_result['mse'],
                        'rmse': eval_result['rmse'],
                        'mae': eval_result['mae'],
                        'r2': eval_result['r2'],
                        'aic': eval_result['aic'],
                        'bic': eval_result['bic']
                    })
                    results.append(eval_result)
                
            pbar.update(1)
    
    return results

def plot_optimization_results(results: list, output_dir: str):
    """Plot optimization results."""
    logger.info("Generating optimization results plots")
    
    # Create DataFrame from results
    df_results = pd.DataFrame([
        {
            'p': r['order'][0],
            'd': r['order'][1],
            'q': r['order'][2],
            'RMSE': r['rmse'],
            'AIC': r['aic'],
            'BIC': r['bic']
        }
        for r in results
    ])
    
    # Plot RMSE distribution
    plt.figure(figsize=(10, 6))
    sns.boxplot(y='RMSE', data=df_results)
    plt.title('Distribution of RMSE across all models')
    plt.savefig(os.path.join(output_dir, 'rmse_distribution.png'))
    plt.close()
    
    # Plot parameter importance
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    sns.boxplot(x='p', y='RMSE', data=df_results, ax=axes[0])
    axes[0].set_title('RMSE by p value')
    
    sns.boxplot(x='d', y='RMSE', data=df_results, ax=axes[1])
    axes[1].set_title('RMSE by d value')
    
    sns.boxplot(x='q', y='RMSE', data=df_results, ax=axes[2])
    axes[2].set_title('RMSE by q value')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'parameter_importance.png'))
    plt.close()
    
    return df_results

def plot_best_model_results(train: pd.DataFrame, test: pd.DataFrame, 
                          best_model: dict, output_dir: str):
    """Plot results from the best model."""
    logger.info("Generating best model results plots")
    
    # Combine actual and predicted values
    predictions = best_model['predictions']
    test_values = test.values.flatten()
    train_values = train.values.flatten()
    
    plt.figure(figsize=(15, 7))
    plt.plot(train.index, train_values, label='Training Data')
    plt.plot(test.index, test_values, label='Actual Test Data')
    plt.plot(test.index, predictions, label='Predictions')
    plt.title(f'ARIMA{best_model["order"]} - Best Model Results')
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'best_model_predictions.png'))
    plt.close()
    
    # Plot residuals
    residuals = test_values - predictions
    plt.figure(figsize=(15, 7))
    plt.subplot(2, 1, 1)
    plt.plot(test.index, residuals)
    plt.title('Residuals over Time')
    plt.xlabel('Date')
    plt.ylabel('Residual')
    
    plt.subplot(2, 1, 2)
    sns.histplot(residuals, kde=True)
    plt.title('Residuals Distribution')
    plt.xlabel('Residual')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'best_model_residuals.png'))
    plt.close()

def main():
    try:
        # Set up directories
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        data_path = os.path.join(project_root, 'data', 'examples', 'consumption_data_france.csv')
        analysis_dir = os.path.join(project_root, 'data', 'analysis', 'arima_optimization')
        logs_dir = os.path.join(project_root, 'data', 'logs')
        
        # Create directories if they don't exist
        os.makedirs(analysis_dir, exist_ok=True)
        os.makedirs(logs_dir, exist_ok=True)
        
        # Suppress warnings
        warnings.filterwarnings('ignore')
        
        logger.info("Starting ARIMA optimization analysis")
        
        # Load and prepare data
        df = load_and_prepare_data(data_path)
        train, test = split_data(df)
        
        # Define parameter grid
        p_values = range(0, 4)
        d_values = range(0, 3)
        q_values = range(0, 4)
        
        # Run optimization
        results = optimize_arima(train, test, p_values, d_values, q_values)
        
        # Find best model based on RMSE
        best_model = min(results, key=lambda x: x['rmse'])
        
        # Log best model results
        logger.info(f"Best model: ARIMA{best_model['order']}")
        logger.info(f"Best model RMSE: {best_model['rmse']:.4f}")
        logger.info(f"Best model AIC: {best_model['aic']:.4f}")
        logger.info(f"Best model BIC: {best_model['bic']:.4f}")
        
        # Plot results
        df_results = plot_optimization_results(results, analysis_dir)
        plot_best_model_results(train, test, best_model, analysis_dir)
        
        # Save results summary
        summary = pd.DataFrame([{
            'order': str(r['order']),
            'rmse': r['rmse'],
            'mae': r['mae'],
            'r2': r['r2'],
            'aic': r['aic'],
            'bic': r['bic']
        } for r in results])
        summary.to_csv(os.path.join(analysis_dir, 'optimization_results.csv'), index=False)
        
        logger.info(f"Analysis complete. Results saved in {analysis_dir}")
        print(f"\nAnalysis complete. Results saved in {analysis_dir}")
        print(f"Logs saved in {logs_dir}")
        
        # Print best model summary
        print("\nBest Model Summary:")
        print(f"Order (p,d,q): {best_model['order']}")
        print(f"RMSE: {best_model['rmse']:.4f}")
        print(f"MAE: {best_model['mae']:.4f}")
        print(f"R²: {best_model['r2']:.4f}")
        print(f"AIC: {best_model['aic']:.4f}")
        print(f"BIC: {best_model['bic']:.4f}")
        
    except Exception as e:
        logger.error(f"Error in ARIMA optimization: {str(e)}")
        raise

if __name__ == "__main__":
    main() 