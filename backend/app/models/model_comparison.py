from typing import Dict, Any, Optional, Tuple, List
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from .base_model import BaseForecastingModel
from ..utils.logger import Logger

logger = Logger()

class ModelComparison:
    def __init__(self):
        logger.info("Initializing ModelComparison")
        self.models: Dict[str, BaseForecastingModel] = {}
        self.results: Dict[str, Dict[str, Any]] = {}
        
    def add_model(self, name: str, model: BaseForecastingModel) -> None:
        """Add a model to the comparison."""
        logger.info(f"Adding model: {name}")
        self.models[name] = model
        
    def train_and_evaluate(self, train_data: pd.DataFrame, test_data: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
        """Train and evaluate all models."""
        logger.info(f"Starting training and evaluation of {len(self.models)} models")
        for name, model in self.models.items():
            logger.info(f"Training model: {name}")
            # Train model
            train_results = model.train(train_data)
            logger.info(f"Training completed for {name}")
            
            # Evaluate model
            logger.info(f"Evaluating model: {name}")
            metrics = model.evaluate(test_data)
            logger.info(f"Evaluation metrics for {name}: {metrics}")
            
            # Generate predictions
            logger.info(f"Generating predictions for {name}")
            predictions, confidence_intervals = model.predict(steps=len(test_data))
            logger.info(f"Predictions generated for {name}")
            
            self.results[name] = {
                "train_results": train_results,
                "metrics": metrics,
                "predictions": predictions,
                "confidence_intervals": confidence_intervals
            }
            
        logger.info("All models trained and evaluated successfully")
        return self.results
        
    def plot_comparison(self, test_data: pd.DataFrame) -> None:
        """Plot predictions from all models."""
        logger.info("Generating comparison plot")
        plt.figure(figsize=(12, 6))
        plt.plot(test_data.index, test_data["value"], label="Actual", color="black")
        
        for name, result in self.results.items():
            logger.debug(f"Plotting predictions for model: {name}")
            plt.plot(test_data.index, result["predictions"], label=name)
            if result["confidence_intervals"]:
                lower, upper = result["confidence_intervals"]
                plt.fill_between(test_data.index, lower, upper, alpha=0.2)
                
        plt.title("Model Predictions Comparison")
        plt.xlabel("Date")
        plt.ylabel("Value")
        plt.legend()
        plt.grid(True)
        plt.show()
        logger.info("Comparison plot generated successfully")
        
    def plot_metrics_comparison(self) -> None:
        """Plot comparison of model metrics."""
        logger.info("Generating metrics comparison plot")
        metrics_df = pd.DataFrame({
            name: result["metrics"]
            for name, result in self.results.items()
        }).T
        
        plt.figure(figsize=(10, 6))
        sns.heatmap(metrics_df, annot=True, cmap="YlOrRd")
        plt.title("Model Metrics Comparison")
        plt.show()
        logger.info("Metrics comparison plot generated successfully")
        
    def get_best_model(self, metric: str = "rmse") -> Tuple[str, float]:
        """Get the best model based on a specified metric."""
        logger.info(f"Finding best model based on metric: {metric}")
        best_name = None
        best_value = float("inf")
        
        for name, result in self.results.items():
            value = result["metrics"][metric]
            if value < best_value:
                best_value = value
                best_name = name
                
        logger.info(f"Best model: {best_name} with {metric} = {best_value}")
        return best_name, best_value
        
    def generate_report(self) -> Dict[str, Any]:
        """Generate a comprehensive comparison report."""
        logger.info("Generating comparison report")
        report = {
            "model_count": len(self.models),
            "metrics_summary": {},
            "best_models": {}
        }
        
        # Calculate average metrics
        for metric in ["rmse", "mae", "mape", "r2"]:
            values = [result["metrics"][metric] for result in self.results.values()]
            report["metrics_summary"][metric] = {
                "mean": np.mean(values),
                "std": np.std(values),
                "min": np.min(values),
                "max": np.max(values)
            }
            
        # Find best model for each metric
        for metric in ["rmse", "mae", "mape", "r2"]:
            best_name, best_value = self.get_best_model(metric)
            report["best_models"][metric] = {
                "model": best_name,
                "value": best_value
            }
            
        logger.info("Comparison report generated successfully")
        return report
        
    def calculate_ensemble_forecast(self, weights: Optional[Dict[str, float]] = None) -> Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Calculate ensemble forecast using weighted average of model predictions."""
        logger.info("Calculating ensemble forecast")
        if not weights:
            weights = {name: 1.0 / len(self.models) for name in self.models}
            logger.info("Using equal weights for ensemble")
            
        # Ensure weights sum to 1
        total_weight = sum(weights.values())
        weights = {name: weight / total_weight for name, weight in weights.items()}
        
        # Calculate weighted average predictions
        ensemble_predictions = np.zeros_like(next(iter(self.results.values()))["predictions"])
        ensemble_lower = np.zeros_like(ensemble_predictions)
        ensemble_upper = np.zeros_like(ensemble_predictions)
        
        for name, weight in weights.items():
            logger.debug(f"Adding {name} to ensemble with weight {weight}")
            result = self.results[name]
            ensemble_predictions += weight * result["predictions"]
            if result["confidence_intervals"]:
                lower, upper = result["confidence_intervals"]
                ensemble_lower += weight * lower
                ensemble_upper += weight * upper
                
        logger.info("Ensemble forecast calculated successfully")
        return ensemble_predictions, (ensemble_lower, ensemble_upper) 