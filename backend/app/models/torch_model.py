import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, Any, Optional, Tuple, Union, List, cast
import pandas as pd
import optuna
import joblib
import os
from pathlib import Path
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import matplotlib.pyplot as plt

from ..utils.logger import Logger
from ..utils.trackers import ProphetTracker
from .base_model import BaseForecastingModel
logger = Logger()

class TorchModel(BaseForecastingModel):
    """Torch forecasting model with preprocessing, training, and prediction capabilities."""
    
    def __init__(
        self,
        config: Dict[str, Any],
        tracker: Optional[ProphetTracker] = None,
    ):
        """Initialize Torch model with configuration and tracker."""
        self.config = config
        self.tracker = tracker
        self.model = None
        self.scaler: Optional[Union[StandardScaler, MinMaxScaler]] = None
        self._training_data = None
        self._model_is_fitted = False
        self.history: List[Dict[str, Any]] = []
        
        # Initialize data preprocessing
        scaling_method = self.config['preprocessing'].get('scaling_method')
        if scaling_method == 'minmax':
            self.scaler = MinMaxScaler()
        else:
            self.scaler = StandardScaler()
        
    def initialize_model(self) -> None:
        """Initialize the Torch model with parameters from config."""
        pass
            
    def prepare_data(self, data: Union[pd.DataFrame, pd.Series, np.ndarray]) -> Dict[str, torch.Tensor]:
        """Prepare data for rorchh model training.
        
        Args:
            data: Input data (DataFrame, Series, or numpy array)
            
        Returns:
            Dictionary with prepared tensors for model training
        """
        logger.info(f"Preparing data for {self.config['model']['model_type'].capitalize()} model...")
        
        try:
            # Convert to DataFrame if necessary
            if isinstance(data, pd.Series):
                data = data.to_frame()
            elif isinstance(data, np.ndarray):
                if data.ndim == 1:
                    data = pd.DataFrame(data, columns=[self.config['data']['target_column']])
                else:
                    data = pd.DataFrame(data)
            
            # Extract target column
            target_col = self.config['data']['target_column']
            
            # Convert data to numpy arrays
            if target_col in data.columns:
                y = data[target_col].values
                X = data.drop(columns=[target_col] if target_col in data.columns else [])
                
                if self.config['data'].get('date_column') in X.columns:
                    X = X.drop(columns=[self.config['data'].get('date_column')])
            else:
                # If target column not in data, assume it's all features
                X = data
                y = np.zeros(len(data))  # Placeholder

            # Scale features
            if self.scaler:
                X_scaled = self.scaler.fit_transform(X)
            else:
                X_scaled = X.values
            # Create sequences for LSTM
            seq_length = self.config['model'].get('sequence_length')
            X_sequences, y_sequences = self._create_sequences(X_scaled, y, seq_length)
            
            # Convert to PyTorch tensors
            X_tensor = torch.FloatTensor(X_sequences)
            y_tensor = torch.FloatTensor(y_sequences)
            
            logger.info(f"Data prepared: {X_tensor.shape} inputs, {y_tensor.shape} targets")
            
            return {
                'X': X_tensor,
                'y': y_tensor,
                'X_raw': X,
                'y_raw': y,
                'X_scaled': X_scaled
            }
            
        except Exception as e:
            logger.error(f"Error preparing data: {str(e)}")
            raise
            
    def _create_sequences(self, X: np.ndarray, y: np.ndarray, seq_length: int) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequence data for LSTM training.
        
        Args:
            X: Feature data array
            y: Target data array
            seq_length: Sequence length for LSTM
            
        Returns:
            Tuple of arrays (X_sequences, y_sequences)
        """
        X_sequences, y_sequences = [], []
        
        for i in range(len(X) - seq_length):
            X_sequences.append(X[i:i+seq_length])
            y_sequences.append(y[i+seq_length])
            
        return np.array(X_sequences), np.array(y_sequences)
    
    def fit(self, data: Union[pd.Series, pd.DataFrame, np.ndarray], 
            params: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
        """Fit the LSTM model to data.
        
        Args:
            data: Training data
            params: Optional parameter overrides
            
        Returns:
            Dictionary of training metrics
        """
        try:
            model_type = self.config['model']['model_type']
            # Update parameters if provided
            if params:
                self.config['model'].update(params)
            
            # Fix: correct the key name (removing the colon)
            prepared_data = self.prepare_data(data)
            self.config['data'].update({'features': prepared_data['X_raw'].columns.tolist()})
            input_size = prepared_data['X'].shape[2]  # Get actual input size from prepared data
            self.config['model'].update({'input_size': input_size})
            
            self.initialize_model()
                        
            if self.model is None:
                raise ValueError("Model must be initialized before fitting")
            
            # Use the already prepared data
            X_tensor, y_tensor = prepared_data['X'], prepared_data['y']
            
            # Store training data for later use
            self._training_data = data.copy() if isinstance(data, pd.DataFrame) else data
            
            # Set up optimizer and loss function
            optimizer_name = self.config['model'].get('optimizer').lower()
            if optimizer_name == 'adam':
                optimizer = optim.Adam(
                    self.model.parameters(), 
                    lr=self.config['model'].get('learning_rate'),
                    weight_decay=self.config['model'].get('weight_decay')
                )
            elif optimizer_name == 'sgd':
                optimizer = optim.SGD(
                    self.model.parameters(), 
                    lr=self.config['model'].get('learning_rate'),
                    momentum=self.config['model'].get('momentum'),
                    weight_decay=self.config['model'].get('weight_decay')
                )
            else:
                optimizer = optim.Adam(self.model.parameters(), lr=0.001)
            
            loss_fn_name = self.config['model'].get('loss_function').lower()
            if loss_fn_name == 'mse':
                criterion = nn.MSELoss()
            elif loss_fn_name == 'mae':
                criterion = nn.L1Loss()
            elif loss_fn_name == 'huber':
                criterion = nn.SmoothL1Loss()
            else:
                criterion = nn.MSELoss()
            
            # Training loop
            epochs = self.config['model'].get('epochs')
            batch_size = self.config['model'].get('batch_size')
            clip_value = self.config['model'].get('clip_gradient')
            
            # Use early stopping if configured
            early_stopping = self.config['training'].get('early_stopping')
            patience = self.config['training'].get('patience')
            min_delta = self.config['training'].get('min_delta')
            best_loss = float('inf')
            patience_counter = 0
            
            # Clear history
            self.history = []
            
            # Split into train/validation if needed
            train_ratio = 1.0 - self.config['training'].get('validation_size')
            train_size = int(len(X_tensor) * train_ratio)
            
            X_train, X_val = X_tensor[:train_size], X_tensor[train_size:]
            y_train, y_val = y_tensor[:train_size], y_tensor[train_size:]
            
            # Create DataLoader for batch training
            train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
            train_loader = torch.utils.data.DataLoader(
                train_dataset, batch_size=batch_size, shuffle=True
            )
            
            logger.info(f"Training {model_type.capitalize()} model with {epochs} epochs, batch size {batch_size}")
            
            # Training loop
            self.model.train()
            for epoch in range(epochs):
                epoch_loss = 0.0
                batch_count = 0
                
                for X_batch, y_batch in train_loader:
                    # Forward pass
                    optimizer.zero_grad()
                    outputs = self.model(X_batch)
                    loss = criterion(outputs, y_batch.unsqueeze(1))
                    
                    # Backward pass and optimize
                    loss.backward()
                    if clip_value > 0:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip_value)
                    optimizer.step()
                    
                    epoch_loss += loss.item()
                    batch_count += 1
                
                # Calculate average loss for epoch
                avg_epoch_loss = epoch_loss / batch_count if batch_count > 0 else 0
                
                # Validation loss
                with torch.no_grad():
                    self.model.eval()
                    val_outputs = self.model(X_val)
                    val_loss = criterion(val_outputs, y_val.unsqueeze(1)).item()
                    self.model.train()
                
                # Store metrics
                self.history.append({
                    'epoch': epoch + 1,
                    'train_loss': avg_epoch_loss,
                    'val_loss': val_loss
                })
                
                # Log progress
                if (epoch + 1) % 10 == 0 or epoch == 0 or epoch == epochs - 1:
                    logger.info(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_epoch_loss:.4f}, Val Loss: {val_loss:.4f}")
                
                # Early stopping
                if early_stopping:
                    if val_loss < best_loss - min_delta:
                        best_loss = val_loss
                        patience_counter = 0
                    else:
                        patience_counter += 1
                        
                    if patience_counter >= patience:
                        logger.info(f"Early stopping triggered after {epoch+1} epochs")
                        break
            
            # Mark as fitted
            self._model_is_fitted = True
            
            # Calculate final metrics
            with torch.no_grad():
                self.model.eval()
                final_train_preds = self.model(X_train).squeeze().numpy()
                final_val_preds = self.model(X_val).squeeze().numpy()
                
                # Calculate metrics
                train_metrics = self._calculate_metrics(y_train.numpy(), final_train_preds)
                val_metrics = self._calculate_metrics(y_val.numpy(), final_val_preds)
                
                metrics = {
                    'train_loss': self.history[-1]['train_loss'],
                    'val_loss': self.history[-1]['val_loss'],
                    'train_rmse': train_metrics['rmse'],
                    'val_rmse': val_metrics['rmse'],
                    'train_mae': train_metrics['mae'],
                    'val_mae': val_metrics['mae'],
                    'train_r2': train_metrics['r2'],
                    'val_r2': val_metrics['r2']
                }
            
            # Convert metrics to native Python types (for JSON serialization)
            metrics = {k: float(v) for k, v in metrics.items()}
            
            logger.info(f"{model_type.capitalize()} model training completed successfully")
            logger.info(f"Final metrics: {metrics}")
            
            # Log to MLflow if tracker is available
            if self.tracker:
                self.tracker.log_params_safely(self.config['model'])
                self.tracker.log_metrics(metrics)
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error training {model_type.capitalize()} model: {str(e)}")
            raise
    
    def predict0(self, data: pd.DataFrame) -> Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Generate predictions for future steps.
        
        Args:
            steps: Number of future steps to predict
            
        Returns:
            Tuple of (predictions, (lower_bound, upper_bound))
        """
        if not self._model_is_fitted or self.model is None:
            raise ValueError("Model must be trained before prediction")
        
        steps = len(data)
        logger.info(f"Generating predictions for {steps} steps")
        
        try:
            if self._training_data is None:
                raise ValueError("No training data available for prediction")
                
            # Prepare the input data for prediction
            logger.info('Preparing training data')
            prepared_data = self.prepare_data(self._training_data)
            X_scaled = prepared_data['X_scaled']
            logger.info('Preparing test data')
            prepared_test_data = self.prepare_data(data)
            
            # Get the sequence length
            seq_length = self.config['model'].get('sequence_length')
            
            # Use the most recent data as input
            if len(X_scaled) >= seq_length:
                last_sequence = X_scaled[-seq_length:].reshape(1, seq_length, -1)
                input_tensor = torch.FloatTensor(last_sequence)
            else:
                # Handle case where we don't have enough data
                pad_length = seq_length - len(X_scaled)
                padded_data = np.vstack([np.zeros((pad_length, X_scaled.shape[1])), X_scaled])
                last_sequence = padded_data.reshape(1, seq_length, -1)
                input_tensor = torch.FloatTensor(last_sequence)
            
            # Storage for predictions
            predictions = np.zeros(steps)
            
            # Set model to evaluation mode
            self.model.eval()
            
            # Generate predictions iteratively
            with torch.no_grad():
                current_input = input_tensor
                
                for i in range(steps):
                    # Predict next step
                    output = self.model(current_input).numpy()[0, 0]
                    predictions[i] = output
                    
                    # Update input sequence for next step prediction
                    # Remove oldest step and add new prediction
                    if steps > 1 and i < steps - 1:
                        # For multivariate case, we need to update only the target variable
                        # Assume that the target variable is the first column
                        new_input = np.zeros((1, 1, current_input.shape[2]))
                        new_input[0, 0, 0] = output  # Set the target variable
                        
                        # Shift the sequence and add the new prediction
                        current_input = torch.cat([
                            current_input[:, 1:, :],
                            torch.FloatTensor(new_input)
                        ], dim=1)
            
            # Calculate confidence intervals
            # For LSTM, we use a simple approach based on the prediction variance
            # In a real-world scenario, you might want to use bootstrapping or other methods
            history_std = np.std(prepared_test_data['y_raw'])
            lower_bound = predictions - 1.96 * history_std
            upper_bound = predictions + 1.96 * history_std
            
            logger.info("Predictions generated successfully")
            return predictions, (lower_bound, upper_bound)
            
        except Exception as e:
            logger.error(f"Error generating predictions: {str(e)}")
            raise
    
    def predict(self, data: pd.DataFrame) -> Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        try:
            # Prepare test data
            prepared_data = self.prepare_data(data)
            X_test, y_test = prepared_data['X'], prepared_data['y']
            
            # Get predictions
            self.model.eval()
            with torch.no_grad():
                y_pred = self.model(X_test).squeeze().numpy()

                history_std = np.std(prepared_data['y_raw'][0:len(y_pred)])
                lower_bound = y_pred - 1.96 * history_std
                upper_bound = y_pred + 1.96 * history_std
                return y_pred, (lower_bound, upper_bound)

        except Exception as e:
            logger.error(f"Error generating predictions: {str(e)}")
            raise

            
    def evaluate(self, test_data: pd.DataFrame) -> Dict[str, float]:
        """Evaluate the Transformer model on test data.
        
        Args:
            test_data: Test dataset
            
        Returns:
            Dictionary of evaluation metrics
        """
        logger.info("Evaluating Transformer model performance")
        
        if not self._model_is_fitted or self.model is None:
            raise ValueError("Model must be trained before evaluation")
            
        try:
            y_pred = self.predict(test_data)
                
            # Calculate metrics
            metrics = self._calculate_metrics(y_test.numpy(), y_pred)
            
            # Log metrics
            logger.info(f"Evaluation metrics: {metrics}")
            
            # Log to MLflow if tracker is available
            if self.tracker:
                self.tracker.log_metrics({f"test_{k}": v for k, v in metrics.items()})
                
            return metrics
            
        except Exception as e:
            logger.error(f"Error evaluating Transformer model: {str(e)}")
            raise

    def fit_with_optuna(self, data: Union[pd.Series, pd.DataFrame, np.ndarray], 
                       n_trials: int = 100, timeout: int = 600) -> Dict[str, Any]:
        """Optimize LSTM hyperparameters using Optuna.
        
        Args:
            data: Training data
            n_trials: Number of optimization trials
            timeout: Timeout in seconds
            
        Returns:
            Dictionary of optimization results
        """
        n_trials = self.config['model'].get('n_trials', n_trials)
        timeout = self.config['model'].get('timeout', timeout)
        
        if data is None:
            raise ValueError("Data cannot be None")
            
        logger.info(f"Starting hyperparameter optimization with {n_trials} trials and {timeout}s timeout")
        
        def objective(trial):
            """Optimization objective function."""
            params = {
                "hidden_size": trial.suggest_categorical("hidden_size", [32, 64, 128, 256]),
                "num_layers": trial.suggest_int("num_layers", 1, 4),
                "learning_rate": trial.suggest_float("learning_rate", 0.0001, 0.01, log=True),
                "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64, 128]),
                "dropout": trial.suggest_float("dropout", 0.1, 0.5),
                "bidirectional": trial.suggest_categorical("bidirectional", [False, True]),
                "optimizer": trial.suggest_categorical("optimizer", ["adam", "sgd"]),
                "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
            }
            
            try:
                # Update model parameters and train
                self.config['model'].update(params)
                metrics = self.fit(data)
                
                # Return validation RMSE as objective
                return metrics.get('val_rmse', float('inf'))
            except Exception as e:
                logger.warning(f"Trial failed: {str(e)}")
                return float('inf')

        return super().fit_with_optuna(data, objective, n_trials, timeout)
    
    def save(self, path: str) -> None:
        """Save the LSTM model to disk.
        
        Args:
            path: Path to save model
        """
        logger.info(f"Saving LSTM model to {path}")
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            
            # Create dictionary with all necessary objects
            save_dict = {
                'model_state_dict': self.model.state_dict() if self.model else None,
                'config': self.config,
                'scaler': self.scaler,
                'history': self.history,
                'is_fitted': self._model_is_fitted
            }
            
            # Save to disk
            with open(path, 'wb') as f:
                joblib.dump(save_dict, f)
                
            logger.info(f"Model saved successfully to {path}")
            
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            raise
    
    def load(self, path: str) -> None:
        """Load the LSTM model from disk.
        
        Args:
            path: Path to load model from
        """
        logger.info(f"Loading LSTM model from {path}")
        try:
            with open(path, 'rb') as f:
                save_dict = joblib.load(f)
            
            # Load configuration
            self.config = save_dict['config']
            
            # Re-initialize model architecture
            self.initialize_model()
            
            # Load model weights
            if self.model and save_dict['model_state_dict']:
                self.model.load_state_dict(save_dict['model_state_dict'])
            
            # Load other attributes
            self.scaler = save_dict['scaler']
            self.history = save_dict['history']
            self._model_is_fitted = save_dict['is_fitted']
            
            logger.info("Model loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
    
    def cross_validation(self, data: Union[pd.Series, pd.DataFrame, np.ndarray], 
                        n_splits: int = 5, sequence_length: Optional[int] = None) -> Dict[str, Any]:
        """Perform time series cross-validation.
        
        Args:
            data: Input data
            n_splits: Number of cross-validation splits
            sequence_length: LSTM sequence length (default: from config)
            
        Returns:
            Dictionary of cross-validation results
        """
        logger.info(f"Starting time series cross-validation with {n_splits} splits")
        
        try:
            # Convert data to DataFrame if necessary
            if isinstance(data, pd.Series):
                data = data.to_frame()
            elif isinstance(data, np.ndarray):
                if data.ndim == 1:
                    data = pd.DataFrame(data, columns=[self.config['data']['target_column']])
                else:
                    data = pd.DataFrame(data)
            
            # Use sequence length from config if not specified
            if sequence_length is None:
                sequence_length = self.config['model'].get('sequence_length')
            
            # Prepare data once
            prepared_data = self.prepare_data(data)
            
            # Get total size and calculate fold size
            n_samples = len(prepared_data['X'])
            fold_size = n_samples // n_splits
            
            # Storage for results
            cv_results = []
            fold_metrics = []
            
            # Perform cross-validation
            for fold in range(n_splits):
                logger.info(f"Processing fold {fold+1}/{n_splits}")
                
                # Calculate validation indices - use time-based splits for time series
                val_start = fold * fold_size
                val_end = (fold + 1) * fold_size if fold < n_splits - 1 else n_samples
                
                # Create train/validation split
                X_train = torch.cat([
                    prepared_data['X'][:val_start], 
                    prepared_data['X'][val_end:]
                ]) if val_start > 0 else prepared_data['X'][val_end:]
                
                y_train = torch.cat([
                    prepared_data['y'][:val_start], 
                    prepared_data['y'][val_end:]
                ]) if val_start > 0 else prepared_data['y'][val_end:]
                
                X_val = prepared_data['X'][val_start:val_end]
                y_val = prepared_data['y'][val_start:val_end]
                
                # Skip if validation set is too small
                if len(X_val) < 10:
                    logger.warning(f"Skipping fold {fold+1} - validation set too small ({len(X_val)} samples)")
                    continue
                
                # Re-initialize model for each fold
                self.initialize_model()
                
                # Create temporary dataset and dataloader
                train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
                batch_size = self.config['model'].get('batch_size')
                train_loader = torch.utils.data.DataLoader(
                    train_dataset, batch_size=batch_size, shuffle=True
                )
                
                # Set up optimizer and loss function
                optimizer = optim.Adam(
                    self.model.parameters(), 
                    lr=self.config['model'].get('learning_rate')
                )
                criterion = nn.MSELoss()
                
                # Training loop for this fold
                epochs = self.config['model'].get('epochs')
                fold_history = []
                
                self.model.train()
                for epoch in range(epochs):
                    epoch_loss = 0.0
                    batch_count = 0
                    
                    for X_batch, y_batch in train_loader:
                        # Forward pass
                        optimizer.zero_grad()
                        outputs = self.model(X_batch)
                        loss = criterion(outputs, y_batch.unsqueeze(1))
                        
                        # Backward pass and optimize
                        loss.backward()
                        optimizer.step()
                        
                        epoch_loss += loss.item()
                        batch_count += 1
                    
                    # Calculate validation loss
                    with torch.no_grad():
                        self.model.eval()
                        val_outputs = self.model(X_val)
                        val_loss = criterion(val_outputs, y_val.unsqueeze(1)).item()
                        self.model.train()
                    
                    # Store metrics
                    fold_history.append({
                        'epoch': epoch + 1,
                        'train_loss': epoch_loss / max(1, batch_count),
                        'val_loss': val_loss
                    })
                    
                    # Log progress periodically
                    if (epoch + 1) % 20 == 0:
                        logger.info(f"Fold {fold+1}, Epoch {epoch+1}/{epochs}, Val Loss: {val_loss:.4f}")
                
                # Evaluate final model on validation set
                with torch.no_grad():
                    self.model.eval()
                    val_preds = self.model(X_val).squeeze().numpy()
                    val_true = y_val.numpy()
                    
                    # Calculate metrics
                    fold_result = self._calculate_metrics(val_true, val_preds)
                    fold_result['fold'] = fold + 1
                    fold_result['val_loss'] = fold_history[-1]['val_loss']
                    fold_metrics.append(fold_result)
                
                # Store results for this fold
                cv_results.append({
                    'fold': fold + 1,
                    'metrics': fold_result,
                    'history': fold_history,
                    'predictions': val_preds,
                    'actual': val_true
                })
                
                logger.info(f"Fold {fold+1} complete - RMSE: {fold_result['rmse']:.4f}")
            
            # Calculate average metrics across folds
            metrics_df = pd.DataFrame(fold_metrics)
            mean_metrics = {
                col: metrics_df[col].mean() for col in ['rmse', 'mae', 'r2', 'mape', 'mse']
                if col in metrics_df.columns
            }
            
            cv_summary = {
                'cv_results': cv_results,
                'mean_metrics': mean_metrics,
                'fold_metrics': fold_metrics,
                'n_splits': n_splits
            }
            
            logger.info(f"Cross-validation complete. Mean RMSE: {mean_metrics['rmse']:.4f}")
            return cv_summary
            
        except Exception as e:
            logger.error(f"Error in cross-validation: {str(e)}")
            raise