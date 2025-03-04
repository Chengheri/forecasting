import pandas as pd
import numpy as np
from typing import Tuple, List, Dict, Any, Optional
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from datetime import datetime, timedelta
from ..utils.logger import Logger
from .preprocessor_tracker import PreprocessorTracker

logger = Logger()

class DataPreprocessor:
    def __init__(self, config: Dict[str, Any]):
        """Initialize the data preprocessor."""
        self.config = config
        self.scaler = None
        self.tracker = PreprocessorTracker()
        self.pipeline_steps = []
        
    def create_sequences(self, data: np.ndarray, seq_length: int) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for time series models."""
        X, y = [], []
        for i in range(len(data) - seq_length):
            X.append(data[i:(i + seq_length)])
            y.append(data[i + seq_length])
        return np.array(X), np.array(y)
    
    def add_time_features(self, df: pd.DataFrame, date_column: str = "date") -> pd.DataFrame:
        """Add time-based features to the dataset."""
        logger.info("Adding time-based features")
        try:
            df = df.copy()
            df[date_column] = pd.to_datetime(df[date_column])
            
            # Add basic time features
            df['hour'] = df[date_column].dt.hour
            df['day_of_week'] = df[date_column].dt.dayofweek
            df['day_of_month'] = df[date_column].dt.day
            df['month'] = df[date_column].dt.month
            df['year'] = df[date_column].dt.year
            df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
            
            # Add cyclical features
            df['hour_sin'] = np.sin(2 * np.pi * df['hour']/24)
            df['hour_cos'] = np.cos(2 * np.pi * df['hour']/24)
            df['day_sin'] = np.sin(2 * np.pi * df['day_of_week']/7)
            df['day_cos'] = np.cos(2 * np.pi * df['day_of_week']/7)
            df['month_sin'] = np.sin(2 * np.pi * df['month']/12)
            df['month_cos'] = np.cos(2 * np.pi * df['month']/12)
            
            logger.info("Time-based features added successfully")
            return df
        except Exception as e:
            logger.error(f"Error adding time features: {str(e)}")
            raise
    
    def add_lag_features(self, df: pd.DataFrame, target_column: str, lag_periods: List[int]) -> pd.DataFrame:
        """Add lagged features to the dataset."""
        logger.info(f"Adding lag features for column {target_column}")
        try:
            df = df.copy()
            for lag in lag_periods:
                df[f'{target_column}_lag_{lag}'] = df[target_column].shift(lag)
            logger.info(f"Added {len(lag_periods)} lag features")
            return df.dropna()
        except Exception as e:
            logger.error(f"Error adding lag features: {str(e)}")
            raise
    
    def add_rolling_features(self, df: pd.DataFrame, target_column: str, 
                           windows: List[int], functions: List[str]) -> pd.DataFrame:
        """Add rolling window features to the dataset."""
        logger.info(f"Adding rolling features for column {target_column}")
        try:
            df = df.copy()
            for window in windows:
                for func in functions:
                    if func == 'mean':
                        df[f'{target_column}_rolling_mean_{window}'] = df[target_column].rolling(window=window).mean()
                    elif func == 'std':
                        df[f'{target_column}_rolling_std_{window}'] = df[target_column].rolling(window=window).std()
                    elif func == 'min':
                        df[f'{target_column}_rolling_min_{window}'] = df[target_column].rolling(window=window).min()
                    elif func == 'max':
                        df[f'{target_column}_rolling_max_{window}'] = df[target_column].rolling(window=window).max()
            logger.info(f"Added rolling features for {len(windows)} window sizes")
            return df.dropna()
        except Exception as e:
            logger.error(f"Error adding rolling features: {str(e)}")
            raise
    
    def handle_missing_values(self, df: pd.DataFrame, method: str = 'interpolate') -> pd.DataFrame:
        """Handle missing values in the dataset."""
        logger.info(f"Handling missing values using method: {method}")
        try:
            if method == 'interpolate':
                df = df.interpolate(method='linear')
                logger.info("Missing values interpolated linearly")
            elif method == 'drop':
                df = df.dropna()
                logger.info(f"Dropped {len(df) - len(df.dropna())} rows with missing values")
            elif method == 'fill':
                df = df.fillna(method='ffill').fillna(method='bfill')
                logger.info("Missing values filled using forward and backward fill")
            else:
                logger.error(f"Unsupported missing value handling method: {method}")
                raise ValueError(f"Unsupported method: {method}")
            return df
        except Exception as e:
            logger.error(f"Error handling missing values: {str(e)}")
            raise
    
    def normalize_data(self, data: np.ndarray) -> Tuple[np.ndarray, StandardScaler]:
        """Normalize the data using StandardScaler."""
        return self.scaler.fit_transform(data.reshape(-1, 1)).reshape(-1), self.scaler
    
    def inverse_normalize(self, data: np.ndarray) -> np.ndarray:
        """Inverse transform normalized data."""
        return self.scaler.inverse_transform(data.reshape(-1, 1)).reshape(-1)
    
    def prepare_data_for_training(self, df: pd.DataFrame, target_column: str,
                                seq_length: int = 24) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare data for training time series models."""
        # Add all relevant features
        df = self.add_time_features(df)
        df = self.add_lag_features(df, target_column, [1, 2, 3, 24, 48])
        df = self.add_rolling_features(df, target_column, 
                                     windows=[6, 12, 24], 
                                     functions=['mean', 'std'])
        
        # Handle missing values
        df = self.handle_missing_values(df)
        
        # Normalize features
        feature_columns = [col for col in df.columns if col != target_column]
        X = df[feature_columns].values
        y = df[target_column].values
        
        # Create sequences
        X_seq, y_seq = self.create_sequences(X, seq_length)
        
        return X_seq, y_seq

    def remove_outliers(self, data: pd.DataFrame, column: str, method: str = 'zscore',
                       threshold: float = 3.0) -> pd.DataFrame:
        """Remove outliers from the dataset."""
        logger.info(f"Removing outliers from column {column} using {method} method")
        try:
            if method == 'zscore':
                z_scores = np.abs((data[column] - data[column].mean()) / data[column].std())
                data = data[z_scores < threshold]
                logger.info(f"Removed {len(data) - len(data[z_scores < threshold])} outliers using z-score")
            elif method == 'iqr':
                Q1 = data[column].quantile(0.25)
                Q3 = data[column].quantile(0.75)
                IQR = Q3 - Q1
                data = data[~((data[column] < (Q1 - 1.5 * IQR)) | (data[column] > (Q3 + 1.5 * IQR)))]
                logger.info(f"Removed outliers using IQR method")
            else:
                logger.error(f"Unsupported outlier removal method: {method}")
                raise ValueError(f"Unsupported method: {method}")
            return data
        except Exception as e:
            logger.error(f"Error removing outliers: {str(e)}")
            raise

    def scale_features(self, data: pd.DataFrame, columns: list) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Scale features using StandardScaler."""
        logger.info(f"Scaling features: {columns}")
        try:
            data = data.copy()
            scaler_params = {}
            
            for column in columns:
                if column in data.columns:
                    data[column] = self.scaler.fit_transform(data[[column]])
                    scaler_params[column] = {
                        'mean': self.scaler.mean_[0],
                        'scale': self.scaler.scale_[0]
                    }
                    
            logger.info("Features scaled successfully")
            return data, scaler_params
        except Exception as e:
            logger.error(f"Error scaling features: {str(e)}")
            raise

    def prepare_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare data with tracking."""
        logger.info("Starting data preparation")
        try:
            # Start tracking
            self.tracker.start_tracking()
            self.tracker.log_preprocessing_config(self.config)
            self.tracker.log_missing_values_stats(data)
            self.tracker.log_feature_stats(data)
            
            # Handle missing values
            if self.config.get('handle_missing_values'):
                data = self.handle_missing_values(data)
                self.pipeline_steps.append({
                    "step": "handle_missing_values",
                    "method": self.config.get('missing_values_method', 'mean')
                })
            
            # Remove outliers
            if self.config.get('remove_outliers'):
                data = self.remove_outliers(data)
                self.pipeline_steps.append({
                    "step": "remove_outliers",
                    "method": self.config.get('outlier_method', 'zscore')
                })
            
            # Add time-based features
            if self.config.get('add_time_features'):
                data = self.add_time_features(data)
                self.pipeline_steps.append({
                    "step": "add_time_features",
                    "features": list(data.columns)
                })
            
            # Add lag features
            if self.config.get('add_lag_features'):
                data = self.add_lag_features(data)
                self.pipeline_steps.append({
                    "step": "add_lag_features",
                    "lags": self.config.get('lag_features', [1, 2, 3])
                })
            
            # Add rolling features
            if self.config.get('add_rolling_features'):
                data = self.add_rolling_features(data)
                self.pipeline_steps.append({
                    "step": "add_rolling_features",
                    "windows": self.config.get('rolling_windows', [7, 30])
                })
            
            # Scale features
            if self.config.get('scale_features'):
                data = self.scale_features(data)
                self.pipeline_steps.append({
                    "step": "scale_features",
                    "method": self.config.get('scaling_method', 'standard')
                })
            
            # Log final feature stats
            self.tracker.log_feature_stats(data)
            
            # Log complete pipeline
            self.tracker.log_preprocessing_pipeline(self.pipeline_steps)
            
            logger.info("Data preparation completed successfully")
            return data
            
        except Exception as e:
            logger.error(f"Error during data preparation: {str(e)}")
            raise
        finally:
            self.tracker.end_tracking()
    
    def handle_missing_values(self, data: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values with tracking."""
        method = self.config.get('missing_values_method', 'mean')
        logger.info(f"Handling missing values using {method} method")
        
        try:
            if method == 'mean':
                data = data.fillna(data.mean())
            elif method == 'median':
                data = data.fillna(data.median())
            elif method == 'mode':
                data = data.fillna(data.mode().iloc[0])
            elif method == 'ffill':
                data = data.fillna(method='ffill')
            elif method == 'bfill':
                data = data.fillna(method='bfill')
            
            self.tracker.log_model_params({
                "missing_values_method": method,
                "columns_handled": list(data.columns)
            })
            
            return data
        except Exception as e:
            logger.error(f"Error handling missing values: {str(e)}")
            raise
    
    def remove_outliers(self, data: pd.DataFrame) -> pd.DataFrame:
        """Remove outliers with tracking."""
        method = self.config.get('outlier_method', 'zscore')
        threshold = self.config.get('outlier_threshold', 3)
        logger.info(f"Removing outliers using {method} method with threshold {threshold}")
        
        try:
            if method == 'zscore':
                z_scores = np.abs((data - data.mean()) / data.std())
                data = data[(z_scores < threshold).all(axis=1)]
            elif method == 'iqr':
                Q1 = data.quantile(0.25)
                Q3 = data.quantile(0.75)
                IQR = Q3 - Q1
                data = data[~((data < (Q1 - 1.5 * IQR)) | (data > (Q3 + 1.5 * IQR))).any(axis=1)]
            
            n_outliers = len(data) - len(data)
            self.tracker.log_model_params({
                "outlier_method": method,
                "outlier_threshold": threshold,
                "n_outliers_removed": n_outliers,
                "outlier_percentage": (n_outliers / len(data)) * 100
            })
            
            return data
        except Exception as e:
            logger.error(f"Error removing outliers: {str(e)}")
            raise
    
    def add_time_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add time-based features with tracking."""
        logger.info("Adding time-based features")
        
        try:
            if 'timestamp' in data.columns:
                data['hour'] = data['timestamp'].dt.hour
                data['day'] = data['timestamp'].dt.day
                data['month'] = data['timestamp'].dt.month
                data['year'] = data['timestamp'].dt.year
                data['dayofweek'] = data['timestamp'].dt.dayofweek
                data['quarter'] = data['timestamp'].dt.quarter
                
                self.tracker.log_model_params({
                    "time_features_added": ['hour', 'day', 'month', 'year', 'dayofweek', 'quarter']
                })
            
            return data
        except Exception as e:
            logger.error(f"Error adding time features: {str(e)}")
            raise
    
    def add_lag_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add lag features with tracking."""
        lags = self.config.get('lag_features', [1, 2, 3])
        logger.info(f"Adding lag features with lags: {lags}")
        
        try:
            target_col = self.config.get('target_column', 'value')
            for lag in lags:
                data[f'{target_col}_lag_{lag}'] = data[target_col].shift(lag)
            
            self.tracker.log_model_params({
                "lag_features_added": [f'{target_col}_lag_{lag}' for lag in lags]
            })
            
            return data
        except Exception as e:
            logger.error(f"Error adding lag features: {str(e)}")
            raise
    
    def add_rolling_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add rolling features with tracking."""
        windows = self.config.get('rolling_windows', [7, 30])
        logger.info(f"Adding rolling features with windows: {windows}")
        
        try:
            target_col = self.config.get('target_column', 'value')
            for window in windows:
                data[f'{target_col}_rolling_mean_{window}'] = data[target_col].rolling(window=window).mean()
                data[f'{target_col}_rolling_std_{window}'] = data[target_col].rolling(window=window).std()
            
            self.tracker.log_model_params({
                "rolling_features_added": [
                    f'{target_col}_rolling_mean_{window}' for window in windows
                ] + [
                    f'{target_col}_rolling_std_{window}' for window in windows
                ]
            })
            
            return data
        except Exception as e:
            logger.error(f"Error adding rolling features: {str(e)}")
            raise
    
    def scale_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Scale features with tracking."""
        method = self.config.get('scaling_method', 'standard')
        columns = self.config.get('scaled_columns', data.columns)
        logger.info(f"Scaling features using {method} method")
        
        try:
            if method == 'standard':
                self.scaler = StandardScaler()
                data[columns] = self.scaler.fit_transform(data[columns])
            elif method == 'minmax':
                self.scaler = MinMaxScaler()
                data[columns] = self.scaler.fit_transform(data[columns])
            
            self.tracker.log_scaling_params(
                method=method,
                columns=list(columns),
                params={
                    "scaler_type": type(self.scaler).__name__,
                    "n_features": len(columns)
                }
            )
            
            return data
        except Exception as e:
            logger.error(f"Error scaling features: {str(e)}")
            raise 