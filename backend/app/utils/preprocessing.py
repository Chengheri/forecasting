import pandas as pd
import numpy as np
from typing import Tuple, List, Dict, Any, Optional
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from datetime import datetime, timedelta
from ..utils.logger import Logger
from .trackers import PreprocessorTracker

logger = Logger()

class DataPreprocessor:
    def __init__(self, config: Dict[str, Any], experiment_name: str = "electricity_forecasting", run_name: Optional[str] = None):
        """Initialize the data preprocessor.
        
        Args:
            config: Dictionary containing preprocessing configuration
            experiment_name: Name of the MLflow experiment
            run_name: Name of the MLflow run (optional)
        """
        self.config = config
        self.scaler = None
        self.tracker = PreprocessorTracker(experiment_name=experiment_name, run_name=run_name)
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

    def scale_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Scale features with tracking."""
        method = self.config.get('scaling_method', 'standard')
        columns = self.config.get('columns_to_scale', data.select_dtypes(include=[np.number]).columns)
        logger.info(f"Scaling features using {method} method")
        
        try:
            if method == 'standard':
                self.scaler = StandardScaler()
                data[columns] = self.scaler.fit_transform(data[columns])
            elif method == 'minmax':
                self.scaler = MinMaxScaler()
                data[columns] = self.scaler.fit_transform(data[columns])
            
            self.tracker.log_model_params({
                "scaling_method": method,
                "scaled_columns": list(columns),
                "scaler_type": type(self.scaler).__name__,
                "n_features_scaled": len(columns)
            })
            
            return data
        except Exception as e:
            logger.error(f"Error scaling features: {str(e)}")
            raise

    def prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare data for modeling with outlier and anomaly detection."""
        logger.info("Starting data preparation")
        try:
            # Track preprocessing steps
            self.pipeline_steps = []
            
            # Handle missing values if configured
            if self.config.get('handle_missing_values', False):
                method = self.config.get('missing_values_method', 'interpolate')
                df = self.handle_missing_values(df, method)
                self.pipeline_steps.append({"step": "handle_missing_values", "method": method})
            
            # Handle outliers and anomalies if configured
            if self.config.get('handle_outliers', False):
                outlier_config = self.config.get('outlier_config', {
                    'method': 'isolation_forest',
                    'contamination': 0.1,
                    'target_column': 'value'
                })
                
                from .advanced_preprocessing import AdvancedPreprocessor
                advanced_preprocessor = AdvancedPreprocessor(
                    config=outlier_config,
                    experiment_name=self.tracker.experiment_name,
                    run_name=self.tracker.run_name
                )
                
                # Detect and handle anomalies
                df = advanced_preprocessor.detect_anomalies(
                    data=df,
                    target_column=outlier_config['target_column'],
                    method=outlier_config['method'],
                    contamination=outlier_config['contamination']
                )
                
                # Clean anomalous data
                if self.config.get('remove_anomalies', False):
                    df = df[~df['is_anomaly']].copy()
                    df = df.drop('is_anomaly', axis=1)
                
                self.pipeline_steps.append({
                    "step": "handle_outliers",
                    "method": outlier_config['method'],
                    "removed": self.config.get('remove_anomalies', False)
                })
            
            # Add time features if configured
            if self.config.get('add_time_features', False):
                df = self.add_time_features(df)
                self.pipeline_steps.append({"step": "add_time_features"})
            
            # Add lag features if configured
            if self.config.get('add_lag_features', False):
                lag_features = self.config.get('lag_features', [1, 2, 3])
                df = self.add_lag_features(df, 'value', lag_features)
                self.pipeline_steps.append({
                    "step": "add_lag_features",
                    "lags": lag_features
                })
            
            # Add rolling features if configured
            if self.config.get('add_rolling_features', False):
                windows = self.config.get('rolling_windows', [24, 168])
                functions = self.config.get('rolling_functions', ['mean', 'std'])
                df = self.add_rolling_features(df, 'value', windows, functions)
                self.pipeline_steps.append({
                    "step": "add_rolling_features",
                    "windows": windows,
                    "functions": functions
                })
            
            # Scale features if configured
            if self.config.get('scale_features', False):
                columns_to_scale = self.config.get('columns_to_scale', ['value'])
                scaling_method = self.config.get('scaling_method', 'standard')
                df = self.scale_features(df)
                self.pipeline_steps.append({
                    "step": "scale_features",
                    "method": scaling_method,
                    "columns": columns_to_scale
                })
            
            # Log preprocessing steps
            self.tracker.log_preprocessing_config({
                "preprocessing_steps": self.pipeline_steps,
                "total_steps": len(self.pipeline_steps)
            })
            
            logger.info("Data preparation completed successfully")
            return df
            
        except Exception as e:
            logger.error(f"Error in data preparation: {str(e)}")
            raise

    def _log_data_stats(self, data, prefix=""):
        """Log statistics for each column in the data."""
        stats = {}
        for column in data.columns:
            safe_column = column.replace(":", "_").replace(" ", "_").replace("%", "pct")
            
            # Basic statistics for all columns
            stats[f"{prefix}_{safe_column}_count"] = len(data[column])
            
            if pd.api.types.is_bool_dtype(data[column]):
                # For boolean columns, calculate proportion of True values
                stats[f"{prefix}_{safe_column}_true_ratio"] = float(data[column].mean())
            elif pd.api.types.is_numeric_dtype(data[column]):
                # For numeric columns (excluding boolean), calculate standard statistics
                stats[f"{prefix}_{safe_column}_mean"] = float(data[column].mean())
                stats[f"{prefix}_{safe_column}_std"] = float(data[column].std())
                stats[f"{prefix}_{safe_column}_min"] = float(data[column].min())
                stats[f"{prefix}_{safe_column}_25pct"] = float(data[column].quantile(0.25))
                stats[f"{prefix}_{safe_column}_50pct"] = float(data[column].quantile(0.50))
                stats[f"{prefix}_{safe_column}_75pct"] = float(data[column].quantile(0.75))
                stats[f"{prefix}_{safe_column}_max"] = float(data[column].max())
                stats[f"{prefix}_{safe_column}_skew"] = float(data[column].skew())
                stats[f"{prefix}_{safe_column}_kurtosis"] = float(data[column].kurtosis())
            elif pd.api.types.is_datetime64_any_dtype(data[column]):
                # For datetime columns, only log basic info
                stats[f"{prefix}_{safe_column}_min"] = str(data[column].min())
                stats[f"{prefix}_{safe_column}_max"] = str(data[column].max())
            else:
                # For other types (categorical, object, etc.)
                stats[f"{prefix}_{safe_column}_unique"] = data[column].nunique()
                if not data[column].empty:
                    top_value = data[column].mode().iloc[0]
                    stats[f"{prefix}_{safe_column}_top"] = str(top_value)
                    stats[f"{prefix}_{safe_column}_freq"] = int(data[column].value_counts().iloc[0])

        self.tracker.log_model_params(stats) 