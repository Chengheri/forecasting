import pandas as pd
import numpy as np
from typing import Tuple, List, Dict, Any, Optional, Union
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from datetime import datetime, timedelta
from ..utils.logger import Logger
from .trackers import PreprocessorTracker
from .mlflow_utils import MLflowTracker
from statsmodels.tsa.stattools import adfuller, kpss

logger = Logger()

class DataPreprocessor:
    def __init__(self, config: Dict[str, Any], tracker: Optional[MLflowTracker] = None, experiment_name: str = "electricity_forecasting"):
        """Initialize the data preprocessor.
        
        Args:
            config: Dictionary containing preprocessing configuration
            tracker: MLflow tracker instance (optional)
            experiment_name: Name of the MLflow experiment (used only if tracker is None)
        """
        self.config = config
        self.scaler = None
        self.tracker = tracker if tracker else PreprocessorTracker(experiment_name=experiment_name)
        self.pipeline_steps = []
        
    def train_test_split_timeseries(self, 
                                  data: pd.DataFrame, 
                                  train_ratio: float = 0.8, 
                                  validation_ratio: float = 0.0,
                                  target_column: Optional[str] = None,
                                  date_column: Optional[str] = None,
                                  gap: int = 0,
                                  shuffle: bool = False) -> Union[Tuple[pd.DataFrame, pd.DataFrame], Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]]:
        """Split time series data into training and test sets with optional validation set.
        
        This function handles time series data properly by maintaining the temporal order
        and providing options for validation sets and gaps between splits.
        
        Args:
            data (pd.DataFrame): The input time series data
            train_ratio (float): Proportion of data to use for training (default: 0.8)
            validation_ratio (float): Proportion of data to use for validation (default: 0.0)
                                    If 0, no validation set is returned
            target_column (str, optional): Name of the target column to be predicted
                                        If provided, will log statistics about the target
            date_column (str, optional): Name of the date/timestamp column
                                      If provided, will log date ranges for each split
            gap (int): Number of time steps to skip between train/validation/test sets (default: 0)
                    Useful for forecasting models to simulate realistic prediction scenarios
            shuffle (bool): Whether to shuffle the data before splitting (default: False)
                        Note: For most time series tasks, shuffling is not recommended
        
        Returns:
            If validation_ratio > 0:
                Tuple of (train_data, validation_data, test_data)
            Else:
                Tuple of (train_data, test_data)
        
        Raises:
            ValueError: If invalid ratio values are provided or if the resulting sets would be empty
        """
        logger.info(f"Splitting time series data with train_ratio={train_ratio}, validation_ratio={validation_ratio}, gap={gap}")
        
        # Validate input parameters
        if train_ratio <= 0 or train_ratio >= 1:
            raise ValueError(f"train_ratio must be between 0 and 1, got {train_ratio}")
        
        if validation_ratio < 0 or validation_ratio >= 1:
            raise ValueError(f"validation_ratio must be between 0 and 1, got {validation_ratio}")
        
        if train_ratio + validation_ratio >= 1:
            raise ValueError(f"Sum of train_ratio and validation_ratio must be less than 1, got {train_ratio + validation_ratio}")
        
        # Make a copy to avoid modifying the original data
        data_copy = data.copy()
        
        # Shuffle if requested (not typical for time series)
        if shuffle:
            logger.warning("Shuffling time series data - this is not recommended for most time series tasks")
            data_copy = data_copy.sample(frac=1).reset_index(drop=True)
        
        # Calculate split indices
        n = len(data_copy)
        train_end = int(n * train_ratio)
        
        if train_end <= 0:
            raise ValueError(f"Training set would be empty with ratio {train_ratio} and data length {n}")
        
        # Create training set
        train_data = data_copy.iloc[:train_end].copy()
        
        # Handle validation set if requested
        if validation_ratio > 0:
            val_start = train_end + gap
            val_end = val_start + int(n * validation_ratio)
            
            if val_start >= n or val_end > n:
                raise ValueError(f"Validation set would be empty with current parameters")
            
            val_data = data_copy.iloc[val_start:val_end].copy()
            test_start = val_end + gap
        else:
            val_data = None
            test_start = train_end + gap
        
        # Create test set
        if test_start >= n:
            raise ValueError(f"Test set would be empty with current parameters")
        
        test_data = data_copy.iloc[test_start:].copy()
        
        # Track split information
        split_info = {
            "preprocessing.split.train_ratio": train_ratio,
            "preprocessing.split.validation_ratio": validation_ratio,
            "preprocessing.split.gap": gap,
            "preprocessing.split.shuffle": shuffle,
            "preprocessing.split.train_size": len(train_data),
            "preprocessing.split.test_size": len(test_data),
            "preprocessing.split.validation_size": len(val_data) if val_data is not None else 0,
            "preprocessing.split.total_size": n
        }
        
        # Add date ranges if available
        if date_column and date_column in data_copy.columns:
            split_info.update({
                "preprocessing.split.train_start": str(train_data[date_column].min()),
                "preprocessing.split.train_end": str(train_data[date_column].max()),
                "preprocessing.split.test_start": str(test_data[date_column].min()),
                "preprocessing.split.test_end": str(test_data[date_column].max())
            })
            if val_data is not None:
                split_info.update({
                    "preprocessing.split.validation_start": str(val_data[date_column].min()),
                    "preprocessing.split.validation_end": str(val_data[date_column].max())
                })
        
        # Add target statistics if available
        if target_column and target_column in data_copy.columns:
            split_info.update({
                "preprocessing.split.train_target_mean": float(train_data[target_column].mean()),
                "preprocessing.split.train_target_std": float(train_data[target_column].std()),
                "preprocessing.split.train_target_min": float(train_data[target_column].min()),
                "preprocessing.split.train_target_max": float(train_data[target_column].max()),
                "preprocessing.split.test_target_mean": float(test_data[target_column].mean()),
                "preprocessing.split.test_target_std": float(test_data[target_column].std()),
                "preprocessing.split.test_target_min": float(test_data[target_column].min()),
                "preprocessing.split.test_target_max": float(test_data[target_column].max())
            })
            if val_data is not None:
                split_info.update({
                    "preprocessing.split.validation_target_mean": float(val_data[target_column].mean()),
                    "preprocessing.split.validation_target_std": float(val_data[target_column].std()),
                    "preprocessing.split.validation_target_min": float(val_data[target_column].min()),
                    "preprocessing.split.validation_target_max": float(val_data[target_column].max())
                })
        
        # Log split information to MLflow
        #self.tracker.log_params_safely(split_info)
        
        # Add split step to pipeline
        self.pipeline_steps.append({
            "step": "train_test_split",
            "train_ratio": train_ratio,
            "validation_ratio": validation_ratio,
            "gap": gap,
            "shuffle": shuffle
        })
        
        # Return appropriate tuple based on whether validation set was requested
        if validation_ratio > 0:
            return train_data, val_data, test_data
        else:
            return train_data, test_data
    
    @property
    def experiment_name(self) -> str:
        """Get the experiment name from the tracker."""
        return self.tracker.experiment_name
    
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
            
            # Get datetime series (either from column or index)
            if date_column == df.index.name or date_column in df.index.names:
                datetime_series = df.index
            else:
                datetime_series = pd.to_datetime(df[date_column])
                df[date_column] = datetime_series
            
            # Add basic time features
            df['hour'] = datetime_series.hour
            df['day_of_week'] = datetime_series.dayofweek
            df['day_of_month'] = datetime_series.day
            df['month'] = datetime_series.month
            df['year'] = datetime_series.year
            df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
            
            # Add cyclical features
            df['hour_sin'] = np.sin(2 * np.pi * df['hour']/24)
            df['hour_cos'] = np.cos(2 * np.pi * df['hour']/24)
            df['day_sin'] = np.sin(2 * np.pi * df['day_of_week']/7)
            df['day_cos'] = np.cos(2 * np.pi * df['day_of_week']/7)
            df['month_sin'] = np.sin(2 * np.pi * df['month']/12)
            df['month_cos'] = np.cos(2 * np.pi * df['month']/12)
            
            # Track time features
            time_features = {
                'basic_features': ['hour', 'day_of_week', 'day_of_month', 'month', 'year', 'is_weekend'],
                'cyclical_features': ['hour_sin', 'hour_cos', 'day_sin', 'day_cos', 'month_sin', 'month_cos']
            }
            self.tracker.log_feature_engineering(
                feature_type='time',
                added_features=time_features['basic_features'] + time_features['cyclical_features']
            )
            
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
            lag_features = []
            for lag in lag_periods:
                feature_name = f'{target_column}_lag_{lag}'
                df[feature_name] = df[target_column].shift(lag)
                lag_features.append(feature_name)
            
            # Track lag features
            self.tracker.log_feature_engineering(
                feature_type='lag',
                added_features=lag_features
            )
            
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
            rolling_features = []
            for window in windows:
                for func in functions:
                    feature_name = f'{target_column}_rolling_{func}_{window}'
                    if func == 'mean':
                        df[feature_name] = df[target_column].rolling(window=window).mean()
                    elif func == 'std':
                        df[feature_name] = df[target_column].rolling(window=window).std()
                    elif func == 'min':
                        df[feature_name] = df[target_column].rolling(window=window).min()
                    elif func == 'max':
                        df[feature_name] = df[target_column].rolling(window=window).max()
                    rolling_features.append(feature_name)
            
            # Track rolling features
            self.tracker.log_feature_engineering(
                feature_type='rolling',
                added_features=rolling_features
            )
            
            logger.info(f"Added rolling features for {len(windows)} window sizes")
            return df.dropna()
        except Exception as e:
            logger.error(f"Error adding rolling features: {str(e)}")
            raise
    
    def handle_missing_values(self, df: pd.DataFrame, method: str = 'interpolate') -> pd.DataFrame:
        """Handle missing values in the dataset."""
        logger.info(f"Handling missing values using method: {method}")
        try:
            initial_missing = df.isnull().sum().to_dict()
            
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
            
            final_missing = df.isnull().sum().to_dict()
            
            # Track missing values handling
            self.tracker.log_missing_values_stats(df)
            self.tracker.log_params_safely({
                "preprocessing.missing_values.method": method,
                "preprocessing.missing_values.initial_count": initial_missing,
                "preprocessing.missing_values.final_count": final_missing
            })
            
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
    
    def preprocess_data(self, df: pd.DataFrame, target_column: str = 'value', create_sequences: bool = False, seq_length: int = 24) -> Union[pd.DataFrame, Tuple[np.ndarray, np.ndarray]]:
        """Prepare data for modeling with comprehensive preprocessing options."""
        logger.info("Starting data preparation")
        try:
            # Track preprocessing steps
            self.pipeline_steps = []
            
            # Track initial data stats
            initial_stats = self._log_data_stats(df, "initial")
            
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
                    'target_column': target_column
                })
                
                from .advanced_preprocessing import AdvancedPreprocessor
                advanced_preprocessor = AdvancedPreprocessor(
                    config=outlier_config,
                    tracker=self.tracker
                )
                
                # Process anomalies (detect and optionally clean)
                df = advanced_preprocessor.process_anomalies(
                    data=df,
                    target_column=target_column,
                    method=outlier_config['method'],
                    contamination=outlier_config.get('contamination', 0.1),
                    handle_anomalies=self.config.get('clean_anomalies', True)
                )
                
                # Remove anomaly column if not needed
                if 'is_anomaly' in df.columns and not self.config.get('keep_anomaly_labels', False):
                    df = df.drop('is_anomaly', axis=1)
                
                self.pipeline_steps.append({
                    "step": "process_anomalies",
                    "method": outlier_config['method'],
                    "handled_anomalies": self.config.get('clean_anomalies', True),
                    "kept_labels": self.config.get('keep_anomaly_labels', False)
                })
            
            # Add time features if configured
            if self.config.get('add_time_features', True):
                date_column = self.config.get('date_column', 'date')
                df = self.add_time_features(df, date_column)
                self.pipeline_steps.append({"step": "add_time_features"})
            
            # Add lag features if configured
            if self.config.get('add_lag_features', True):
                lag_periods = self.config.get('lag_features', [1, 2, 3, 24, 48])
                df = self.add_lag_features(df, target_column, lag_periods)
                self.pipeline_steps.append({
                    "step": "add_lag_features",
                    "lags": lag_periods
                })
            
            # Add rolling features if configured
            if self.config.get('add_rolling_features', True):
                windows = self.config.get('rolling_windows', [6, 12, 24])
                functions = self.config.get('rolling_functions', ['mean', 'std'])
                df = self.add_rolling_features(df, target_column, windows, functions)
                self.pipeline_steps.append({
                    "step": "add_rolling_features",
                    "windows": windows,
                    "functions": functions
                })
            
            # Scale features if configured
            if self.config.get('scale_features', False):
                df = self.scale_features(df)
                self.pipeline_steps.append({
                    "step": "scale_features",
                    "method": self.config.get('scaling_method', 'standard')
                })
            
            # Track final data stats
            final_stats = self._log_data_stats(df, "final")
            
            # Log preprocessing pipeline
            self.tracker.log_preprocessing_pipeline(self.pipeline_steps)
            
            # Create sequences if requested
            if create_sequences:
                feature_columns = [col for col in df.columns if col != target_column]
                X = df[feature_columns].values
                y = df[target_column].values
                X_seq, y_seq = self.create_sequences(X, seq_length)
                
                # Track sequence creation
                self.tracker.log_params_safely({
                    "preprocessing.sequences.length": seq_length,
                    "preprocessing.sequences.n_features": len(feature_columns),
                    "preprocessing.sequences.n_sequences": len(X_seq)
                })
                
                return X_seq, y_seq
            
            logger.info("Data preparation completed successfully")
            return df
            
        except Exception as e:
            logger.error(f"Error in data preparation: {str(e)}")
            raise

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
            original_stats = {
                col: {
                    'mean': float(data[col].mean()),
                    'std': float(data[col].std()),
                    'min': float(data[col].min()),
                    'max': float(data[col].max())
                } for col in columns
            }
            
            if method == 'standard':
                self.scaler = StandardScaler()
                data[columns] = self.scaler.fit_transform(data[columns])
            elif method == 'minmax':
                self.scaler = MinMaxScaler()
                data[columns] = self.scaler.fit_transform(data[columns])
            
            scaled_stats = {
                col: {
                    'mean': float(data[col].mean()),
                    'std': float(data[col].std()),
                    'min': float(data[col].min()),
                    'max': float(data[col].max())
                } for col in columns
            }
            
            # Track scaling information
            self.tracker.log_params_safely({
                "preprocessing.scaling.method": method,
                "preprocessing.scaling.n_features": len(columns),
                "preprocessing.scaling.features": list(columns),
                "preprocessing.scaling.original_stats": original_stats,
                "preprocessing.scaling.scaled_stats": scaled_stats
            })
            
            return data
        except Exception as e:
            logger.error(f"Error scaling features: {str(e)}")
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

        # Add prefix to all stats
        prefixed_stats = {f"preprocessing.data.{k}": v for k, v in stats.items()}
        #self.tracker.log_params_safely(prefixed_stats)
        return stats 

    