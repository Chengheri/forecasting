import pandas as pd
import numpy as np
from typing import Tuple, List, Dict, Any, Optional
from sklearn.preprocessing import StandardScaler
from datetime import datetime, timedelta
from ..utils.logger import Logger

logger = Logger()

class DataPreprocessor:
    def __init__(self):
        logger.info("Initializing DataPreprocessor")
        self.scaler = StandardScaler()
        
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

    def prepare_data(self, data: pd.DataFrame, config: Dict[str, Any]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Prepare data according to the configuration."""
        logger.info("Starting data preparation")
        try:
            # Handle missing values
            data = self.handle_missing_values(data, config.get('missing_values_method', 'interpolate'))
            
            # Detect and handle anomalies using advanced preprocessing
            if 'anomaly_detection' in config:
                logger.info("Using advanced anomaly detection")
                from .advanced_preprocessing import AdvancedPreprocessor
                advanced_preprocessor = AdvancedPreprocessor()
                
                anomaly_config = config['anomaly_detection']
                data = advanced_preprocessor.detect_anomalies(
                    data,
                    target_column=config['target_column'],
                    method=anomaly_config.get('method', 'isolation_forest'),
                    window=anomaly_config.get('window', 24),
                    threshold=anomaly_config.get('threshold', 3.0),
                    contamination=anomaly_config.get('contamination', 0.1)
                )
                
                # Clean detected anomalies if specified
                if anomaly_config.get('clean_anomalies', False):
                    logger.info("Cleaning detected anomalies")
                    data = advanced_preprocessor.clean_data(
                        data,
                        target_column=config['target_column'],
                        features=[config['target_column']]
                    )
            else:
                # Use basic outlier removal if no advanced anomaly detection specified
                if 'outlier_removal' in config:
                    data = self.remove_outliers(
                        data,
                        config['outlier_removal']['column'],
                        config['outlier_removal']['method'],
                        config['outlier_removal'].get('threshold', 3.0)
                    )
            
            # Add time features
            data = self.add_time_features(data, config.get('date_column', 'date'))
            
            # Add lag features
            if 'lags' in config:
                data = self.add_lag_features(data, config['target_column'], config['lags'])
            
            # Add rolling features
            if 'rolling_windows' in config:
                data = self.add_rolling_features(data, config['target_column'], config['rolling_windows'], ['mean', 'std'])
            
            # Scale features
            if 'scale_columns' in config:
                data, scaler_params = self.scale_features(data, config['scale_columns'])
            else:
                scaler_params = {}
            
            logger.info("Data preparation completed successfully")
            return data, scaler_params
        except Exception as e:
            logger.error(f"Error preparing data: {str(e)}")
            raise 