import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from .preprocessing import DataPreprocessor
from ..utils.logger import Logger

logger = Logger()

class DataLoader:
    def __init__(self, config: Dict[str, Any]):
        logger.info(f"Initializing DataLoader with config: {config}")
        self.config = config
        self.preprocessor = DataPreprocessor()
        
    def load_csv(self, file_path: str, date_column: str = 'timestamp',
                 value_column: str = 'consumption') -> pd.DataFrame:
        """Load data from CSV file."""
        logger.info(f"Loading data from file: {file_path}")
        try:
            df = pd.read_csv(file_path)
            df[date_column] = pd.to_datetime(df[date_column])
            logger.info(f"Successfully loaded {len(df)} rows of data")
            return df.sort_values(date_column)
        except Exception as e:
            logger.error(f"Error loading data from {file_path}: {str(e)}")
            raise
    
    def load_from_database(self, start_date: datetime, end_date: datetime,
                          db_session) -> pd.DataFrame:
        """Load data from database."""
        from ..core.database import ConsumptionData
        
        query = db_session.query(ConsumptionData).filter(
            ConsumptionData.timestamp.between(start_date, end_date)
        ).order_by(ConsumptionData.timestamp)
        
        return pd.read_sql(query.statement, query.session.bind)
    
    def prepare_training_data(self, df: pd.DataFrame, 
                            target_column: str = 'consumption',
                            date_column: str = 'timestamp',
                            test_size: float = 0.2,
                            sequence_length: int = 24) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Prepare data for training."""
        logger.info(f"Preparing training data with sequence length {sequence_length}")
        try:
            # Use the preprocessing method to prepare the data
            X_seq, y_seq = self.preprocessor.prepare_data_for_training(
                df, target_column, sequence_length
            )
            
            # Split into train and test sets
            logger.info(f"Splitting data with test size {test_size}")
            X_train, X_test, y_train, y_test = train_test_split(
                X_seq, y_seq, test_size=test_size, shuffle=False
            )
            
            logger.info(f"Data prepared: {len(X_train)} training and {len(X_test)} test sequences")
            return X_train, X_test, y_train, y_test
        except Exception as e:
            logger.error(f"Error preparing training data: {str(e)}")
            raise
    
    def prepare_forecast_data(self, df: pd.DataFrame, 
                            target_column: str = 'consumption',
                            date_column: str = 'timestamp',
                            sequence_length: int = 24) -> Tuple[np.ndarray, List[datetime]]:
        """Prepare data for forecasting."""
        logger.info(f"Preparing forecast data with sequence length {sequence_length}")
        try:
            # Use the preprocessing method to prepare the data
            X_seq, _ = self.preprocessor.prepare_data_for_training(
                df, target_column, sequence_length
            )
            
            # Get the corresponding dates
            dates = df[date_column].iloc[sequence_length:].tolist()
            
            logger.info(f"Prepared {len(X_seq)} sequences for forecasting")
            return X_seq, dates
        except Exception as e:
            logger.error(f"Error preparing forecast data: {str(e)}")
            raise
    
    def generate_future_features(self, last_data: pd.DataFrame,
                               forecast_horizon: int,
                               target_column: str = 'consumption',
                               date_column: str = 'timestamp') -> pd.DataFrame:
        """Generate feature matrix for future dates."""
        last_date = last_data[date_column].max()
        future_dates = [last_date + timedelta(hours=i+1) for i in range(forecast_horizon)]
        
        # Create future DataFrame
        future_df = pd.DataFrame({date_column: future_dates})
        
        # Add time features
        future_df = self.preprocessor.add_time_features(future_df, date_column)
        
        # Add the last known values for lag features
        for lag in [1, 2, 3, 24, 48]:
            future_df[f'lag_{lag}'] = np.nan
        
        # Add the last known values for rolling features
        for window in [6, 12, 24]:
            for func in ['mean', 'std']:
                future_df[f'rolling_{func}_{window}'] = np.nan
        
        return future_df
    
    def save_to_database(self, df: pd.DataFrame, db_session) -> None:
        """Save data to database."""
        from ..core.database import ConsumptionData
        
        for _, row in df.iterrows():
            data = ConsumptionData(
                timestamp=row['timestamp'],
                consumption=row['consumption'],
                temperature=row.get('temperature'),
                humidity=row.get('humidity'),
                is_holiday=row.get('is_holiday', 0),
                day_type=row.get('day_type', 'workday')
            )
            db_session.add(data)
        
        db_session.commit()

    def load_data(self, file_path: str) -> pd.DataFrame:
        """Load data from a CSV file."""
        logger.info(f"Loading data from file: {file_path}")
        try:
            data = pd.read_csv(file_path)
            logger.info(f"Successfully loaded {len(data)} rows of data")
            return data
        except Exception as e:
            logger.error(f"Error loading data from {file_path}: {str(e)}")
            raise
            
    def prepare_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for the model."""
        logger.info("Preparing features from data")
        try:
            # Add time-based features
            data['hour'] = data['date'].dt.hour
            data['day_of_week'] = data['date'].dt.dayofweek
            data['month'] = data['date'].dt.month
            data['is_weekend'] = data['day_of_week'].isin([5, 6]).astype(int)
            
            # Add lag features
            for lag in self.config.get('lags', [1, 24, 168]):  # 1h, 24h, 1w
                data[f'lag_{lag}'] = data['value'].shift(lag)
                
            # Add rolling statistics
            for window in self.config.get('windows', [24, 168]):  # 1d, 1w
                data[f'rolling_mean_{window}'] = data['value'].rolling(window=window).mean()
                data[f'rolling_std_{window}'] = data['value'].rolling(window=window).std()
                
            # Drop rows with NaN values
            data = data.dropna()
            logger.info(f"Prepared features for {len(data)} rows")
            return data
        except Exception as e:
            logger.error(f"Error preparing features: {str(e)}")
            raise
            
    def split_data(self, data: pd.DataFrame, train_ratio: float = 0.8) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Split data into training and testing sets."""
        logger.info(f"Splitting data with train ratio: {train_ratio}")
        try:
            train_size = int(len(data) * train_ratio)
            train_data = data[:train_size]
            test_data = data[train_size:]
            logger.info(f"Split data into {len(train_data)} training and {len(test_data)} testing samples")
            return train_data, test_data
        except Exception as e:
            logger.error(f"Error splitting data: {str(e)}")
            raise
            
    def create_sequences(self, data: pd.DataFrame, sequence_length: int) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for time series prediction."""
        logger.info(f"Creating sequences with length: {sequence_length}")
        try:
            X, y = [], []
            for i in range(len(data) - sequence_length):
                X.append(data.iloc[i:(i + sequence_length)].values)
                y.append(data.iloc[i + sequence_length]['value'])
            X = np.array(X)
            y = np.array(y)
            logger.info(f"Created {len(X)} sequences")
            return X, y
        except Exception as e:
            logger.error(f"Error creating sequences: {str(e)}")
            raise
            
    def normalize_data(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Normalize the data and return normalization parameters."""
        logger.info("Normalizing data")
        try:
            params = {}
            for column in data.columns:
                if column not in ['date', 'value']:
                    mean = data[column].mean()
                    std = data[column].std()
                    data[column] = (data[column] - mean) / std
                    params[column] = {'mean': mean, 'std': std}
            logger.info("Data normalization completed")
            return data, params
        except Exception as e:
            logger.error(f"Error normalizing data: {str(e)}")
            raise
            
    def denormalize_predictions(self, predictions: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        """Denormalize predictions using the stored parameters."""
        logger.info("Denormalizing predictions")
        try:
            mean = params['value']['mean']
            std = params['value']['std']
            denormalized = predictions * std + mean
            logger.info("Predictions denormalized successfully")
            return denormalized
        except Exception as e:
            logger.error(f"Error denormalizing predictions: {str(e)}")
            raise
            
    def save_data(self, data: pd.DataFrame, file_path: str):
        """Save processed data to a CSV file."""
        logger.info(f"Saving data to file: {file_path}")
        try:
            data.to_csv(file_path, index=False)
            logger.info("Data saved successfully")
        except Exception as e:
            logger.error(f"Error saving data to {file_path}: {str(e)}")
            raise 