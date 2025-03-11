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

    def load_csv(self, file_path: str = "", date_column: str = None,
                 value_column: str = None) -> pd.DataFrame:
        """Load data from CSV file.
        
        Args:
            file_path (str): Path to the CSV file. If empty, uses path from config.
            date_column (str): Name of the date column. If None, uses from config.
            value_column (str): Name of the value column. If None, uses from config.
            
        Returns:
            pd.DataFrame: Loaded and formatted DataFrame with datetime index.
            
        Raises:
            FileNotFoundError: If the CSV file is not found.
            ValueError: If required columns are missing.
        """
        try:
            # Use parameters from config if not provided
            actual_path = file_path if file_path else self.config['data']['path']
            actual_date_column = date_column if date_column else self.config['data']['index_column']
            
            logger.info(f"Loading data from file: {actual_path}")
            logger.debug(f"Using date column: {actual_date_column}")
            
            # Load and parse the CSV file
            data = pd.read_csv(actual_path, parse_dates=[actual_date_column])
            
            # Validate data
            if actual_date_column not in data.columns:
                raise ValueError(f"Date column '{actual_date_column}' not found in CSV")
            
            # Sort and set index
            data = data.sort_values(actual_date_column)
            data.set_index(actual_date_column, inplace=True)
            
            logger.info(f"Successfully loaded {len(data)} rows of data")
            logger.debug(f"DataFrame columns: {list(data.columns)}")
            
            return data
            
        except FileNotFoundError as e:
            logger.error(f"CSV file not found at {actual_path}")
            raise
        except pd.errors.EmptyDataError as e:
            logger.error(f"CSV file is empty: {actual_path}")
            raise
        except Exception as e:
            logger.error(f"Error loading data from {actual_path}: {str(e)}")
            raise
    
    def load_from_database(self, start_date: datetime, end_date: datetime,
                          db_session) -> pd.DataFrame:
        """Load data from database."""
        from ..core.database import ConsumptionData
        
        query = db_session.query(ConsumptionData).filter(
            ConsumptionData.timestamp.between(start_date, end_date)
        ).order_by(ConsumptionData.timestamp)
        
        return pd.read_sql(query.statement, query.session.bind)
    
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
            
    def save_data(self, data: pd.DataFrame, file_path: str):
        """Save processed data to a CSV file."""
        logger.info(f"Saving data to file: {file_path}")
        try:
            data.to_csv(file_path, index=False)
            logger.info("Data saved successfully")
        except Exception as e:
            logger.error(f"Error saving data to {file_path}: {str(e)}")
            raise 