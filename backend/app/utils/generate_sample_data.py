import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
from ..utils.logger import Logger

logger = Logger()

def generate_consumption_data(
    start_date: datetime,
    end_date: datetime,
    freq: str = 'H',
    base_value: float = 100.0,
    noise_level: float = 0.1,
    seasonality: bool = True,
    trend: bool = True,
    holidays: bool = True
) -> pd.DataFrame:
    """Generate sample electricity consumption data."""
    logger.info(f"Generating sample consumption data from {start_date} to {end_date}")
    try:
        # Generate date range
        dates = pd.date_range(start=start_date, end=end_date, freq=freq)
        n_points = len(dates)
        
        # Initialize data
        data = pd.DataFrame({'date': dates})
        
        # Add base consumption
        data['value'] = base_value
        
        # Add trend
        if trend:
            logger.debug("Adding trend to the data")
            trend_factor = np.linspace(0, 0.2, n_points)  # 20% increase over the period
            data['value'] *= (1 + trend_factor)
        
        # Add seasonality
        if seasonality:
            logger.debug("Adding seasonality patterns")
            # Daily seasonality
            daily_pattern = np.sin(2 * np.pi * np.arange(24) / 24)
            daily_pattern = np.tile(daily_pattern, n_points // 24 + 1)[:n_points]
            
            # Weekly seasonality
            weekly_pattern = np.sin(2 * np.pi * np.arange(7) / 7)
            weekly_pattern = np.tile(weekly_pattern, n_points // (24 * 7) + 1)[:n_points]
            
            # Monthly seasonality
            monthly_pattern = np.sin(2 * np.pi * np.arange(12) / 12)
            monthly_pattern = np.tile(monthly_pattern, n_points // (24 * 30) + 1)[:n_points]
            
            # Combine patterns
            data['value'] *= (1 + 0.3 * daily_pattern + 0.2 * weekly_pattern + 0.1 * monthly_pattern)
        
        # Add holidays
        if holidays:
            logger.debug("Adding holiday effects")
            # Generate random holidays
            n_holidays = int(n_points / (24 * 30))  # Approximately one holiday per month
            holiday_dates = np.random.choice(n_points, n_holidays, replace=False)
            data.loc[holiday_dates, 'value'] *= 0.7  # 30% reduction on holidays
        
        # Add noise
        logger.debug("Adding random noise")
        noise = np.random.normal(0, noise_level, n_points)
        data['value'] *= (1 + noise)
        
        # Ensure positive values
        data['value'] = np.maximum(data['value'], 0)
        
        logger.info(f"Generated {len(data)} data points")
        return data
    except Exception as e:
        logger.error(f"Error generating consumption data: {str(e)}")
        raise

def add_anomalies(
    data: pd.DataFrame,
    anomaly_type: str = 'spike',
    n_anomalies: int = 10,
    magnitude: float = 2.0
) -> pd.DataFrame:
    """Add anomalies to the data."""
    logger.info(f"Adding {n_anomalies} {anomaly_type} anomalies")
    try:
        data = data.copy()
        n_points = len(data)
        
        # Generate random anomaly positions
        anomaly_positions = np.random.choice(n_points, n_anomalies, replace=False)
        
        if anomaly_type == 'spike':
            logger.debug("Adding spike anomalies")
            data.loc[anomaly_positions, 'value'] *= magnitude
        elif anomaly_type == 'dip':
            logger.debug("Adding dip anomalies")
            data.loc[anomaly_positions, 'value'] /= magnitude
        elif anomaly_type == 'noise':
            logger.debug("Adding noise anomalies")
            noise = np.random.normal(0, magnitude, n_anomalies)
            data.loc[anomaly_positions, 'value'] *= (1 + noise)
        else:
            logger.error(f"Unsupported anomaly type: {anomaly_type}")
            raise ValueError(f"Unsupported anomaly type: {anomaly_type}")
        
        logger.info("Anomalies added successfully")
        return data
    except Exception as e:
        logger.error(f"Error adding anomalies: {str(e)}")
        raise

def generate_sample_dataset(
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    config: Optional[Dict[str, Any]] = None
) -> pd.DataFrame:
    """Generate a complete sample dataset with configurable parameters."""
    logger.info("Generating sample dataset")
    try:
        # Set default dates if not provided
        if start_date is None:
            start_date = datetime.now() - timedelta(days=365)
        if end_date is None:
            end_date = datetime.now()
            
        # Set default configuration
        if config is None:
            config = {
                'base_value': 100.0,
                'noise_level': 0.1,
                'seasonality': True,
                'trend': True,
                'holidays': True,
                'anomalies': {
                    'type': 'spike',
                    'count': 10,
                    'magnitude': 2.0
                }
            }
        
        # Generate base data
        data = generate_consumption_data(
            start_date=start_date,
            end_date=end_date,
            base_value=config['base_value'],
            noise_level=config['noise_level'],
            seasonality=config['seasonality'],
            trend=config['trend'],
            holidays=config['holidays']
        )
        
        # Add anomalies if configured
        if 'anomalies' in config:
            data = add_anomalies(
                data,
                anomaly_type=config['anomalies']['type'],
                n_anomalies=config['anomalies']['count'],
                magnitude=config['anomalies']['magnitude']
            )
        
        logger.info("Sample dataset generated successfully")
        return data
    except Exception as e:
        logger.error(f"Error generating sample dataset: {str(e)}")
        raise

if __name__ == "__main__":
    # Generate one year of hourly data
    df = generate_consumption_data(datetime.now() - timedelta(days=365), datetime.now())
    
    # Add some anomalies
    df = add_anomalies(df)
    
    # Save to CSV
    df.to_csv("sample_consumption_data.csv", index=False) 