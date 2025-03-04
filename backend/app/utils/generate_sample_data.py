import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
from ..utils.logger import Logger

logger = Logger()

def generate_weather_data(
    dates: pd.DatetimeIndex,
    base_temp: float = 20.0,
    temp_amplitude: float = 10.0,
    base_humidity: float = 60.0,
    humidity_amplitude: float = 20.0
) -> pd.DataFrame:
    """Generate synthetic weather data with seasonal patterns."""
    n_points = len(dates)
    
    # Generate temperature with seasonal pattern
    # Shift the phase to have summer months (6-8) be warmest
    temp_pattern = np.sin(2 * np.pi * ((dates.month - 6) % 12) / 12)  # Annual cycle
    daily_temp = np.sin(2 * np.pi * np.arange(24) / 24)  # Daily cycle
    daily_temp = np.tile(daily_temp, n_points // 24 + 1)[:n_points]
    
    temperature = base_temp + temp_amplitude * temp_pattern + 5 * daily_temp
    
    # Generate humidity with seasonal pattern (inverse correlation with temperature)
    humidity_pattern = -0.5 * temp_pattern  # Inverse of temperature
    daily_humidity = -0.3 * daily_temp  # Inverse of daily temperature
    
    humidity = base_humidity + humidity_amplitude * humidity_pattern + 10 * daily_humidity
    
    # Add some random noise
    temp_noise = np.random.normal(0, 2, n_points)
    humidity_noise = np.random.normal(0, 5, n_points)
    
    temperature += temp_noise
    humidity += humidity_noise
    
    # Ensure realistic ranges
    temperature = np.clip(temperature, -5, 35)  # Temperature range: -5°C to 35°C
    humidity = np.clip(humidity, 30, 90)  # Humidity range: 30% to 90%
    
    return pd.DataFrame({
        'temperature': temperature,
        'humidity': humidity
    })

def generate_consumption_data(
    start_date: datetime,
    end_date: datetime,
    freq: str = 'h',
    base_value: float = 100.0,
    noise_level: float = 0.1,
    seasonality: bool = True,
    trend: bool = True,
    holidays: bool = True,
    weather_effects: bool = True,
    base_temp: float = 20.0,
    temp_amplitude: float = 10.0,
    base_humidity: float = 60.0,
    humidity_amplitude: float = 20.0
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
            
            # Weekly seasonality (using the day of week from the date)
            weekly_pattern = np.sin(2 * np.pi * data['date'].dt.dayofweek / 7)
            
            # Monthly seasonality (using the month number from the date)
            monthly_pattern = np.sin(2 * np.pi * (data['date'].dt.month - 1) / 12)
            
            # Combine patterns
            data['value'] *= (1 + 0.3 * daily_pattern + 0.2 * weekly_pattern + 0.1 * monthly_pattern)
        
        # Add weather effects
        if weather_effects:
            logger.debug("Adding weather effects")
            weather_data = generate_weather_data(
                dates,
                base_temp=base_temp,
                temp_amplitude=temp_amplitude,
                base_humidity=base_humidity,
                humidity_amplitude=humidity_amplitude
            )
            
            # Temperature effects (heating and cooling)
            temp_effect = np.zeros(n_points)
            temp_effect[weather_data['temperature'] < 15] = 0.2  # Heating effect
            temp_effect[weather_data['temperature'] > 25] = 0.3  # Cooling effect
            
            # Humidity effects (increased consumption in high humidity)
            humidity_effect = np.zeros(n_points)
            humidity_effect[weather_data['humidity'] > 70] = 0.1  # High humidity effect
            
            # Combine weather effects
            weather_factor = 1 + temp_effect + humidity_effect
            data['value'] *= weather_factor
            
            # Add weather data to the output
            data['temperature'] = weather_data['temperature']
            data['humidity'] = weather_data['humidity']
        
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
                'weather_effects': True,
                'base_temp': 20.0,
                'temp_amplitude': 10.0,
                'base_humidity': 60.0,
                'humidity_amplitude': 20.0,
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
            holidays=config['holidays'],
            weather_effects=config.get('weather_effects', True),
            base_temp=config.get('base_temp', 20.0),
            temp_amplitude=config.get('temp_amplitude', 10.0),
            base_humidity=config.get('base_humidity', 60.0),
            humidity_amplitude=config.get('humidity_amplitude', 20.0)
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