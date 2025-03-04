import pandas as pd
from datetime import datetime, timedelta
import sys
import os

# Add the project root directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from backend.app.utils.generate_sample_data import generate_consumption_data, add_anomalies

def main():
    # Generate base consumption data
    start_date = datetime(2023, 1, 1)
    end_date = datetime(2023, 12, 31)
    
    # Configuration based on French weather patterns
    config = {
        'name': 'france',
        'base_value': 1000,  # Base consumption value
        'noise_level': 0.1,  # Random variation
        'seasonality': True,  # Strong seasonal patterns in France
        'trend': True,       # General trend in consumption
        'holidays': True,    # French holidays affect consumption
        'weather_effects': True,
        # Temperature settings based on French climate
        'base_temp': 11.5,   # Average annual temperature in France
        'temp_amplitude': 12.0,  # Temperature variation throughout the year (larger to match seasonal differences)
        # Humidity settings based on French climate
        'base_humidity': 77.0,  # Average humidity in France (higher in winter)
        'humidity_amplitude': 10.0  # Humidity variation
    }
    
    print(f"\nGenerating {config['name']} dataset with French weather patterns...")
    
    # Generate base data
    df = generate_consumption_data(
        start_date=start_date,
        end_date=end_date,
        freq='h',  # Using 'h' instead of 'H' to avoid deprecation warning
        base_value=config['base_value'],
        noise_level=config['noise_level'],
        seasonality=config['seasonality'],
        trend=config['trend'],
        holidays=config['holidays'],
        weather_effects=config['weather_effects'],
        base_temp=config['base_temp'],
        temp_amplitude=config['temp_amplitude'],
        base_humidity=config['base_humidity'],
        humidity_amplitude=config['humidity_amplitude']
    )
    
    # Add anomalies (e.g., extreme weather events, holidays)
    df = add_anomalies(
        df,
        anomaly_type='spike',
        n_anomalies=10,
        magnitude=2.0
    )
    
    # Save to CSV in data/examples directory
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    output_file = os.path.join(project_root, 'data', 'examples', f'consumption_data_{config["name"]}.csv')
    df.to_csv(output_file, index=True)
    print(f"Generated and saved {output_file}")
    print(f"Dataset shape: {df.shape}")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"Value range: {df['value'].min():.2f} to {df['value'].max():.2f}")
    print(f"Temperature range: {df['temperature'].min():.1f}°C to {df['temperature'].max():.1f}°C")
    print(f"Humidity range: {df['humidity'].min():.1f}% to {df['humidity'].max():.1f}%")
    
    # Print seasonal averages
    df['season'] = pd.cut(df['date'].dt.month, 
                         bins=[0, 3, 6, 9, 12],
                         labels=['Winter', 'Spring', 'Summer', 'Fall'])
    
    seasonal_stats = df.groupby('season', observed=True).agg({
        'temperature': 'mean',
        'humidity': 'mean',
        'value': 'mean'
    }).round(2)
    
    print("\nSeasonal Averages:")
    print(seasonal_stats)
    
    # Print monthly averages
    monthly_stats = df.groupby(df['date'].dt.month).agg({
        'temperature': 'mean',
        'humidity': 'mean',
        'value': 'mean'
    }).round(2)
    
    print("\nMonthly Averages:")
    print(monthly_stats)

if __name__ == "__main__":
    main() 