import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os
import sys

# Add the project root directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from backend.app.utils.logger import Logger

# Initialize logger
logger = Logger()

def load_and_preprocess_data(filename: str) -> pd.DataFrame:
    """Load and preprocess data from a CSV file."""
    logger.info(f"Loading and preprocessing data from {filename}")
    try:
        df = pd.read_csv(filename)
        df['date'] = pd.to_datetime(df['date'])
        df['hour'] = df['date'].dt.hour
        df['day'] = df['date'].dt.day
        df['month'] = df['date'].dt.month
        df['year'] = df['date'].dt.year
        df['dayofweek'] = df['date'].dt.dayofweek
        df['season'] = pd.cut(df['date'].dt.month, 
                             bins=[0, 3, 6, 9, 12],
                             labels=['Winter', 'Spring', 'Summer', 'Fall'])
        logger.info(f"Data loaded successfully. Shape: {df.shape}")
        return df
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        raise

def plot_time_series(df: pd.DataFrame, output_dir: str):
    """Plot time series data."""
    logger.info("Generating time series plots")
    try:
        plt.figure(figsize=(15, 10))
        
        # Plot consumption over time
        plt.subplot(3, 1, 1)
        plt.plot(df['date'], df['value'])
        plt.title('Electricity Consumption Over Time')
        plt.xlabel('Date')
        plt.ylabel('Consumption')
        
        # Plot temperature over time
        plt.subplot(3, 1, 2)
        plt.plot(df['date'], df['temperature'])
        plt.title('Temperature Over Time')
        plt.xlabel('Date')
        plt.ylabel('Temperature (°C)')
        
        # Plot humidity over time
        plt.subplot(3, 1, 3)
        plt.plot(df['date'], df['humidity'])
        plt.title('Humidity Over Time')
        plt.xlabel('Date')
        plt.ylabel('Humidity (%)')
        
        plt.tight_layout()
        output_path = os.path.join(output_dir, 'time_series_plots.png')
        plt.savefig(output_path)
        plt.close()
        logger.info(f"Time series plots saved to {output_path}")
    except Exception as e:
        logger.error(f"Error generating time series plots: {str(e)}")
        raise

def plot_seasonal_patterns(df: pd.DataFrame, output_dir: str):
    """Plot seasonal patterns."""
    logger.info("Generating seasonal pattern plots")
    try:
        plt.figure(figsize=(15, 10))
        
        # Monthly averages
        monthly_avg = df.groupby('month').agg({
            'value': 'mean',
            'temperature': 'mean',
            'humidity': 'mean'
        })
        
        # Plot monthly patterns
        plt.subplot(3, 1, 1)
        monthly_avg['value'].plot(kind='bar')
        plt.title('Average Consumption by Month')
        plt.xlabel('Month')
        plt.ylabel('Consumption')
        
        plt.subplot(3, 1, 2)
        monthly_avg['temperature'].plot(kind='bar')
        plt.title('Average Temperature by Month')
        plt.xlabel('Month')
        plt.ylabel('Temperature (°C)')
        
        plt.subplot(3, 1, 3)
        monthly_avg['humidity'].plot(kind='bar')
        plt.title('Average Humidity by Month')
        plt.xlabel('Month')
        plt.ylabel('Humidity (%)')
        
        plt.tight_layout()
        output_path = os.path.join(output_dir, 'seasonal_patterns.png')
        plt.savefig(output_path)
        plt.close()
        logger.info(f"Seasonal pattern plots saved to {output_path}")
    except Exception as e:
        logger.error(f"Error generating seasonal pattern plots: {str(e)}")
        raise

def plot_daily_patterns(df: pd.DataFrame, output_dir: str):
    """Plot daily patterns."""
    logger.info("Generating daily pattern plots")
    try:
        plt.figure(figsize=(15, 10))
        
        # Hourly averages
        hourly_avg = df.groupby('hour').agg({
            'value': 'mean',
            'temperature': 'mean',
            'humidity': 'mean'
        })
        
        # Plot hourly patterns
        plt.subplot(3, 1, 1)
        hourly_avg['value'].plot()
        plt.title('Average Consumption by Hour')
        plt.xlabel('Hour')
        plt.ylabel('Consumption')
        
        plt.subplot(3, 1, 2)
        hourly_avg['temperature'].plot()
        plt.title('Average Temperature by Hour')
        plt.xlabel('Hour')
        plt.ylabel('Temperature (°C)')
        
        plt.subplot(3, 1, 3)
        hourly_avg['humidity'].plot()
        plt.title('Average Humidity by Hour')
        plt.xlabel('Hour')
        plt.ylabel('Humidity (%)')
        
        plt.tight_layout()
        output_path = os.path.join(output_dir, 'daily_patterns.png')
        plt.savefig(output_path)
        plt.close()
        logger.info(f"Daily pattern plots saved to {output_path}")
    except Exception as e:
        logger.error(f"Error generating daily pattern plots: {str(e)}")
        raise

def plot_correlations(df: pd.DataFrame, output_dir: str):
    """Plot correlation heatmap and scatter plots."""
    logger.info("Generating correlation plots")
    try:
        # Correlation heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(df[['value', 'temperature', 'humidity']].corr(), 
                    annot=True, cmap='coolwarm', center=0)
        plt.title('Correlation Heatmap')
        output_path = os.path.join(output_dir, 'correlation_heatmap.png')
        plt.savefig(output_path)
        plt.close()
        
        # Scatter plots
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 2, 1)
        plt.scatter(df['temperature'], df['value'], alpha=0.5)
        plt.title('Consumption vs Temperature')
        plt.xlabel('Temperature (°C)')
        plt.ylabel('Consumption')
        
        plt.subplot(1, 2, 2)
        plt.scatter(df['humidity'], df['value'], alpha=0.5)
        plt.title('Consumption vs Humidity')
        plt.xlabel('Humidity (%)')
        plt.ylabel('Consumption')
        
        plt.tight_layout()
        output_path = os.path.join(output_dir, 'scatter_plots.png')
        plt.savefig(output_path)
        plt.close()
        logger.info("Correlation plots saved successfully")
    except Exception as e:
        logger.error(f"Error generating correlation plots: {str(e)}")
        raise

def plot_distributions(df: pd.DataFrame, output_dir: str):
    """Plot distribution plots."""
    logger.info("Generating distribution plots")
    try:
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 3, 1)
        sns.histplot(df['value'], kde=True)
        plt.title('Consumption Distribution')
        plt.xlabel('Consumption')
        
        plt.subplot(1, 3, 2)
        sns.histplot(df['temperature'], kde=True)
        plt.title('Temperature Distribution')
        plt.xlabel('Temperature (°C)')
        
        plt.subplot(1, 3, 3)
        sns.histplot(df['humidity'], kde=True)
        plt.title('Humidity Distribution')
        plt.xlabel('Humidity (%)')
        
        plt.tight_layout()
        output_path = os.path.join(output_dir, 'distributions.png')
        plt.savefig(output_path)
        plt.close()
        logger.info(f"Distribution plots saved to {output_path}")
    except Exception as e:
        logger.error(f"Error generating distribution plots: {str(e)}")
        raise

def generate_summary_statistics(df: pd.DataFrame, output_dir: str):
    """Generate summary statistics."""
    logger.info("Generating summary statistics")
    try:
        # Basic statistics
        basic_stats = df[['value', 'temperature', 'humidity']].describe()
        
        # Additional statistics by season
        seasonal_stats = df.groupby('season', observed=True).agg({
            'value': ['mean', 'std', 'min', 'max'],
            'temperature': ['mean', 'std', 'min', 'max'],
            'humidity': ['mean', 'std', 'min', 'max']
        })
        
        # Save statistics to CSV
        basic_stats_path = os.path.join(output_dir, 'basic_statistics.csv')
        seasonal_stats_path = os.path.join(output_dir, 'seasonal_statistics.csv')
        
        basic_stats.to_csv(basic_stats_path)
        seasonal_stats.to_csv(seasonal_stats_path)
        
        logger.info(f"Statistics saved to {output_dir}")
        return basic_stats, seasonal_stats
    except Exception as e:
        logger.error(f"Error generating summary statistics: {str(e)}")
        raise

def main():
    """Main function to run the analysis."""
    logger.info("Starting data analysis")
    
    try:
        # Set up directories
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        analysis_dir = os.path.join(project_root, 'data', 'analysis')
        logs_dir = os.path.join(project_root, 'data', 'logs')
        
        # Create directories if they don't exist
        os.makedirs(analysis_dir, exist_ok=True)
        os.makedirs(logs_dir, exist_ok=True)
        
        # Load data from data/examples directory
        data_path = os.path.join(project_root, 'data', 'examples', 'consumption_data_france.csv')
        df = load_and_preprocess_data(data_path)
        
        # Generate and save plots
        plot_time_series(df, analysis_dir)
        plot_seasonal_patterns(df, analysis_dir)
        plot_daily_patterns(df, analysis_dir)
        plot_correlations(df, analysis_dir)
        plot_distributions(df, analysis_dir)
        
        # Generate summary statistics
        basic_stats, seasonal_stats = generate_summary_statistics(df, analysis_dir)
        
        # Print summary
        print("\nBasic Statistics:")
        print(basic_stats)
        print("\nSeasonal Statistics:")
        print(seasonal_stats)
        
        logger.info(f"Analysis complete. Results saved in {analysis_dir}")
        print(f"\nAnalysis complete. Results saved in {analysis_dir}")
        print(f"Logs saved in {logs_dir}")
        
    except Exception as e:
        logger.error(f"Error in main analysis: {str(e)}")
        raise

if __name__ == "__main__":
    main() 