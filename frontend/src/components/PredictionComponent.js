import React, { useState, useEffect } from 'react';
import { 
  Box, 
  Button, 
  Typography, 
  Alert, 
  Paper, 
  LinearProgress,
  Grid,
  Card,
  CardContent,
  Divider,
  Chip,
  CircularProgress,
  List,
  ListItem,
  ListItemText
} from '@mui/material';
import DatasetIcon from '@mui/icons-material/Dataset';
import TimelineIcon from '@mui/icons-material/Timeline';
import AssessmentIcon from '@mui/icons-material/Assessment';
import axios from 'axios';
import Plot from 'react-plotly.js';

const PredictionComponent = ({ uploadedData, onPrediction, predictions }) => {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [forecastData, setForecastData] = useState(null);
  const [metrics, setMetrics] = useState(null);

  const handleRunForecast = async () => {
    if (!uploadedData) {
      setError('No data uploaded');
      return;
    }

    setLoading(true);
    setError(null);

    try {
      // In a real implementation, this would be an actual API call
      // For now, we'll simulate a successful forecast
      await new Promise(resolve => setTimeout(resolve, 2000));
      
      // Generate mock data for the forecast
      const startDate = new Date(uploadedData.prediction_range.start_date);
      const endDate = new Date(uploadedData.prediction_range.end_date);
      const dayDiff = Math.ceil((endDate - startDate) / (1000 * 60 * 60 * 24)) + 1;
      
      const dates = Array.from({ length: dayDiff }, (_, i) => {
        const date = new Date(startDate);
        date.setDate(date.getDate() + i);
        return date.toISOString().split('T')[0];
      });
      
      // Generate some random forecast values
      const forecastValues = Array.from({ length: dayDiff }, () => 
        Math.round(Math.random() * 1000 + 500)
      );
      
      // Lower and upper bounds for confidence intervals
      const lowerBounds = forecastValues.map(val => Math.round(val * 0.8));
      const upperBounds = forecastValues.map(val => Math.round(val * 1.2));
      
      const result = {
        dates,
        forecast: forecastValues,
        lower_bound: lowerBounds,
        upper_bound: upperBounds,
        model_name: uploadedData.model_type
      };
      
      // Simulate metrics
      const mockMetrics = {
        mape: (Math.random() * 10).toFixed(2),
        rmse: (Math.random() * 100 + 50).toFixed(2),
        mae: (Math.random() * 50 + 20).toFixed(2),
        r2: (Math.random() * 0.3 + 0.7).toFixed(2)
      };
      
      setForecastData(result);
      setMetrics(mockMetrics);
      
      // Pass the prediction data to the parent component
      onPrediction({ forecast: result, metrics: mockMetrics });
    } catch (err) {
      console.error('Forecast error:', err);
      setError(err.response?.data?.detail || 'Failed to generate forecast');
    } finally {
      setLoading(false);
    }
  };

  return (
    <Box sx={{ p: 2 }}>
      <Typography variant="h5" gutterBottom>
        Electricity Consumption Forecast
      </Typography>
      
      {uploadedData && (
        <Paper elevation={1} sx={{ p: 2, mb: 3, backgroundColor: '#f5f5f5' }}>
          <Typography variant="subtitle1" gutterBottom>
            Using {uploadedData.model_type.toUpperCase()} model for prediction
          </Typography>
          <Typography variant="body2">
            Data file: {uploadedData.filename} • 
            Period: {uploadedData.prediction_range.start_date} to {uploadedData.prediction_range.end_date}
          </Typography>
        </Paper>
      )}

      {!forecastData && !loading && (
        <Box sx={{ textAlign: 'center', my: 4 }}>
          <Button 
            variant="contained" 
            color="primary" 
            onClick={handleRunForecast}
            size="large"
            disabled={loading}
            startIcon={<TimelineIcon />}
          >
            Generate Forecast
          </Button>
        </Box>
      )}

      {loading && (
        <Box sx={{ display: 'flex', flexDirection: 'column', alignItems: 'center', my: 4 }}>
          <CircularProgress sx={{ mb: 2 }} />
          <Typography variant="body1">
            Generating forecast...
          </Typography>
        </Box>
      )}

      {error && (
        <Alert severity="error" sx={{ my: 2 }}>
          {error}
        </Alert>
      )}

      {forecastData && (
        <Grid container spacing={3}>
          <Grid item xs={12}>
            <Paper elevation={3} sx={{ p: 2 }}>
              <Typography variant="h6" gutterBottom>
                Forecast Results
              </Typography>
              <Plot
                data={[
                  {
                    x: forecastData.dates,
                    y: forecastData.forecast,
                    type: 'scatter',
                    mode: 'lines',
                    name: 'Forecast',
                    line: { color: 'rgb(31, 119, 180)' }
                  },
                  {
                    x: forecastData.dates,
                    y: forecastData.upper_bound,
                    type: 'scatter',
                    mode: 'lines',
                    name: 'Upper Bound',
                    line: { width: 0 },
                    showlegend: false
                  },
                  {
                    x: forecastData.dates,
                    y: forecastData.lower_bound,
                    type: 'scatter',
                    mode: 'lines',
                    name: 'Lower Bound',
                    line: { width: 0 },
                    fill: 'tonexty',
                    fillcolor: 'rgba(31, 119, 180, 0.2)',
                    showlegend: false
                  }
                ]}
                layout={{
                  title: 'Electricity Consumption Forecast',
                  xaxis: {
                    title: 'Date',
                    tickangle: -45
                  },
                  yaxis: {
                    title: 'Consumption (kWh)'
                  },
                  autosize: true,
                  height: 500,
                  margin: {
                    l: 50,
                    r: 20,
                    t: 70,
                    b: 100
                  }
                }}
                useResizeHandler={true}
                style={{ width: '100%', height: '100%' }}
              />
            </Paper>
          </Grid>

          {metrics && (
            <Grid item xs={12} md={6}>
              <Paper elevation={3} sx={{ p: 2, height: '100%' }}>
                <Typography variant="h6" gutterBottom>
                  Model Performance Metrics
                </Typography>
                <List>
                  <ListItem>
                    <ListItemText 
                      primary="Mean Absolute Percentage Error (MAPE)" 
                      secondary={`${metrics.mape}%`} 
                    />
                  </ListItem>
                  <Divider />
                  <ListItem>
                    <ListItemText 
                      primary="Root Mean Square Error (RMSE)" 
                      secondary={metrics.rmse} 
                    />
                  </ListItem>
                  <Divider />
                  <ListItem>
                    <ListItemText 
                      primary="Mean Absolute Error (MAE)" 
                      secondary={metrics.mae} 
                    />
                  </ListItem>
                  <Divider />
                  <ListItem>
                    <ListItemText 
                      primary="R² Score" 
                      secondary={metrics.r2} 
                    />
                  </ListItem>
                </List>
              </Paper>
            </Grid>
          )}

          <Grid item xs={12} md={6}>
            <Paper elevation={3} sx={{ p: 2, height: '100%' }}>
              <Typography variant="h6" gutterBottom>
                Forecast Summary
              </Typography>
              <Box sx={{ mb: 2 }}>
                <Chip 
                  icon={<DatasetIcon />} 
                  label={`${forecastData.dates.length} days forecast`} 
                  sx={{ mr: 1, mb: 1 }} 
                />
                <Chip 
                  icon={<AssessmentIcon />} 
                  label={`Model: ${forecastData.model_name}`} 
                  sx={{ mr: 1, mb: 1 }} 
                  color="primary"
                />
              </Box>
              <Typography variant="body1" sx={{ mb: 1 }}>
                Forecast Period:
              </Typography>
              <Typography variant="body2" color="textSecondary" sx={{ mb: 2 }}>
                From {forecastData.dates[0]} to {forecastData.dates[forecastData.dates.length - 1]}
              </Typography>
              
              <Typography variant="body1" sx={{ mb: 1 }}>
                Peak Consumption:
              </Typography>
              <Typography variant="body2" color="textSecondary">
                {Math.max(...forecastData.forecast)} kWh on {
                  forecastData.dates[forecastData.forecast.indexOf(Math.max(...forecastData.forecast))]
                }
              </Typography>
            </Paper>
          </Grid>

          <Grid item xs={12}>
            <Box sx={{ display: 'flex', justifyContent: 'flex-end', mt: 2 }}>
              <Button 
                variant="outlined" 
                onClick={handleRunForecast}
                sx={{ mr: 2 }}
              >
                Re-run Forecast
              </Button>
              <Button 
                variant="contained" 
                color="primary"
                onClick={() => {
                  // In a real application, this would download the forecast data
                  alert('Download functionality would be implemented here');
                }}
              >
                Download Results
              </Button>
            </Box>
          </Grid>
        </Grid>
      )}
    </Box>
  );
};

export default PredictionComponent; 