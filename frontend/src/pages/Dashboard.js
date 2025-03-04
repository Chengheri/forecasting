import React, { useState, useEffect } from 'react';
import { Container, Grid, Paper, Typography, Box } from '@mui/material';
import Plot from 'react-plotly.js';
import ForecastForm from '../components/ForecastForm';
import MetricsCard from '../components/MetricsCard';
import { getForecast } from '../services/api';

const Dashboard = () => {
  const [forecastData, setForecastData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleForecastSubmit = async (formData) => {
    try {
      setLoading(true);
      setError(null);
      const response = await getForecast(formData);
      setForecastData(response);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <Container maxWidth="lg" sx={{ mt: 4, mb: 4 }}>
      <Grid container spacing={3}>
        {/* Forecast Form */}
        <Grid item xs={12} md={4}>
          <Paper sx={{ p: 2 }}>
            <Typography variant="h6" gutterBottom>
              Forecast Parameters
            </Typography>
            <ForecastForm onSubmit={handleForecastSubmit} loading={loading} />
          </Paper>
        </Grid>

        {/* Main Chart */}
        <Grid item xs={12} md={8}>
          <Paper sx={{ p: 2 }}>
            <Typography variant="h6" gutterBottom>
              Consumption Forecast
            </Typography>
            {forecastData && (
              <Plot
                data={[
                  {
                    x: forecastData.dates,
                    y: forecastData.forecast,
                    type: 'scatter',
                    mode: 'lines+markers',
                    name: 'Forecast',
                  },
                ]}
                layout={{
                  autosize: true,
                  margin: { l: 50, r: 50, t: 30, b: 30 },
                  xaxis: { title: 'Date' },
                  yaxis: { title: 'Consumption (kWh)' },
                }}
                useResizeHandler={true}
                style={{ width: '100%', height: '400px' }}
              />
            )}
          </Paper>
        </Grid>

        {/* Metrics Cards */}
        {forecastData && (
          <>
            <Grid item xs={12} md={4}>
              <MetricsCard
                title="MAPE"
                value={forecastData.metrics.mape}
                unit="%"
                description="Mean Absolute Percentage Error"
              />
            </Grid>
            <Grid item xs={12} md={4}>
              <MetricsCard
                title="RMSE"
                value={forecastData.metrics.rmse}
                unit="kWh"
                description="Root Mean Square Error"
              />
            </Grid>
            <Grid item xs={12} md={4}>
              <MetricsCard
                title="MAE"
                value={forecastData.metrics.mae}
                unit="kWh"
                description="Mean Absolute Error"
              />
            </Grid>
          </>
        )}

        {/* Error Display */}
        {error && (
          <Grid item xs={12}>
            <Paper sx={{ p: 2, bgcolor: 'error.light' }}>
              <Typography color="error">{error}</Typography>
            </Paper>
          </Grid>
        )}
      </Grid>
    </Container>
  );
};

export default Dashboard; 