import React, { useState, useEffect } from 'react';
import {
  Container,
  Grid,
  Paper,
  Typography,
  Box,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  CircularProgress,
  Alert,
} from '@mui/material';
import Plot from 'react-plotly.js';
import { compareModels, getMetrics } from '../services/api';

const ModelComparison = () => {
  const [comparisonData, setComparisonData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    fetchComparisonData();
  }, []);

  const fetchComparisonData = async () => {
    try {
      setLoading(true);
      setError(null);
      const response = await compareModels();
      setComparisonData(response);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const renderMetricsTable = () => {
    if (!comparisonData?.metrics) return null;

    return (
      <TableContainer component={Paper}>
        <Table>
          <TableHead>
            <TableRow>
              <TableCell>Model</TableCell>
              <TableCell align="right">RMSE</TableCell>
              <TableCell align="right">MAE</TableCell>
              <TableCell align="right">MAPE</TableCell>
              <TableCell align="right">R²</TableCell>
            </TableRow>
          </TableHead>
          <TableBody>
            {Object.entries(comparisonData.metrics).map(([model, metrics]) => (
              <TableRow key={model}>
                <TableCell component="th" scope="row">
                  {model}
                </TableCell>
                <TableCell align="right">{metrics.rmse.toFixed(2)}</TableCell>
                <TableCell align="right">{metrics.mae.toFixed(2)}</TableCell>
                <TableCell align="right">{metrics.mape.toFixed(2)}%</TableCell>
                <TableCell align="right">{metrics.r2.toFixed(3)}</TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </TableContainer>
    );
  };

  const renderForecastPlot = () => {
    if (!comparisonData?.forecasts) return null;

    const traces = Object.entries(comparisonData.forecasts).map(([model, data]) => ({
      x: data.dates,
      y: data.values,
      type: 'scatter',
      mode: 'lines',
      name: model,
    }));

    return (
      <Plot
        data={traces}
        layout={{
          title: 'Model Forecasts Comparison',
          autosize: true,
          margin: { l: 50, r: 50, t: 50, b: 50 },
          xaxis: { title: 'Date' },
          yaxis: { title: 'Consumption (kWh)' },
          showlegend: true,
          legend: { orientation: 'h', y: -0.2 },
        }}
        useResizeHandler={true}
        style={{ width: '100%', height: '400px' }}
      />
    );
  };

  const renderResidualPlots = () => {
    if (!comparisonData?.residuals) return null;

    const traces = Object.entries(comparisonData.residuals).map(([model, data]) => ({
      x: data.dates,
      y: data.values,
      type: 'scatter',
      mode: 'markers',
      name: model,
    }));

    return (
      <Plot
        data={traces}
        layout={{
          title: 'Residual Analysis',
          autosize: true,
          margin: { l: 50, r: 50, t: 50, b: 50 },
          xaxis: { title: 'Date' },
          yaxis: { title: 'Residual' },
          showlegend: true,
          legend: { orientation: 'h', y: -0.2 },
        }}
        useResizeHandler={true}
        style={{ width: '100%', height: '400px' }}
      />
    );
  };

  if (loading) {
    return (
      <Box display="flex" justifyContent="center" alignItems="center" minHeight="80vh">
        <CircularProgress />
      </Box>
    );
  }

  if (error) {
    return (
      <Container maxWidth="lg" sx={{ mt: 4 }}>
        <Alert severity="error">{error}</Alert>
      </Container>
    );
  }

  return (
    <Container maxWidth="lg" sx={{ mt: 4, mb: 4 }}>
      <Grid container spacing={3}>
        {/* Metrics Table */}
        <Grid item xs={12}>
          <Paper sx={{ p: 2 }}>
            <Typography variant="h6" gutterBottom>
              Model Performance Metrics
            </Typography>
            {renderMetricsTable()}
          </Paper>
        </Grid>

        {/* Forecast Plot */}
        <Grid item xs={12}>
          <Paper sx={{ p: 2 }}>
            <Typography variant="h6" gutterBottom>
              Forecast Comparison
            </Typography>
            {renderForecastPlot()}
          </Paper>
        </Grid>

        {/* Residual Analysis */}
        <Grid item xs={12}>
          <Paper sx={{ p: 2 }}>
            <Typography variant="h6" gutterBottom>
              Residual Analysis
            </Typography>
            {renderResidualPlots()}
          </Paper>
        </Grid>

        {/* Preprocessing Impact */}
        <Grid item xs={12}>
          <Paper sx={{ p: 2 }}>
            <Typography variant="h6" gutterBottom>
              Preprocessing Impact
            </Typography>
            {comparisonData?.preprocessingImpact && (
              <Plot
                data={[
                  {
                    x: comparisonData.preprocessingImpact.methods,
                    y: comparisonData.preprocessingImpact.improvements,
                    type: 'bar',
                  },
                ]}
                layout={{
                  title: 'Impact of Preprocessing Methods',
                  autosize: true,
                  margin: { l: 50, r: 50, t: 50, b: 50 },
                  xaxis: { title: 'Preprocessing Method' },
                  yaxis: { title: 'Improvement (%)' },
                }}
                useResizeHandler={true}
                style={{ width: '100%', height: '300px' }}
              />
            )}
          </Paper>
        </Grid>
      </Grid>
    </Container>
  );
};

export default ModelComparison; 