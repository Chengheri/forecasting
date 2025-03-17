import React, { useState } from 'react';
import { 
  Box, 
  Button, 
  Typography, 
  Alert, 
  Paper, 
  LinearProgress,
  Grid,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  TextField
} from '@mui/material';
import UploadFileIcon from '@mui/icons-material/UploadFile';
import axios from 'axios';

const UploadComponent = ({ onDataUploaded }) => {
  const [file, setFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [success, setSuccess] = useState(false);
  const [modelType, setModelType] = useState('prophet');
  const [dateRange, setDateRange] = useState({
    startDate: '',
    endDate: ''
  });

  const handleFileChange = (event) => {
    const selectedFile = event.target.files[0];
    setFile(selectedFile);
    setError(null);
    setSuccess(false);
  };

  const handleModelChange = (event) => {
    setModelType(event.target.value);
  };

  const handleDateChange = (event) => {
    setDateRange({
      ...dateRange,
      [event.target.name]: event.target.value
    });
  };

  const handleUpload = async () => {
    if (!file) {
      setError('Please select a file first');
      return;
    }

    if (!dateRange.startDate || !dateRange.endDate) {
      setError('Please select start and end dates for prediction');
      return;
    }

    const formData = new FormData();
    formData.append('file', file);
    formData.append('model_type', modelType);
    formData.append('start_date', dateRange.startDate);
    formData.append('end_date', dateRange.endDate);

    setLoading(true);
    setError(null);

    try {
      // In a real implementation, this would be an actual API endpoint
      // For now, we'll simulate a successful upload
      await new Promise(resolve => setTimeout(resolve, 1500));
      
      // Simulate data received from backend
      const mockData = {
        filename: file.name,
        model_type: modelType,
        prediction_range: {
          start_date: dateRange.startDate,
          end_date: dateRange.endDate
        },
        data_columns: ['date', 'consumption'],
        data_rows: 1000,
        upload_success: true
      };
      
      setSuccess(true);
      onDataUploaded(mockData);
    } catch (err) {
      console.error('Upload error:', err);
      setError(err.response?.data?.detail || 'Failed to upload file');
    } finally {
      setLoading(false);
    }
  };

  return (
    <Box sx={{ p: 2 }}>
      <Typography variant="h5" gutterBottom>
        Upload Electricity Consumption Data
      </Typography>
      <Typography variant="body1" paragraph>
        Upload your CSV or Excel file containing time series data for electricity consumption.
      </Typography>

      <Grid container spacing={3}>
        <Grid item xs={12} md={6}>
          <Paper 
            elevation={3} 
            sx={{ 
              p: 3, 
              display: 'flex', 
              flexDirection: 'column', 
              alignItems: 'center',
              border: '2px dashed #ccc',
              backgroundColor: '#fafafa',
              height: '100%',
              justifyContent: 'center'
            }}
          >
            <input
              type="file"
              accept=".csv,.xlsx,.xls"
              style={{ display: 'none' }}
              id="upload-file"
              onChange={handleFileChange}
            />
            <label htmlFor="upload-file">
              <Button 
                variant="contained" 
                component="span" 
                startIcon={<UploadFileIcon />}
                sx={{ mb: 2 }}
              >
                Select File
              </Button>
            </label>
            {file ? (
              <Typography variant="body2">
                Selected file: {file.name}
              </Typography>
            ) : (
              <Typography variant="body2" color="textSecondary">
                Drag a file here or click to select
              </Typography>
            )}
          </Paper>
        </Grid>

        <Grid item xs={12} md={6}>
          <Paper elevation={3} sx={{ p: 3, height: '100%' }}>
            <Typography variant="h6" gutterBottom>
              Forecast Settings
            </Typography>

            <FormControl fullWidth sx={{ mb: 2 }}>
              <InputLabel id="model-select-label">Forecasting Model</InputLabel>
              <Select
                labelId="model-select-label"
                id="model-select"
                value={modelType}
                label="Forecasting Model"
                onChange={handleModelChange}
              >
                <MenuItem value="prophet">Prophet</MenuItem>
                <MenuItem value="neuralprophet">Neural Prophet</MenuItem>
                <MenuItem value="sarima">SARIMA</MenuItem>
                <MenuItem value="xgboost">XGBoost</MenuItem>
              </Select>
            </FormControl>

            <TextField
              fullWidth
              label="Prediction Start Date"
              type="date"
              name="startDate"
              value={dateRange.startDate}
              onChange={handleDateChange}
              InputLabelProps={{ shrink: true }}
              sx={{ mb: 2 }}
            />

            <TextField
              fullWidth
              label="Prediction End Date"
              type="date"
              name="endDate"
              value={dateRange.endDate}
              onChange={handleDateChange}
              InputLabelProps={{ shrink: true }}
              sx={{ mb: 2 }}
            />
          </Paper>
        </Grid>
      </Grid>

      {loading && (
        <Box sx={{ mt: 2 }}>
          <LinearProgress />
        </Box>
      )}

      {error && (
        <Alert severity="error" sx={{ mt: 2 }}>
          {error}
        </Alert>
      )}

      {success && (
        <Alert severity="success" sx={{ mt: 2 }}>
          File uploaded successfully!
        </Alert>
      )}

      <Box sx={{ mt: 3, display: 'flex', justifyContent: 'flex-end' }}>
        <Button 
          variant="contained" 
          color="primary" 
          onClick={handleUpload}
          disabled={loading || !file}
        >
          Upload & Continue
        </Button>
      </Box>
    </Box>
  );
};

export default UploadComponent; 