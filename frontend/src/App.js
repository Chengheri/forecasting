import React, { useState } from 'react';
import { 
  Box, 
  Container, 
  Paper, 
  Typography, 
  Tabs, 
  Tab,
  AppBar,
  Toolbar,
  CssBaseline,
  ThemeProvider,
  createTheme
} from '@mui/material';
import UploadComponent from './components/UploadComponent';
import PredictionComponent from './components/PredictionComponent';

// Create a theme with blue and green colors (representing electricity and forecasting)
const theme = createTheme({
  palette: {
    primary: {
      main: '#1976d2',
    },
    secondary: {
      main: '#2e7d32',
    },
  },
});

function App() {
  const [value, setValue] = useState(0);
  const [uploadedData, setUploadedData] = useState(null);
  const [predictions, setPredictions] = useState(null);

  const handleChange = (event, newValue) => {
    setValue(newValue);
  };

  const handleDataUploaded = (data) => {
    setUploadedData(data);
    // Automatically move to prediction tab after successful upload
    setValue(1);
  };

  const handlePrediction = (predictionData) => {
    setPredictions(predictionData);
  };

  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <Box sx={{ display: 'flex', flexDirection: 'column', minHeight: '100vh' }}>
        <AppBar position="static">
          <Toolbar>
            <Typography variant="h6" component="div" sx={{ flexGrow: 1 }}>
              Electricity Consumption Forecasting
            </Typography>
          </Toolbar>
        </AppBar>

        <Container maxWidth="lg" sx={{ mt: 4, mb: 4, flexGrow: 1 }}>
          <Paper sx={{ p: 2 }}>
            <Box sx={{ borderBottom: 1, borderColor: 'divider', mb: 2 }}>
              <Tabs value={value} onChange={handleChange} aria-label="forecasting tabs">
                <Tab label="Upload Data" />
                <Tab label="Prediction" disabled={!uploadedData} />
              </Tabs>
            </Box>
            
            {value === 0 && (
              <UploadComponent onDataUploaded={handleDataUploaded} />
            )}
            
            {value === 1 && (
              <PredictionComponent 
                uploadedData={uploadedData} 
                onPrediction={handlePrediction}
                predictions={predictions}
              />
            )}
          </Paper>
        </Container>

        <Box component="footer" sx={{ py: 3, px: 2, mt: 'auto', backgroundColor: (theme) => theme.palette.grey[200] }}>
          <Container maxWidth="sm">
            <Typography variant="body2" color="text.secondary" align="center">
              Electricity Consumption Forecasting © {new Date().getFullYear()}
            </Typography>
          </Container>
        </Box>
      </Box>
    </ThemeProvider>
  );
}

export default App; 