import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { ThemeProvider, createTheme } from '@mui/material/styles';
import CssBaseline from '@mui/material/CssBaseline';
import Dashboard from './pages/Dashboard';
import Navigation from './components/Navigation';
import ModelComparison from './pages/ModelComparison';
import DataUpload from './pages/DataUpload';

const theme = createTheme({
  palette: {
    mode: 'light',
    primary: {
      main: '#1976d2',
    },
    secondary: {
      main: '#dc004e',
    },
  },
});

function App() {
  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <Router>
        <Navigation />
        <Routes>
          <Route path="/" element={<Dashboard />} />
          <Route path="/compare" element={<ModelComparison />} />
          <Route path="/upload" element={<DataUpload />} />
        </Routes>
      </Router>
    </ThemeProvider>
  );
}

export default App; 