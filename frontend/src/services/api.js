import axios from 'axios';

const API_BASE_URL = process.env.REACT_APP_API_BASE_URL || 'http://localhost:8000';

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

export const getForecast = async (modelId, params) => {
  try {
    const response = await axios.get(`${API_BASE_URL}/api/models/${modelId}/forecast`, { params });
    return response.data;
  } catch (error) {
    throw new Error('Failed to get forecast');
  }
};

export const getModels = async () => {
  const response = await api.get('/models');
  return response.data.models;
};

export const uploadData = async (formData) => {
  const response = await api.post('/upload', formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
  });
  return response.data;
};

export const getMetrics = async (modelId) => {
  try {
    const response = await axios.get(`${API_BASE_URL}/api/models/${modelId}/metrics`);
    return response.data;
  } catch (error) {
    throw new Error('Failed to fetch model metrics');
  }
};

export const compareModels = async () => {
  try {
    const response = await axios.get(`${API_BASE_URL}/api/models/compare`);
    return response.data;
  } catch (error) {
    throw new Error('Failed to fetch model comparison data');
  }
};

export const trainModel = async (modelId, data) => {
  try {
    const response = await axios.post(`${API_BASE_URL}/api/models/${modelId}/train`, data);
    return response.data;
  } catch (error) {
    throw new Error('Failed to train model');
  }
};

export const getPreprocessingImpact = async () => {
  try {
    const response = await axios.get(`${API_BASE_URL}/api/preprocessing/impact`);
    return response.data;
  } catch (error) {
    throw new Error('Failed to fetch preprocessing impact data');
  }
};

export default api; 