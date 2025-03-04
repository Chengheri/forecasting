# Electricity Consumption Forecasting

A Python-based project for forecasting electricity consumption using various time series models.

## Features

- Multiple forecasting models (ARIMA, SARIMA, Prophet, NeuralProphet, LSTM, etc.)
- Advanced data preprocessing and anomaly detection
- Comprehensive logging system
- Database integration for data storage
- Model comparison and evaluation tools
- Modern React frontend for visualization and interaction

## Project Structure

```
forecasting/
├── backend/
│   └── app/
│       ├── core/           # Core functionality
│       ├── models/         # Forecasting models
│       └── utils/          # Utility functions
├── frontend/
│   ├── src/
│   │   ├── components/     # React components
│   │   ├── pages/         # Page components
│   │   └── services/      # API services
│   └── public/            # Static files
├── data/
│   └── logs/              # Application logs
└── examples/              # Example scripts
```

## Setup

### Backend Setup

1. Clone the repository
2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Set up environment variables:
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

### Frontend Setup

1. Navigate to the frontend directory:
   ```bash
   cd frontend
   ```
2. Install Node.js dependencies:
   ```bash
   npm install
   ```
3. Create a `.env` file in the frontend directory:
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

## Running the Application

### Start the Backend Server

1. Activate the virtual environment (if not already activated):
   ```bash
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
2. Start the backend server:
   ```bash
   cd backend
   uvicorn main:app --reload
   ```
   The backend will be available at http://localhost:8000

### Start the Frontend Development Server

1. In a new terminal, navigate to the frontend directory:
   ```bash
   cd frontend
   ```
2. Start the development server:
   ```bash
   npm start
   ```
   The frontend will be available at http://localhost:3000

## Usage

See the `examples/` directory for usage examples.

## Logging

Logs are stored in `data/logs/` with the format `forecasting_YYYY-MM-DD.log`. Each day gets its own log file, and logs are kept indefinitely. 