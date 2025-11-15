# Concentrated Position Analyzer - Streamlit App

A user-friendly web application for analyzing concentrated stock positions, perfect for presenting to non-technical stakeholders.

## Features

- **Clean Web Interface**: No coding required - just fill in the form and click analyze
- **Interactive Inputs**: Easy-to-use sidebar with sliders and text inputs
- **Visual Metrics**: Key metrics displayed prominently with color-coded cards
- **Interactive Charts**: Historical price comparison and proceeds breakdown
- **Export Functionality**: Download analysis report as text file

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

## Running the App

```bash
streamlit run streamlit_app.py
```

The app will open in your default web browser at `http://localhost:8501`

## Usage

1. **Enter Stock Ticker**: Type the stock symbol (e.g., TSLA, AAPL)
2. **Set Position Details**: 
   - Number of shares to sell
   - Average cost basis per share
   - Tax rate (use slider)
3. **Click "Analyze Position"**: The app will fetch data and calculate all metrics
4. **Review Results**: 
   - Risk metrics (Volatility, Beta, Max Drawdown)
   - Tax simulation results
   - Historical price chart
5. **Export Report**: Download the analysis as a text file

## What It Shows

- **Risk Metrics**:
  - Annualized Volatility
  - Beta vs. SPY
  - Maximum Drawdown with dates

- **Liquidity & Tax Simulation**:
  - Estimated proceeds from sale
  - Tax estimate based on your rate
  - After-tax proceeds
  - Visual breakdown chart

- **Visualization**:
  - Normalized price comparison (stock vs. SPY)
  - Easy-to-read line chart

## Perfect For

- Client presentations
- Executive briefings
- Non-technical stakeholders
- Quick position analysis
- Tax planning discussions

