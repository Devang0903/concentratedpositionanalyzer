# Concentrated Position Risk & Liquidity Analyzer

A Python-based tool that analyzes a single, concentrated stock or ETF position and provides a foundational risk and liquidity assessment. This tool is designed for analyzing concentrated positions held by entrepreneurs and executives.

## 🚀 Live Demo

[Deploy on Streamlit Cloud](https://share.streamlit.io) - Free hosting for Streamlit apps!

Repository: [https://github.com/Devang0903/concentratedpositionanalyzer](https://github.com/Devang0903/concentratedpositionanalyzer)

## Features

- **Risk Analytics Engine**
  - Volatility: Annualized standard deviation of daily returns
  - Beta vs. Benchmark: The stock's sensitivity to the market (SPY)
  - Maximum Drawdown: The largest peak-to-trough decline over the analysis period
  - **Rolling Risk Metrics**: Time-series plots showing how volatility and beta evolved over time (60-day rolling window)

- **Liquidity & Tax Simulation**
  - Calculate estimated proceeds from selling shares
  - Tax estimation based on user-provided cost basis and tax rate
  - After-tax proceeds calculation
  - **Liquidity Metrics**:
    - Average Daily Volume (ADV) for 30-day and 90-day periods
    - Dollar ADV (ADV × current price)
    - Days to Exit Position (shares ÷ ADV)
    - Position size as percentage of ADV (flags if >10-20%, indicating illiquidity)
  - **Scenario Analysis**: Compare selling 10%, 25%, 50%, and 100% of position with detailed breakdowns

- **Data Management**
  - Automatic data fetching from Yahoo Finance (via yfinance)
  - Intelligent caching system (data cached for 1 day to minimize API calls)
  - 5 years of historical data analysis
  - Ticker validation using Yahoo Finance API to verify ticker exists before analysis
  - **Rate limiting protection**: Automatic retry with exponential backoff to prevent "Too Many Requests" errors

- **Concentration Risk Flags**
  - Automatic warnings for:
    - High Beta (>1.5): Stock highly sensitive to market movements
    - High Volatility (>40%): Significant price swings
    - Large Drawdown (>50%): Stock experienced significant decline
    - Low Liquidity: Position represents >10% of ADV

- **Visualization**
  - Historical price comparison chart (stock vs. benchmark)
  - Normalized price comparison for easy visual analysis
  - **Rolling Risk Metrics Chart**: Time-series visualization of rolling volatility and beta

## Installation

1. Clone or download this repository

2. Create and activate a virtual environment (recommended):
```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
# venv\Scripts\activate
```

3. Install required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Make sure your virtual environment is activated, then run the analyzer script:

```bash
python analyzer.py
```

The script will prompt you for:
1. **Stock ticker** (e.g., TSLA, AAPL, etc.)
2. **Number of shares** to hypothetically sell
3. **Average cost basis** per share (in dollars)
4. **Tax rate** as a decimal (e.g., 0.20 for 20%)

### Example Session

```
======================================================================
CONCENTRATED POSITION RISK & LIQUIDITY ANALYZER
======================================================================

Enter stock ticker (e.g., TSLA): TSLA
Enter number of shares to hypothetically sell: 1000
Enter average cost basis per share ($): 150.00
Enter tax rate as decimal (e.g., 0.20 for 20%): 0.20

Analyzing TSLA...
----------------------------------------------------------------------
Fetching TSLA data from Yahoo Finance...
✓ Fetched and cached TSLA data
Fetching SPY data from Yahoo Finance...
✓ Fetched and cached SPY data

Calculating risk metrics...
Calculating tax simulation...

======================================================================
CONCENTRATED POSITION ANALYSIS: TSLA
======================================================================

📊 CURRENT PRICE: $245.50

----------------------------------------------------------------------
RISK METRICS
----------------------------------------------------------------------
  Volatility (Annualized):     45.23%
  Beta vs. SPY:                2.15
  Maximum Drawdown:            68.42%
  Max Drawdown Period:         2021-11-04 to 2022-12-27
  Max Drawdown Value:          $168.20

----------------------------------------------------------------------
LIQUIDITY & TAX SIMULATION
----------------------------------------------------------------------
  Shares to Sell:              1,000
  Estimated Proceeds:          $245,500.00
  Total Cost Basis:           $150,000.00
  Estimated Gain:             $95,500.00
  Tax Estimate:               $19,100.00
  After-Tax Proceeds:         $226,400.00

======================================================================

Generating visualization...
✓ Chart saved as 'TSLA_vs_SPY_chart.png'
```

## Output Files

- **Chart**: A PNG file named `{TICKER}_vs_SPY_chart.png` showing normalized price comparison
- **Cache**: Data is cached in the `cache/` directory to minimize API calls

## Technical Details

### Risk Metrics Calculation

- **Volatility**: Annualized by multiplying daily standard deviation by √252 (trading days per year)
- **Beta**: Calculated as covariance(stock, benchmark) / variance(benchmark)
- **Maximum Drawdown**: Largest percentage decline from a peak to a subsequent trough

### Data Caching

- Data is cached locally in the `cache/` directory
- Cache expires after 1 day to ensure reasonably fresh data
- Cache files are named `{TICKER}_cache.pkl`

### Rate Limiting Protection

The tool includes built-in rate limiting protection to prevent "Too Many Requests" errors:

- **Automatic delays**: Minimum 0.5 seconds between API requests
- **Retry logic**: Automatically retries up to 3 times with exponential backoff (2s, 4s, 8s)
- **Smart error detection**: Identifies rate limit errors and handles them gracefully
- **User-friendly messages**: Clear notifications when rate limited with retry progress

If you encounter rate limiting, the tool will automatically wait and retry. If all retries fail, you'll be prompted to wait a few minutes before trying again.

## Testing

The project includes comprehensive test cases to validate inputs and calculations. To run the tests:

```bash
# Make sure virtual environment is activated
source venv/bin/activate

# Run all tests
pytest test_analyzer.py -v

# Run specific test class
pytest test_analyzer.py::TestTickerValidation -v

# Run with coverage (if pytest-cov is installed)
pytest test_analyzer.py --cov=analyzer
```

### Test Coverage

The test suite includes:
- **Ticker Validation**: Valid/invalid tickers, edge cases, format validation
- **Input Validation**: Shares, cost basis, tax rate boundary conditions
- **Risk Metrics**: Volatility, beta, maximum drawdown calculations
- **Tax Simulation**: Various scenarios (profit, loss, break-even, edge cases)

## Requirements

- Python 3.7+
- yfinance >= 0.2.28
- pandas >= 2.0.0
- numpy >= 1.24.0
- matplotlib >= 3.7.0
- pytest >= 7.0.0 (for testing)

## Notes

- The tool uses SPY (S&P 500 ETF) as the default benchmark
- Analysis period is set to 5 years of historical data
- All calculations assume daily trading data
- Tax calculations are simplified estimates and should not replace professional tax advice

## Future Enhancements

This tool serves as a foundational module that can be extended to:
- Support multiple positions
- Add more sophisticated risk metrics (VaR, CVaR, etc.)
- Integrate with portfolio management systems
- Build a web-based dashboard
- Add real-time alerts and monitoring

## License

This project is for internal use at LCM.

