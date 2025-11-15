"""
Concentrated Position Risk & Liquidity Analyzer

This tool analyzes a single, concentrated stock or ETF position and provides:
- Risk metrics (Volatility, Beta, Maximum Drawdown)
- Liquidity & Tax simulation
- Historical price visualization vs. benchmark
"""

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import time
from datetime import datetime, timedelta
from typing import Callable, Any
import warnings
warnings.filterwarnings('ignore')
import requests

# Set style for better-looking plots (with fallback)
try:
    plt.style.use('seaborn-v0_8-darkgrid')
except:
    try:
        plt.style.use('seaborn-darkgrid')
    except:
        plt.style.use('default')


# Configuration
CACHE_DIR = 'cache'
YEARS_OF_DATA = 5
BENCHMARK = 'SPY'

# Rate limiting configuration
MIN_REQUEST_DELAY = 1.0  # Minimum delay between requests in seconds (increased to reduce rate limiting)
MAX_RETRIES = 2  # Maximum number of retries for rate-limited requests (reduced to fail faster)
INITIAL_RETRY_DELAY = 5  # Initial delay before retry (seconds) - increased
MAX_RETRY_DELAY = 60  # Maximum delay before retry (seconds)

# Global rate limiter - tracks last request time
_last_request_time = 0


def _rate_limit_delay():
    """
    Enforce minimum delay between API requests to prevent rate limiting.
    """
    global _last_request_time
    current_time = time.time()
    time_since_last_request = current_time - _last_request_time
    
    if time_since_last_request < MIN_REQUEST_DELAY:
        sleep_time = MIN_REQUEST_DELAY - time_since_last_request
        time.sleep(sleep_time)
    
    _last_request_time = time.time()


def _is_rate_limit_error(error: Exception) -> bool:
    """
    Check if an error is a rate limit error.
    
    Parameters:
    -----------
    error : Exception
        The exception to check
    
    Returns:
    --------
    bool
        True if the error appears to be a rate limit error
    """
    error_msg = str(error).lower()
    error_type = type(error).__name__.lower()
    
    # Check error type first (HTTP 429 is a clear indicator)
    if '429' in error_type or '429' in error_msg:
        return True
    
    # More specific rate limit indicators
    rate_limit_indicators = [
        'too many requests',
        'rate limit exceeded',
        'rate limited',
        '429',
        'throttled',
        'quota exceeded',
        'try after'
    ]
    
    # Only return True if we find a clear rate limit indicator
    # Be more conservative to avoid false positives
    for indicator in rate_limit_indicators:
        if indicator in error_msg:
            return True
    
    return False


def _retry_with_backoff(func: Callable, *args, **kwargs) -> Any:
    """
    Execute a function with exponential backoff retry logic for rate limit errors.
    
    Parameters:
    -----------
    func : Callable
        The function to execute
    *args, **kwargs
        Arguments to pass to the function
    
    Returns:
    --------
    Any
        The return value of the function
    
    Raises:
    -------
    Exception
        The last exception if all retries fail
    """
    last_exception = None
    
    for attempt in range(MAX_RETRIES + 1):
        try:
            # Enforce rate limiting before each attempt
            _rate_limit_delay()
            return func(*args, **kwargs)
        except Exception as e:
            last_exception = e
            
            # Check if it's a rate limit error
            if _is_rate_limit_error(e):
                if attempt < MAX_RETRIES:
                    # Calculate exponential backoff delay
                    delay = min(INITIAL_RETRY_DELAY * (2 ** attempt), MAX_RETRY_DELAY)
                    print(f"⚠️  Rate limited. Waiting {delay:.1f} seconds before retry ({attempt + 1}/{MAX_RETRIES})...")
                    print(f"   (This is normal - Yahoo Finance limits API requests)")
                    time.sleep(delay)
                    continue
                else:
                    raise ValueError(
                        f"\n⚠️  Rate limit exceeded after {MAX_RETRIES} retries.\n"
                        f"   Please wait 2-3 minutes and try again.\n"
                        f"   Error: {e}"
                    )
            else:
                # Not a rate limit error, re-raise immediately
                raise
    
    # Should never reach here, but just in case
    if last_exception:
        raise last_exception




def fetch_data_with_cache(ticker, benchmark=BENCHMARK, years=YEARS_OF_DATA, cache_dir=CACHE_DIR):
    """
    Fetch historical price data with caching to avoid repeated API calls.
    
    Parameters:
    -----------
    ticker : str
        Stock ticker symbol (e.g., 'TSLA')
    benchmark : str
        Benchmark ticker (default: 'SPY')
    years : int
        Number of years of historical data (default: 5)
    cache_dir : str
        Directory to store cached data
    
    Returns:
    --------
    tuple : (stock_data, benchmark_data)
        DataFrames with historical price data
    """
    # Create cache directory if it doesn't exist
    os.makedirs(cache_dir, exist_ok=True)
    
    # Define cache file paths
    stock_cache = os.path.join(cache_dir, f'{ticker}_cache.pkl')
    benchmark_cache = os.path.join(cache_dir, f'{benchmark}_cache.pkl')
    
    # Calculate date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=years * 365)
    
    # Try to load from cache
    stock_data = None
    benchmark_data = None
    
    # Check if cache exists and is recent (within 1 day)
    if os.path.exists(stock_cache):
        cache_time = datetime.fromtimestamp(os.path.getmtime(stock_cache))
        if (datetime.now() - cache_time).days < 1:
            try:
                with open(stock_cache, 'rb') as f:
                    stock_data = pickle.load(f)
                print(f"✓ Loaded {ticker} data from cache")
            except:
                pass
    
    if os.path.exists(benchmark_cache):
        cache_time = datetime.fromtimestamp(os.path.getmtime(benchmark_cache))
        if (datetime.now() - cache_time).days < 1:
            try:
                with open(benchmark_cache, 'rb') as f:
                    benchmark_data = pickle.load(f)
                print(f"✓ Loaded {benchmark} data from cache")
            except:
                pass
    
    # Fetch stock data if not cached (simple direct fetch like notebook)
    if stock_data is None:
        print(f"Fetching {ticker} data from Yahoo Finance...")
        try:
            session = requests.Session()
            stock = yf.Ticker(ticker, session=session)
            stock_data = stock.history(start=start_date, end=end_date)
            if stock_data.empty:
                raise ValueError(f"No data found for ticker {ticker}")
            # Cache the data
            with open(stock_cache, 'wb') as f:
                pickle.dump(stock_data, f)
            print(f"✓ Fetched and cached {ticker} data")
        except Exception as e:
            print(f"❌ Error fetching {ticker}: {e}")
            raise
    
    # Fetch benchmark data if not cached (simple direct fetch like notebook)
    if benchmark_data is None:
        print(f"Fetching {benchmark} data from Yahoo Finance...")
        try:
            bench = yf.Ticker(benchmark)
            benchmark_data = bench.history(start=start_date, end=end_date)
            if benchmark_data.empty:
                raise ValueError(f"No data found for ticker {benchmark}")
            # Cache the data
            with open(benchmark_cache, 'wb') as f:
                pickle.dump(benchmark_data, f)
            print(f"✓ Fetched and cached {benchmark} data")
        except Exception as e:
            print(f"❌ Error fetching {benchmark}: {e}")
            raise
    
    return stock_data, benchmark_data


def calculate_daily_returns(prices):
    """
    Calculate daily returns from price series.
    
    Parameters:
    -----------
    prices : pd.Series
        Series of closing prices
    
    Returns:
    --------
    pd.Series
        Daily returns
    """
    return prices.pct_change().dropna()


def calculate_volatility(daily_returns):
    """
    Calculate annualized volatility from daily returns.
    
    Parameters:
    -----------
    daily_returns : pd.Series
        Daily returns
    
    Returns:
    --------
    float
        Annualized volatility (as a decimal, e.g., 0.25 for 25%)
    """
    # Annualize by multiplying by sqrt(252) - number of trading days per year
    return daily_returns.std() * np.sqrt(252)


def calculate_beta(stock_returns, benchmark_returns):
    """
    Calculate beta (sensitivity to market) of stock vs benchmark.
    
    Parameters:
    -----------
    stock_returns : pd.Series
        Daily returns of the stock
    benchmark_returns : pd.Series
        Daily returns of the benchmark
    
    Returns:
    --------
    float
        Beta coefficient
    """
    # Align the returns by date
    aligned_returns = pd.DataFrame({
        'stock': stock_returns,
        'benchmark': benchmark_returns
    }).dropna()
    
    if len(aligned_returns) < 2:
        raise ValueError("Insufficient overlapping data to calculate beta")
    
    # Calculate covariance and variance
    covariance = aligned_returns['stock'].cov(aligned_returns['benchmark'])
    benchmark_variance = aligned_returns['benchmark'].var()
    
    if benchmark_variance == 0:
        return 0.0
    
    beta = covariance / benchmark_variance
    return beta


def calculate_max_drawdown(prices):
    """
    Calculate maximum drawdown (largest peak-to-trough decline).
    
    Parameters:
    -----------
    prices : pd.Series
        Series of closing prices
    
    Returns:
    --------
    tuple : (max_drawdown_pct, max_drawdown_value, peak_date, trough_date)
        Maximum drawdown as percentage, absolute value, and dates
    """
    # Calculate running maximum (peak)
    running_max = prices.expanding().max()
    
    # Calculate drawdown from peak
    drawdown = (prices - running_max) / running_max
    
    # Find maximum drawdown
    max_drawdown_idx = drawdown.idxmin()
    max_drawdown_value = drawdown.min()
    
    # Find the peak before the maximum drawdown
    peak_price = running_max.loc[:max_drawdown_idx].max()
    peak_date = running_max.loc[:max_drawdown_idx].idxmax()
    
    return abs(max_drawdown_value), abs(max_drawdown_value * peak_price), peak_date, max_drawdown_idx


def calculate_rolling_volatility(returns, window=60):
    """
    Calculate rolling annualized volatility over a specified window.
    
    Parameters:
    -----------
    returns : pd.Series
        Daily returns
    window : int
        Rolling window size in days (default: 60)
    
    Returns:
    --------
    pd.Series
        Rolling annualized volatility
    """
    rolling_std = returns.rolling(window=window).std()
    return rolling_std * np.sqrt(252)


def calculate_rolling_beta(stock_returns, benchmark_returns, window=60):
    """
    Calculate rolling beta over a specified window.
    
    Parameters:
    -----------
    stock_returns : pd.Series
        Daily returns of the stock
    benchmark_returns : pd.Series
        Daily returns of the benchmark
    window : int
        Rolling window size in days (default: 60)
    
    Returns:
    --------
    pd.Series
        Rolling beta values
    """
    # Align returns
    aligned = pd.DataFrame({
        'stock': stock_returns,
        'benchmark': benchmark_returns
    }).dropna()
    
    rolling_beta = pd.Series(index=aligned.index, dtype=float)
    
    for i in range(window, len(aligned) + 1):
        window_data = aligned.iloc[i-window:i]
        if len(window_data) >= window:
            covariance = window_data['stock'].cov(window_data['benchmark'])
            benchmark_variance = window_data['benchmark'].var()
            if benchmark_variance != 0 and not pd.isna(covariance):
                rolling_beta.iloc[i-1] = covariance / benchmark_variance
    
    return rolling_beta


def calculate_liquidity_metrics(stock_data, shares, current_price, days_30=30, days_90=90):
    """
    Calculate liquidity metrics including ADV and days to exit.
    
    Parameters:
    -----------
    stock_data : pd.DataFrame
        Stock historical data (must have 'Volume' column)
    shares : float
        Number of shares in position
    current_price : float
        Current stock price
    days_30 : int
        Number of days for 30-day ADV (default: 30)
    days_90 : int
        Number of days for 90-day ADV (default: 90)
    
    Returns:
    --------
    dict
        Dictionary with liquidity metrics
    """
    if 'Volume' not in stock_data.columns:
        return {
            'adv_30': None,
            'adv_90': None,
            'dollar_adv_30': None,
            'dollar_adv_90': None,
            'days_to_exit_30': None,
            'days_to_exit_90': None,
            'position_pct_adv_30': None,
            'position_pct_adv_90': None,
            'is_illiquid_30': False,
            'is_illiquid_90': False
        }
    
    volume = stock_data['Volume']
    
    # Calculate ADV for last 30 and 90 days
    recent_30 = volume.tail(days_30)
    recent_90 = volume.tail(days_90)
    
    adv_30 = recent_30.mean() if len(recent_30) > 0 else 0
    adv_90 = recent_90.mean() if len(recent_90) > 0 else 0
    
    # Dollar ADV
    dollar_adv_30 = adv_30 * current_price
    dollar_adv_90 = adv_90 * current_price
    
    # Days to exit position
    days_to_exit_30 = shares / adv_30 if adv_30 > 0 else float('inf')
    days_to_exit_90 = shares / adv_90 if adv_90 > 0 else float('inf')
    
    # Position as % of ADV
    position_pct_adv_30 = (shares / adv_30 * 100) if adv_30 > 0 else float('inf')
    position_pct_adv_90 = (shares / adv_90 * 100) if adv_90 > 0 else float('inf')
    
    # Flag if illiquid (>10% of ADV)
    is_illiquid_30 = position_pct_adv_30 > 10
    is_illiquid_90 = position_pct_adv_90 > 10
    
    return {
        'adv_30': adv_30,
        'adv_90': adv_90,
        'dollar_adv_30': dollar_adv_30,
        'dollar_adv_90': dollar_adv_90,
        'days_to_exit_30': days_to_exit_30,
        'days_to_exit_90': days_to_exit_90,
        'position_pct_adv_30': position_pct_adv_30,
        'position_pct_adv_90': position_pct_adv_90,
        'is_illiquid_30': is_illiquid_30,
        'is_illiquid_90': is_illiquid_90
    }


def check_concentration_risks(volatility, beta, max_drawdown_pct, liquidity_metrics):
    """
    Check for concentration risk flags and return warnings.
    
    Parameters:
    -----------
    volatility : float
        Annualized volatility (as decimal)
    beta : float
        Beta coefficient
    max_drawdown_pct : float
        Maximum drawdown as decimal
    liquidity_metrics : dict
        Liquidity metrics dictionary
    
    Returns:
    --------
    list
        List of warning messages
    """
    warnings = []
    
    # Beta > 1.5
    if beta > 1.5:
        warnings.append(f"⚠️  High Beta ({beta:.2f}): Stock is highly sensitive to market movements")
    
    # Volatility > 40%
    if volatility > 0.40:
        warnings.append(f"⚠️  High Volatility ({volatility*100:.1f}%): Stock shows significant price swings")
    
    # Max drawdown > 50%
    if max_drawdown_pct > 0.50:
        warnings.append(f"⚠️  Large Drawdown ({max_drawdown_pct*100:.1f}%): Stock experienced significant decline")
    
    # Liquidity low
    if liquidity_metrics.get('is_illiquid_30', False):
        pct = liquidity_metrics.get('position_pct_adv_30', 0)
        warnings.append(f"⚠️  Low Liquidity: Position represents {pct:.1f}% of 30-day ADV (illiquid)")
    elif liquidity_metrics.get('is_illiquid_90', False):
        pct = liquidity_metrics.get('position_pct_adv_90', 0)
        warnings.append(f"⚠️  Low Liquidity: Position represents {pct:.1f}% of 90-day ADV (illiquid)")
    
    return warnings


def calculate_scenario_analysis(total_shares, current_price, cost_basis, tax_rate):
    """
    Calculate scenario analysis for different sell percentages.
    
    Parameters:
    -----------
    total_shares : float
        Total shares in position
    current_price : float
        Current stock price
    cost_basis : float
        Average cost basis per share
    tax_rate : float
        Tax rate as decimal
    
    Returns:
    --------
    list
        List of dictionaries with scenario results
    """
    scenarios = [0.10, 0.25, 0.50, 1.0]  # 10%, 25%, 50%, 100%
    results = []
    
    for pct in scenarios:
        shares_to_sell = total_shares * pct
        tax_result = calculate_tax_estimate(shares_to_sell, current_price, cost_basis, tax_rate)
        tax_result['percentage'] = pct * 100
        tax_result['shares'] = shares_to_sell
        results.append(tax_result)
    
    return results


def calculate_tax_estimate(shares, current_price, cost_basis, tax_rate):
    """
    Calculate estimated proceeds and tax from selling shares.
    
    Parameters:
    -----------
    shares : float
        Number of shares to sell
    current_price : float
        Current price per share
    cost_basis : float
        Average cost basis per share
    tax_rate : float
        Tax rate as decimal (e.g., 0.20 for 20%)
    
    Returns:
    --------
    dict
        Dictionary with proceeds, gain, and tax estimate
    """
    proceeds = shares * current_price
    total_cost_basis = shares * cost_basis
    gain = proceeds - total_cost_basis
    tax_estimate = gain * tax_rate if gain > 0 else 0
    after_tax_proceeds = proceeds - tax_estimate
    
    return {
        'proceeds': proceeds,
        'total_cost_basis': total_cost_basis,
        'gain': gain,
        'tax_estimate': tax_estimate,
        'after_tax_proceeds': after_tax_proceeds
    }


def plot_rolling_risk_metrics(stock_returns, benchmark_returns, ticker, benchmark=BENCHMARK, window=60):
    """
    Plot rolling volatility and beta over time.
    
    Parameters:
    -----------
    stock_returns : pd.Series
        Stock daily returns
    benchmark_returns : pd.Series
        Benchmark daily returns
    ticker : str
        Stock ticker symbol
    benchmark : str
        Benchmark ticker symbol
    window : int
        Rolling window size in days (default: 60)
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    # Calculate rolling metrics
    rolling_vol = calculate_rolling_volatility(stock_returns, window=window)
    rolling_beta = calculate_rolling_beta(stock_returns, benchmark_returns, window=window)
    
    # Plot rolling volatility
    ax1.plot(rolling_vol.index, rolling_vol.values * 100, 
             label=f'{ticker} Rolling Volatility ({window}-day)', 
             linewidth=2, alpha=0.8, color='#2E86AB')
    ax1.axhline(y=40, color='r', linestyle='--', alpha=0.5, label='40% Threshold (High Risk)')
    ax1.set_ylabel('Volatility (%)', fontsize=12)
    ax1.set_title(f'{ticker} - Rolling Risk Metrics ({window}-Day Window)', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Plot rolling beta
    ax2.plot(rolling_beta.index, rolling_beta.values, 
             label=f'{ticker} Rolling Beta vs {benchmark} ({window}-day)', 
             linewidth=2, alpha=0.8, color='#A23B72')
    ax2.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='Beta = 1.0 (Market)')
    ax2.axhline(y=1.5, color='r', linestyle='--', alpha=0.5, label='1.5 Threshold (High Risk)')
    ax2.set_xlabel('Date', fontsize=12)
    ax2.set_ylabel('Beta', fontsize=12)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    filename = f'{ticker}_rolling_risk_metrics.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"✓ Rolling risk metrics chart saved as '{filename}'")
    plt.show()


def plot_price_comparison(stock_data, benchmark_data, ticker, benchmark=BENCHMARK):
    """
    Plot historical price comparison between stock and benchmark.
    
    Parameters:
    -----------
    stock_data : pd.DataFrame
        Stock historical data
    benchmark_data : pd.DataFrame
        Benchmark historical data
    ticker : str
        Stock ticker symbol
    benchmark : str
        Benchmark ticker symbol
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Normalize prices to start at 100 for easier comparison
    stock_prices = stock_data['Close']
    benchmark_prices = benchmark_data['Close']
    
    # Align dates
    common_dates = stock_prices.index.intersection(benchmark_prices.index)
    stock_aligned = stock_prices.loc[common_dates]
    benchmark_aligned = benchmark_prices.loc[common_dates]
    
    # Normalize to starting value of 100
    stock_normalized = (stock_aligned / stock_aligned.iloc[0]) * 100
    benchmark_normalized = (benchmark_aligned / benchmark_aligned.iloc[0]) * 100
    
    # Plot
    ax.plot(stock_normalized.index, stock_normalized.values, 
            label=ticker, linewidth=2, alpha=0.8)
    ax.plot(benchmark_normalized.index, benchmark_normalized.values, 
            label=benchmark, linewidth=2, alpha=0.8, linestyle='--')
    
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Normalized Price (Starting at 100)', fontsize=12)
    ax.set_title(f'{ticker} vs {benchmark} - Historical Price Comparison', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{ticker}_vs_{benchmark}_chart.png', dpi=300, bbox_inches='tight')
    print(f"\nChart saved as '{ticker}_vs_{benchmark}_chart.png'")
    plt.show()


def print_analysis_summary(ticker, volatility, beta, max_drawdown_pct, max_drawdown_value, 
                          peak_date, trough_date, current_price, tax_results, 
                          liquidity_metrics, concentration_warnings, scenario_results):
    """
    Print a formatted summary of the analysis.
    
    Parameters:
    -----------
    ticker : str
        Stock ticker
    volatility : float
        Annualized volatility
    beta : float
        Beta coefficient
    max_drawdown_pct : float
        Maximum drawdown as percentage
    max_drawdown_value : float
        Maximum drawdown absolute value
    peak_date : datetime
        Date of peak before max drawdown
    trough_date : datetime
        Date of trough (max drawdown)
    current_price : float
        Current stock price
    tax_results : dict
        Results from tax calculation
    """
    print("\n" + "="*70)
    print(f"CONCENTRATED POSITION ANALYSIS: {ticker}")
    print("="*70)
    
    print(f"\nCURRENT PRICE: ${current_price:.2f}")
    
    print("\n" + "-"*70)
    print("RISK METRICS")
    print("-"*70)
    print(f"  Volatility (Annualized):     {volatility*100:.2f}%")
    print(f"  Beta vs. SPY:                {beta:.2f}")
    print(f"  Maximum Drawdown:            {max_drawdown_pct*100:.2f}%")
    print(f"  Max Drawdown Period:         {peak_date.strftime('%Y-%m-%d')} to {trough_date.strftime('%Y-%m-%d')}")
    print(f"  Max Drawdown Value:          ${max_drawdown_value:.2f}")
    
    # Concentration Risk Flags
    if concentration_warnings:
        print("\n" + "-"*70)
        print("⚠️  CONCENTRATION RISK WARNINGS")
        print("-"*70)
        for warning in concentration_warnings:
            print(f"  {warning}")
    
    # Liquidity Metrics
    if liquidity_metrics.get('adv_30') is not None:
        print("\n" + "-"*70)
        print("LIQUIDITY METRICS")
        print("-"*70)
        print(f"  Average Daily Volume (30-day):  {liquidity_metrics['adv_30']:,.0f} shares")
        print(f"  Average Daily Volume (90-day):  {liquidity_metrics['adv_90']:,.0f} shares")
        print(f"  Dollar ADV (30-day):            ${liquidity_metrics['dollar_adv_30']:,.2f}")
        print(f"  Dollar ADV (90-day):            ${liquidity_metrics['dollar_adv_90']:,.2f}")
        
        days_30 = liquidity_metrics['days_to_exit_30']
        days_90 = liquidity_metrics['days_to_exit_90']
        if days_30 != float('inf'):
            print(f"  Days to Exit (30-day ADV):      {days_30:.1f} days")
        else:
            print(f"  Days to Exit (30-day ADV):      N/A")
        if days_90 != float('inf'):
            print(f"  Days to Exit (90-day ADV):      {days_90:.1f} days")
        else:
            print(f"  Days to Exit (90-day ADV):      N/A")
        
        pct_30 = liquidity_metrics['position_pct_adv_30']
        pct_90 = liquidity_metrics['position_pct_adv_90']
        if pct_30 != float('inf'):
            print(f"  Position as % of 30-day ADV:   {pct_30:.2f}%")
        if pct_90 != float('inf'):
            print(f"  Position as % of 90-day ADV:   {pct_90:.2f}%")
    
    print("\n" + "-"*70)
    print("LIQUIDITY & TAX SIMULATION")
    print("-"*70)
    print(f"  Shares to Sell:              {tax_results['shares']:,.0f}")
    print(f"  Estimated Proceeds:           ${tax_results['proceeds']:,.2f}")
    print(f"  Total Cost Basis:            ${tax_results['total_cost_basis']:,.2f}")
    print(f"  Estimated Gain:              ${tax_results['gain']:,.2f}")
    print(f"  Tax Estimate:                ${tax_results['tax_estimate']:,.2f}")
    print(f"  After-Tax Proceeds:          ${tax_results['after_tax_proceeds']:,.2f}")
    
    # Scenario Analysis
    if scenario_results:
        print("\n" + "-"*70)
        print("SCENARIO ANALYSIS")
        print("-"*70)
        print(f"{'Sell %':<10} {'Shares':<15} {'Proceeds':<15} {'Gain':<15} {'Tax':<15} {'After-Tax':<15}")
        print("-" * 70)
        for scenario in scenario_results:
            pct = scenario['percentage']
            shares = scenario['shares']
            proceeds = scenario['proceeds']
            gain = scenario['gain']
            tax = scenario['tax_estimate']
            after_tax = scenario['after_tax_proceeds']
            print(f"{pct:>5.0f}%     {shares:>12,.0f}  ${proceeds:>12,.0f}  ${gain:>12,.0f}  ${tax:>12,.0f}  ${after_tax:>12,.0f}")
    
    print("\n" + "="*70 + "\n")


def validate_ticker(ticker):
    """
    Validate that a ticker exists and is accessible via Yahoo Finance.
    
    Parameters:
    -----------
    ticker : str
        Stock ticker symbol to validate
    
    Returns:
    --------
    bool
        True if ticker is valid, False otherwise
    
    Raises:
    -------
    ValueError
        If ticker is invalid or cannot be accessed
    """
    # Handle None or empty ticker
    if ticker is None:
        raise ValueError("Ticker cannot be empty")
    
    ticker = str(ticker).strip().upper()
    
    if not ticker:
        raise ValueError("Ticker cannot be empty")
    
    # Basic format validation (alphanumeric, typically 1-5 characters)
    if not ticker.replace('.', '').replace('-', '').isalnum():
        raise ValueError(f"Invalid ticker format: {ticker}")
    
    # Simple validation like notebook - just try to fetch 1 day of data
    try:
        print(f"Validating ticker {ticker}...")
        stock = yf.Ticker(ticker)
        hist = stock.history(period="1d")
        if hist.empty:
            raise ValueError(f"Ticker {ticker} not found or has no trading data")
        # If we got history data, ticker is valid
        return True
    except ValueError:
        # Re-raise ValueError as-is (our own validation errors)
        raise
    except Exception as e:
        # More specific error handling for network/API errors
        error_msg = str(e).lower()
        if _is_rate_limit_error(e):
            # Rate limit errors should have been handled by retry logic
            # If we get here, all retries failed
            raise ValueError(
                f"\n⚠️  Rate limit exceeded while validating ticker {ticker}.\n"
                f"   Yahoo Finance is temporarily limiting requests.\n"
                f"   Solutions:\n"
                f"   1. Wait 2-3 minutes and try again\n"
                f"   2. Try a different ticker that was recently cached\n"
                f"   3. Check your internet connection\n"
            )
        elif 'not found' in error_msg or 'invalid' in error_msg or 'no data' in error_msg:
            raise ValueError(f"Ticker {ticker} not found. Please check the ticker symbol and try again.")
        elif 'timeout' in error_msg or 'connection' in error_msg or 'network' in error_msg:
            raise ValueError(f"Connection error while validating ticker {ticker}. Please check your internet connection.")
        else:
            raise ValueError(f"Error validating ticker {ticker}: {e}")


def get_user_inputs():
    """
    Prompt user for required inputs with validation.
    
    Returns:
    --------
    tuple : (ticker, shares, cost_basis, tax_rate)
    """
    print("\n" + "="*70)
    print("CONCENTRATED POSITION RISK & LIQUIDITY ANALYZER")
    print("="*70 + "\n")
    
    # Get ticker with validation
    while True:
        ticker = input("Enter stock ticker (e.g., TSLA): ").strip().upper()
        if not ticker:
            print("Ticker cannot be empty. Please try again.")
            continue
        
        try:
            validate_ticker(ticker)
            print(f"✓ Validated ticker: {ticker}")
            break
        except ValueError as e:
            print(f"❌ {e}")
            retry = input("Would you like to try another ticker? (y/n): ").strip().lower()
            if retry != 'y':
                raise
            continue
    
    # Get shares
    while True:
        try:
            shares = float(input("Enter number of shares to hypothetically sell: "))
            if shares <= 0:
                print("Shares must be greater than 0. Please try again.")
                continue
            break
        except ValueError:
            print("Invalid input. Please enter a number.")
    
    # Get cost basis
    while True:
        try:
            cost_basis = float(input("Enter average cost basis per share ($): "))
            if cost_basis < 0:
                print("Cost basis cannot be negative. Please try again.")
                continue
            break
        except ValueError:
            print("Invalid input. Please enter a number.")
    
    # Get tax rate
    while True:
        try:
            tax_rate_input = input("Enter tax rate as decimal (e.g., 0.20 for 20%): ")
            tax_rate = float(tax_rate_input)
            if tax_rate < 0 or tax_rate > 1:
                print("Tax rate must be between 0 and 1. Please try again.")
                continue
            break
        except ValueError:
            print("Invalid input. Please enter a number between 0 and 1.")
    
    return ticker, shares, cost_basis, tax_rate


def main():
    """
    Main execution function.
    """
    try:
        # Get user inputs
        ticker, shares, cost_basis, tax_rate = get_user_inputs()
        
        print(f"\nAnalyzing {ticker}...")
        print("-" * 70)
        
        # Fetch data
        stock_data, benchmark_data = fetch_data_with_cache(ticker)
        
        # Extract closing prices
        stock_prices = stock_data['Close']
        benchmark_prices = benchmark_data['Close']
        
        # Calculate daily returns
        stock_returns = calculate_daily_returns(stock_prices)
        benchmark_returns = calculate_daily_returns(benchmark_prices)
        
        # Calculate risk metrics
        print("\nCalculating risk metrics...")
        volatility = calculate_volatility(stock_returns)
        beta = calculate_beta(stock_returns, benchmark_returns)
        max_drawdown_pct, max_drawdown_value, peak_date, trough_date = calculate_max_drawdown(stock_prices)
        
        # Get current price (most recent closing price)
        current_price = stock_prices.iloc[-1]
        
        # Calculate liquidity metrics
        print("Calculating liquidity metrics...")
        liquidity_metrics = calculate_liquidity_metrics(stock_data, shares, current_price)
        
        # Check concentration risks
        print("Checking concentration risks...")
        concentration_warnings = check_concentration_risks(volatility, beta, max_drawdown_pct, liquidity_metrics)
        
        # Calculate tax simulation
        print("Calculating tax simulation...")
        tax_results = calculate_tax_estimate(shares, current_price, cost_basis, tax_rate)
        tax_results['shares'] = shares  # Add shares to results for printing
        
        # Calculate scenario analysis
        print("Calculating scenario analysis...")
        scenario_results = calculate_scenario_analysis(shares, current_price, cost_basis, tax_rate)
        
        # Print summary
        print_analysis_summary(
            ticker, volatility, beta, max_drawdown_pct, max_drawdown_value,
            peak_date, trough_date, current_price, tax_results,
            liquidity_metrics, concentration_warnings, scenario_results
        )
        
        # Generate visualizations
        print("Generating visualizations...")
        plot_price_comparison(stock_data, benchmark_data, ticker)
        plot_rolling_risk_metrics(stock_returns, benchmark_returns, ticker)
        
        print("\n✓ Analysis complete!")
        
    except KeyboardInterrupt:
        print("\n\nAnalysis cancelled by user.")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

