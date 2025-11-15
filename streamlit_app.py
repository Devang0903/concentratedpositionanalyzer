"""
Concentrated Position Risk & Liquidity Analyzer - Streamlit App

A user-friendly web application for analyzing concentrated stock positions.
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Concentrated Position Analyzer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configuration
YEARS_OF_DATA = 5

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .stMetric {
        background-color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    </style>
    """, unsafe_allow_html=True)

def validate_ticker(ticker):
    """Validate that a ticker exists and has data available."""
    if not ticker:
        return False, "Ticker cannot be empty"
    
    # Basic format validation (alphanumeric, typically 1-5 characters, may include dots/dashes)
    if not ticker.replace('.', '').replace('-', '').isalnum():
        return False, f"Invalid ticker format: {ticker}. Tickers should be alphanumeric (e.g., AAPL, BRK.B, SPY)"
    
    # Try to fetch 1 day of data to verify ticker exists
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="1d")
        if hist.empty:
            return False, f"Ticker '{ticker}' not found or has no trading data. Please check the ticker symbol."
        return True, None
    except Exception as e:
        error_msg = str(e).lower()
        if 'not found' in error_msg or 'invalid' in error_msg or 'no data' in error_msg:
            return False, f"Ticker '{ticker}' not found. Please check the ticker symbol and try again."
        elif 'timeout' in error_msg or 'connection' in error_msg or 'network' in error_msg:
            return False, f"Connection error while validating ticker '{ticker}'. Please check your internet connection."
        else:
            return False, f"Error validating ticker '{ticker}': {str(e)}"

def fetch_data(ticker, benchmark, years):
    """Fetch historical data for ticker and benchmark."""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=years * 365)
    
    try:
        # Let yfinance handle session management automatically
        stock = yf.Ticker(ticker)
        stock_data = stock.history(start=start_date, end=end_date)
        
        bench = yf.Ticker(benchmark)
        benchmark_data = bench.history(start=start_date, end=end_date)
        
        if stock_data.empty or benchmark_data.empty:
            if stock_data.empty:
                return None, None, f"No data found for ticker '{ticker}'. Please verify the ticker symbol."
            else:
                return None, None, f"No data found for benchmark '{benchmark}'. Please try a different benchmark."
        
        return stock_data, benchmark_data, None
    except Exception as e:
        return None, None, f"Error fetching data: {str(e)}"

def calculate_risk_metrics(stock_prices, benchmark_prices):
    """Calculate risk metrics: volatility, beta, and maximum drawdown."""
    # Calculate daily returns
    stock_returns = stock_prices.pct_change().dropna()
    benchmark_returns = benchmark_prices.pct_change().dropna()
    
    # Volatility (annualized)
    volatility = stock_returns.std() * np.sqrt(252)
    
    # Beta
    aligned_returns = pd.DataFrame({
        'stock': stock_returns,
        'benchmark': benchmark_returns
    }).dropna()
    
    if len(aligned_returns) < 2:
        return None, None, None, None, None, None, None, "Insufficient data for calculations"
    
    covariance = aligned_returns['stock'].cov(aligned_returns['benchmark'])
    benchmark_variance = aligned_returns['benchmark'].var()
    beta = covariance / benchmark_variance if benchmark_variance != 0 else 0.0
    
    # Maximum Drawdown
    running_max = stock_prices.expanding().max()
    drawdown = (stock_prices - running_max) / running_max
    max_drawdown = abs(drawdown.min())
    max_drawdown_idx = drawdown.idxmin()
    
    # Find peak before maximum drawdown
    peak_date = running_max.loc[:max_drawdown_idx].idxmax()
    trough_date = max_drawdown_idx
    
    return volatility, beta, max_drawdown, peak_date, trough_date, stock_returns, benchmark_returns, aligned_returns, None

def calculate_tax_simulation(shares, current_price, cost_basis, tax_rate):
    """Calculate tax simulation results."""
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

def calculate_additional_metrics(stock_prices, benchmark_prices, stock_returns, benchmark_returns, aligned_returns, volatility, beta, years):
    """Calculate additional metrics: correlation, Sharpe ratio, returns, etc."""
    window = 60
    
    # Rolling correlation (60-day window)
    rolling_correlation = aligned_returns['stock'].rolling(window=window).corr(aligned_returns['benchmark'])
    
    # Rolling volatility (60-day window)
    rolling_volatility = stock_returns.rolling(window=window).std() * np.sqrt(252)
    
    # Rolling beta (60-day window)
    rolling_beta = pd.Series(index=aligned_returns.index, dtype=float)
    for i in range(window, len(aligned_returns) + 1):
        window_data = aligned_returns.iloc[i-window:i]
        if len(window_data) >= window:
            cov = window_data['stock'].cov(window_data['benchmark'])
            var = window_data['benchmark'].var()
            if var != 0 and not pd.isna(cov):
                rolling_beta.iloc[i-1] = cov / var
    
    # Overall correlation
    overall_correlation = aligned_returns['stock'].corr(aligned_returns['benchmark'])
    
    # Recent correlation (30-day)
    recent_correlation = aligned_returns.tail(30)['stock'].corr(aligned_returns.tail(30)['benchmark'])
    
    # Sharpe Ratio (assuming risk-free rate of 0)
    mean_return = stock_returns.mean() * 252  # Annualized mean return
    sharpe_ratio = mean_return / volatility if volatility > 0 else 0
    
    # Return metrics
    total_return = (stock_prices.iloc[-1] / stock_prices.iloc[0] - 1) * 100
    benchmark_return = (benchmark_prices.iloc[-1] / benchmark_prices.iloc[0] - 1) * 100
    excess_return = total_return - benchmark_return
    avg_daily_return = stock_returns.mean() * 100
    
    # Trading days analysis
    positive_days = (stock_returns > 0).sum()
    negative_days = (stock_returns < 0).sum()
    win_rate = (positive_days / len(stock_returns)) * 100
    
    # Extreme moves
    best_day_return = stock_returns.max() * 100
    worst_day_return = stock_returns.min() * 100
    best_day_date = stock_returns.idxmax()
    worst_day_date = stock_returns.idxmin()
    
    return {
        'rolling_correlation': rolling_correlation,
        'rolling_volatility': rolling_volatility,
        'rolling_beta': rolling_beta,
        'overall_correlation': overall_correlation,
        'recent_correlation': recent_correlation,
        'sharpe_ratio': sharpe_ratio,
        'mean_return': mean_return,
        'total_return': total_return,
        'benchmark_return': benchmark_return,
        'excess_return': excess_return,
        'avg_daily_return': avg_daily_return,
        'positive_days': positive_days,
        'negative_days': negative_days,
        'win_rate': win_rate,
        'best_day_return': best_day_return,
        'worst_day_return': worst_day_return,
        'best_day_date': best_day_date,
        'worst_day_date': worst_day_date,
        'window': window
    }

def create_price_chart(stock_prices, benchmark_prices, ticker, benchmark):
    """Create normalized price comparison chart."""
    # Align dates
    common_dates = stock_prices.index.intersection(benchmark_prices.index)
    stock_aligned = stock_prices.loc[common_dates]
    benchmark_aligned = benchmark_prices.loc[common_dates]
    
    # Normalize to starting value of 100
    stock_normalized = (stock_aligned / stock_aligned.iloc[0]) * 100
    benchmark_normalized = (benchmark_aligned / benchmark_aligned.iloc[0]) * 100
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(stock_normalized.index, stock_normalized.values, 
            label=ticker, linewidth=2, alpha=0.8, color='#1f77b4')
    ax.plot(benchmark_normalized.index, benchmark_normalized.values, 
            label=benchmark, linewidth=2, alpha=0.8, linestyle='--', color='#ff7f0e')
    
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Normalized Price (Starting at 100)', fontsize=12)
    ax.set_title(f'{ticker} vs {benchmark} - Historical Price Comparison', 
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def create_rolling_metrics_chart(rolling_correlation, rolling_volatility, rolling_beta, 
                                  overall_correlation, volatility, beta, ticker, benchmark, window):
    """Create rolling correlation and risk metrics visualization."""
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    
    # Plot 1: Rolling Correlation
    axes[0].plot(rolling_correlation.index, rolling_correlation.values, 
                 label=f'{ticker} vs {benchmark} Rolling Correlation ({window}-day)', 
                 linewidth=2, alpha=0.8, color='#2E86AB')
    axes[0].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    axes[0].axhline(y=overall_correlation, color='r', linestyle='--', alpha=0.5, 
                    label=f'Overall Correlation: {overall_correlation:.3f}')
    axes[0].set_ylabel('Correlation', fontsize=11)
    axes[0].set_title(f'{ticker} - Rolling Correlation & Risk Metrics ({window}-Day Window)', 
                      fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim([-1, 1])
    
    # Plot 2: Rolling Volatility
    axes[1].plot(rolling_volatility.index, rolling_volatility.values * 100, 
                 label=f'{ticker} Rolling Volatility ({window}-day)', 
                 linewidth=2, alpha=0.8, color='#A23B72')
    axes[1].axhline(y=volatility*100, color='r', linestyle='--', alpha=0.5, 
                    label=f'Overall Volatility: {volatility*100:.2f}%')
    axes[1].set_ylabel('Volatility (%)', fontsize=11)
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)
    
    # Plot 3: Rolling Beta
    axes[2].plot(rolling_beta.index, rolling_beta.values, 
                 label=f'{ticker} Rolling Beta vs {benchmark} ({window}-day)', 
                 linewidth=2, alpha=0.8, color='#F18F01')
    axes[2].axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='Beta = 1.0 (Market)')
    axes[2].axhline(y=beta, color='r', linestyle='--', alpha=0.5, label=f'Overall Beta: {beta:.2f}')
    axes[2].set_xlabel('Date', fontsize=12)
    axes[2].set_ylabel('Beta', fontsize=11)
    axes[2].legend(fontsize=9)
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

# Main App
def main():
    # Header
    st.markdown('<p class="main-header">📊 Concentrated Position Risk & Liquidity Analyzer</p>', 
                unsafe_allow_html=True)
    
    st.markdown("""
    This tool analyzes a single, concentrated stock or ETF position and provides:
    - **Risk metrics**: Volatility, Beta, and Maximum Drawdown
    - **Liquidity & Tax simulation**: Estimated proceeds and tax calculations
    - **Visualization**: Historical price comparison vs. benchmark
    """)
    
    st.divider()
    
    # Sidebar for inputs
    with st.sidebar:
        st.header("📝 Input Parameters")
        
        ticker = st.text_input(
            "Stock Ticker",
            value="TSLA",
            help="Enter the stock or ETF ticker symbol (e.g., TSLA, AAPL, MSFT)"
        ).upper().strip()
        
        benchmark = st.selectbox(
            "Benchmark",
            options=["SPY", "QQQ", "DIA", "IWM", "VTI", "VOO", "VEA", "VWO", "AGG", "BND"],
            index=0,
            help="Select a benchmark ETF for comparison (SPY = S&P 500, QQQ = Nasdaq 100, etc.)"
        )
        
        shares = st.number_input(
            "Number of Shares to Sell",
            min_value=0.0,
            value=1000.0,
            step=100.0,
            help="Enter the number of shares you want to hypothetically sell"
        )
        
        cost_basis = st.number_input(
            "Average Cost Basis per Share ($)",
            min_value=0.0,
            value=150.0,
            step=10.0,
            format="%.2f",
            help="Enter your average purchase price per share"
        )
        
        tax_rate = st.slider(
            "Tax Rate (%)",
            min_value=0.0,
            max_value=50.0,
            value=20.0,
            step=1.0,
            help="Enter your expected tax rate (e.g., 20% for long-term capital gains)"
        ) / 100  # Convert to decimal
        
        analyze_button = st.button("🔍 Analyze Position", type="primary", use_container_width=True)
    
    # Main content area
    if analyze_button:
        if not ticker:
            st.error("⚠️ Please enter a stock ticker.")
            return
        
        # Validate ticker before fetching data
        with st.spinner(f"Validating ticker {ticker}..."):
            is_valid, validation_error = validate_ticker(ticker)
            if not is_valid:
                st.error(f"❌ {validation_error}")
                st.info("💡 **Tip**: Make sure you're using the correct ticker symbol (e.g., AAPL for Apple, TSLA for Tesla). Some tickers may include dots (e.g., BRK.B) or dashes.")
                return
        
        # Validate benchmark
        with st.spinner(f"Validating benchmark {benchmark}..."):
            is_valid, validation_error = validate_ticker(benchmark)
            if not is_valid:
                st.error(f"❌ Benchmark validation failed: {validation_error}")
                st.info("💡 **Tip**: Please select a different benchmark from the dropdown.")
                return
        
        # Show loading spinner
        with st.spinner(f"Fetching data for {ticker} and {benchmark}..."):
            stock_data, benchmark_data, error = fetch_data(ticker, benchmark, YEARS_OF_DATA)
        
        if error:
            st.error(f"❌ {error}")
            return
        
        if stock_data is None or benchmark_data is None:
            st.error("❌ Failed to fetch data. Please check your ticker symbol and try again.")
            return
        
        st.success(f"✓ Successfully fetched {len(stock_data)} trading days of data")
        
        # Extract prices
        stock_prices = stock_data['Close']
        benchmark_prices = benchmark_data['Close']
        current_price = stock_prices.iloc[-1]
        
        # Calculate risk metrics
        with st.spinner("Calculating risk metrics..."):
            volatility, beta, max_drawdown, peak_date, trough_date, stock_returns, benchmark_returns, aligned_returns, calc_error = calculate_risk_metrics(
                stock_prices, benchmark_prices
            )
        
        if calc_error:
            st.error(f"❌ {calc_error}")
            return
        
        # Calculate additional metrics
        with st.spinner("Calculating additional metrics..."):
            additional_metrics = calculate_additional_metrics(
                stock_prices, benchmark_prices, stock_returns, benchmark_returns, 
                aligned_returns, volatility, beta, YEARS_OF_DATA
            )
        
        # Calculate tax simulation
        tax_results = calculate_tax_simulation(shares, current_price, cost_basis, tax_rate)
        
        # Display results
        st.divider()
        
        # Current Price
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Current Price", f"${current_price:,.2f}")
        with col2:
            st.metric("Total Position Value", f"${shares * current_price:,.2f}")
        with col3:
            st.metric("Total Cost Basis", f"${tax_results['total_cost_basis']:,.2f}")
        
        st.divider()
        
        # Risk Metrics Section
        st.header("📈 Risk Metrics")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Volatility (Annualized)",
                f"{volatility*100:.2f}%",
                help="Annualized standard deviation of daily returns"
            )
        
        with col2:
            st.metric(
                f"Beta vs. {benchmark}",
                f"{beta:.2f}",
                help="Stock's sensitivity to market movements (1.0 = moves with market)"
            )
        
        with col3:
            st.metric(
                "Maximum Drawdown",
                f"{max_drawdown*100:.2f}%",
                help="Largest peak-to-trough decline over the period"
            )
        
        # Drawdown period
        st.info(f"📅 **Maximum Drawdown Period**: {peak_date.strftime('%Y-%m-%d')} to {trough_date.strftime('%Y-%m-%d')}")
        
        st.divider()
        
        # Liquidity & Tax Simulation Section
        st.header("💰 Liquidity & Tax Simulation")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Sale Details")
            st.metric("Shares to Sell", f"{shares:,.0f}")
            st.metric("Estimated Proceeds", f"${tax_results['proceeds']:,.2f}")
            st.metric("Estimated Gain", f"${tax_results['gain']:,.2f}")
        
        with col2:
            st.subheader("Tax Impact")
            st.metric("Tax Rate", f"{tax_rate*100:.0f}%")
            st.metric("Tax Estimate", f"${tax_results['tax_estimate']:,.2f}")
            st.metric("After-Tax Proceeds", f"${tax_results['after_tax_proceeds']:,.2f}")
        
        # Visual breakdown
        st.subheader("Proceeds Breakdown")
        breakdown_data = pd.DataFrame({
            'Category': ['After-Tax Proceeds', 'Taxes', 'Cost Basis'],
            'Amount': [
                tax_results['after_tax_proceeds'],
                tax_results['tax_estimate'],
                tax_results['total_cost_basis']
            ]
        })
        
        # Create a simple bar chart
        fig_breakdown, ax_breakdown = plt.subplots(figsize=(10, 4))
        colors = ['#2ecc71', '#e74c3c', '#3498db']
        bars = ax_breakdown.barh(breakdown_data['Category'], breakdown_data['Amount'], color=colors)
        ax_breakdown.set_xlabel('Amount ($)', fontsize=11)
        ax_breakdown.set_title('Proceeds Breakdown', fontsize=12, fontweight='bold')
        ax_breakdown.grid(True, alpha=0.3, axis='x')
        
        # Add value labels on bars
        for i, (bar, value) in enumerate(zip(bars, breakdown_data['Amount'])):
            ax_breakdown.text(value, i, f'${value:,.0f}', 
                            va='center', ha='left', fontweight='bold', fontsize=10)
        
        plt.tight_layout()
        st.pyplot(fig_breakdown)
        
        st.divider()
        
        # Visualization Section
        st.header("📊 Historical Price Comparison")
        
        with st.spinner("Generating chart..."):
            fig = create_price_chart(stock_prices, benchmark_prices, ticker, benchmark)
            st.pyplot(fig)
        
        st.caption(f"Chart shows normalized price performance (starting at 100) for {ticker} vs {benchmark} over the last {YEARS_OF_DATA} years")
        
        st.divider()
        
        # Rolling Correlation & Risk Metrics Section
        st.header("📊 Rolling Correlation & Risk Metrics")
        
        with st.spinner("Generating rolling metrics chart..."):
            fig_rolling = create_rolling_metrics_chart(
                additional_metrics['rolling_correlation'],
                additional_metrics['rolling_volatility'],
                additional_metrics['rolling_beta'],
                additional_metrics['overall_correlation'],
                volatility,
                beta,
                ticker,
                benchmark,
                additional_metrics['window']
            )
            st.pyplot(fig_rolling)
        
        st.caption(f"Rolling metrics calculated using a {additional_metrics['window']}-day window. Reference lines show overall metrics.")
        
        st.divider()
        
        # Additional Statistics Section
        st.header("📈 Additional Statistics & Insights")
        
        # Return Metrics
        st.subheader("📈 Return Metrics")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric(f"{ticker} Total Return", f"{additional_metrics['total_return']:.2f}%")
        with col2:
            st.metric(f"{benchmark} Total Return", f"{additional_metrics['benchmark_return']:.2f}%")
        with col3:
            st.metric("Excess Return", f"{additional_metrics['excess_return']:.2f}%")
        with col4:
            st.metric("Sharpe Ratio", f"{additional_metrics['sharpe_ratio']:.2f}")
        
        # Correlation Analysis
        st.subheader("🔗 Correlation Analysis")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Overall Correlation", f"{additional_metrics['overall_correlation']:.3f}")
        with col2:
            st.metric("Recent Correlation (30-day)", f"{additional_metrics['recent_correlation']:.3f}")
        with col3:
            corr_interp = "Highly correlated with market" if additional_metrics['overall_correlation'] > 0.7 else \
                          "Moderately correlated with market" if additional_metrics['overall_correlation'] > 0.4 else \
                          "Low positive correlation with market" if additional_metrics['overall_correlation'] > 0 else \
                          "Negative correlation with market"
            st.metric("Interpretation", corr_interp)
        
        # Trading Days Analysis
        st.subheader("📊 Trading Days Analysis")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Trading Days", f"{len(stock_returns):,}")
        with col2:
            st.metric("Positive Days", f"{additional_metrics['positive_days']:,} ({additional_metrics['win_rate']:.1f}%)")
        with col3:
            st.metric("Negative Days", f"{additional_metrics['negative_days']:,} ({100-additional_metrics['win_rate']:.1f}%)")
        
        # Extreme Moves
        st.subheader("📉 Extreme Moves")
        col1, col2 = st.columns(2)
        with col1:
            st.metric(
                "Best Day",
                f"{additional_metrics['best_day_return']:.2f}%",
                delta=f"on {additional_metrics['best_day_date'].strftime('%Y-%m-%d')}"
            )
        with col2:
            st.metric(
                "Worst Day",
                f"{additional_metrics['worst_day_return']:.2f}%",
                delta=f"on {additional_metrics['worst_day_date'].strftime('%Y-%m-%d')}",
                delta_color="inverse"
            )
        
        # Additional Risk-Adjusted Metrics
        st.subheader("📊 Additional Risk Metrics")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Annualized Mean Return", f"{additional_metrics['mean_return']*100:.2f}%")
        with col2:
            st.metric("Average Daily Return", f"{additional_metrics['avg_daily_return']:.3f}%")
        with col3:
            st.metric("Volatility", f"{volatility*100:.2f}%")
        
        st.divider()
        
        # Summary Section
        st.header("📋 Summary")
        
        summary_text = f"""
        **Position Analysis for {ticker}**
        
        - **Current Position Value**: ${shares * current_price:,.2f}
        - **Risk Level**: {'High' if volatility > 0.40 or beta > 1.5 else 'Moderate' if volatility > 0.25 or beta > 1.2 else 'Low'}
        - **Estimated After-Tax Proceeds**: ${tax_results['after_tax_proceeds']:,.2f}
        - **Tax Impact**: ${tax_results['tax_estimate']:,.2f} ({tax_rate*100:.0f}% of gains)
        """
        
        st.markdown(summary_text)
        
        # Download button for results
        st.divider()
        st.markdown("### 💾 Export Results")
        
        results_text = f"""
CONCENTRATED POSITION ANALYSIS: {ticker}
{'='*70}

CURRENT PRICE: ${current_price:.2f}

RISK METRICS
{'-'*70}
Volatility (Annualized):     {volatility*100:.2f}%
Beta vs. {benchmark}:                {beta:.2f}
Maximum Drawdown:            {max_drawdown*100:.2f}%
Max Drawdown Period:         {peak_date.strftime('%Y-%m-%d')} to {trough_date.strftime('%Y-%m-%d')}

LIQUIDITY & TAX SIMULATION
{'-'*70}
Shares to Sell:              {shares:,.0f}
Estimated Proceeds:           ${tax_results['proceeds']:,.2f}
Total Cost Basis:            ${tax_results['total_cost_basis']:,.2f}
Estimated Gain:              ${tax_results['gain']:,.2f}
Tax Estimate:                ${tax_results['tax_estimate']:,.2f}
After-Tax Proceeds:          ${tax_results['after_tax_proceeds']:,.2f}
"""
        
        st.download_button(
            label="📥 Download Analysis Report",
            data=results_text,
            file_name=f"{ticker}_analysis_report.txt",
            mime="text/plain"
        )

if __name__ == "__main__":
    main()

