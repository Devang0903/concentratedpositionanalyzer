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
import requests
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
BENCHMARK = 'SPY'

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

def fetch_data(ticker, benchmark, years):
    """Fetch historical data for ticker and benchmark."""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=years * 365)
    
    try:
        # Use session for better connection handling (like analyzer.py)
        session = requests.Session()
        stock = yf.Ticker(ticker, session=session)
        stock_data = stock.history(start=start_date, end=end_date)
        
        bench = yf.Ticker(benchmark, session=session)
        benchmark_data = bench.history(start=start_date, end=end_date)
        
        if stock_data.empty or benchmark_data.empty:
            return None, None, "No data found for one or both tickers."
        
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
        return None, None, None, None, None, "Insufficient data for calculations"
    
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
    
    return volatility, beta, max_drawdown, peak_date, trough_date, None

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
        
        # Show loading spinner
        with st.spinner(f"Fetching data for {ticker} and {BENCHMARK}..."):
            stock_data, benchmark_data, error = fetch_data(ticker, BENCHMARK, YEARS_OF_DATA)
        
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
            volatility, beta, max_drawdown, peak_date, trough_date, calc_error = calculate_risk_metrics(
                stock_prices, benchmark_prices
            )
        
        if calc_error:
            st.error(f"❌ {calc_error}")
            return
        
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
                "Beta vs. SPY",
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
            fig = create_price_chart(stock_prices, benchmark_prices, ticker, BENCHMARK)
            st.pyplot(fig)
        
        st.caption(f"Chart shows normalized price performance (starting at 100) for {ticker} vs {BENCHMARK} over the last {YEARS_OF_DATA} years")
        
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
Beta vs. SPY:                {beta:.2f}
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

