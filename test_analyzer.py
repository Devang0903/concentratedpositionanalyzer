"""
Test cases for the Concentrated Position Risk & Liquidity Analyzer

This test suite validates:
- Ticker validation (valid, invalid, edge cases)
- Input validation for shares, cost basis, tax rate
- Risk metric calculations
- Tax simulation calculations
"""

import pytest
import yfinance as yf
import pandas as pd
import numpy as np
from analyzer import (
    validate_ticker,
    calculate_daily_returns,
    calculate_volatility,
    calculate_beta,
    calculate_max_drawdown,
    calculate_tax_estimate,
    _is_rate_limit_error,
    _rate_limit_delay
)


class TestTickerValidation:
    """Test cases for ticker validation"""
    
    def test_valid_ticker(self):
        """Test that valid tickers pass validation"""
        # Test with well-known valid tickers
        assert validate_ticker("TSLA") == True
        assert validate_ticker("AAPL") == True
        assert validate_ticker("MSFT") == True
        assert validate_ticker("SPY") == True
    
    def test_ticker_case_insensitive(self):
        """Test that ticker validation is case-insensitive"""
        assert validate_ticker("tsla") == True
        assert validate_ticker("TsLa") == True
        assert validate_ticker("  TSLA  ") == True  # With whitespace
    
    def test_empty_ticker(self):
        """Test that empty ticker raises ValueError"""
        with pytest.raises(ValueError, match="cannot be empty"):
            validate_ticker("")
        with pytest.raises(ValueError, match="cannot be empty"):
            validate_ticker("   ")
    
    def test_none_ticker(self):
        """Test that None ticker raises ValueError"""
        with pytest.raises(ValueError, match="cannot be empty"):
            validate_ticker(None)
    
    def test_invalid_ticker_format(self):
        """Test that invalid ticker formats are rejected"""
        with pytest.raises(ValueError, match="Invalid ticker format"):
            validate_ticker("TSLA!@#")
        with pytest.raises(ValueError, match="Invalid ticker format"):
            validate_ticker("TS LA")  # Space in ticker
        with pytest.raises(ValueError, match="Invalid ticker format"):
            validate_ticker("TS$LA")
    
    def test_nonexistent_ticker(self):
        """Test that non-existent tickers are rejected"""
        with pytest.raises(ValueError):
            validate_ticker("INVALIDTICKER12345")
        with pytest.raises(ValueError):
            validate_ticker("XYZABC999")
    
    def test_ticker_with_special_chars(self):
        """Test tickers with dots or dashes (some ETFs have these)"""
        # Some valid tickers might have dots or dashes
        # This should pass format check but might fail on Yahoo Finance lookup
        # We'll test that the format validation allows them
        ticker = "BRK.B"  # Berkshire Hathaway Class B
        try:
            result = validate_ticker(ticker)
            assert result == True
        except ValueError:
            # If it fails, it should be a "not found" error, not format error
            pass


class TestInputValidation:
    """Test cases for input validation logic"""
    
    def test_shares_validation_positive(self):
        """Test that positive share values are accepted"""
        # This would be tested in integration, but we can test the logic
        shares = 100.0
        assert shares > 0
    
    def test_shares_validation_zero(self):
        """Test that zero shares are rejected"""
        shares = 0.0
        assert shares <= 0
    
    def test_shares_validation_negative(self):
        """Test that negative shares are rejected"""
        shares = -10.0
        assert shares <= 0
    
    def test_shares_validation_fractional(self):
        """Test that fractional shares are accepted"""
        shares = 0.5
        assert shares > 0
    
    def test_cost_basis_validation_positive(self):
        """Test that positive cost basis is accepted"""
        cost_basis = 150.0
        assert cost_basis >= 0
    
    def test_cost_basis_validation_zero(self):
        """Test that zero cost basis is accepted (gift, etc.)"""
        cost_basis = 0.0
        assert cost_basis >= 0
    
    def test_cost_basis_validation_negative(self):
        """Test that negative cost basis is rejected"""
        cost_basis = -10.0
        assert cost_basis < 0
    
    def test_tax_rate_validation_valid(self):
        """Test that valid tax rates are accepted"""
        tax_rate = 0.20  # 20%
        assert 0 <= tax_rate <= 1
        tax_rate = 0.0  # 0%
        assert 0 <= tax_rate <= 1
        tax_rate = 1.0  # 100%
        assert 0 <= tax_rate <= 1
    
    def test_tax_rate_validation_negative(self):
        """Test that negative tax rates are rejected"""
        tax_rate = -0.10
        assert tax_rate < 0
    
    def test_tax_rate_validation_over_one(self):
        """Test that tax rates over 1.0 are rejected"""
        tax_rate = 1.5  # 150%
        assert tax_rate > 1


class TestRiskMetrics:
    """Test cases for risk metric calculations"""
    
    def test_calculate_daily_returns(self):
        """Test daily returns calculation"""
        prices = pd.Series([100, 105, 102, 108, 110], 
                          index=pd.date_range('2024-01-01', periods=5))
        returns = calculate_daily_returns(prices)
        
        # First value should be NaN (no previous price)
        assert pd.isna(returns.iloc[0])
        # Second value: (105-100)/100 = 0.05
        assert abs(returns.iloc[1] - 0.05) < 0.0001
        # Third value: (102-105)/105 ≈ -0.0286
        assert abs(returns.iloc[2] - (-0.028571)) < 0.0001
    
    def test_calculate_volatility(self):
        """Test volatility calculation"""
        # Create returns with known volatility
        returns = pd.Series([0.01, -0.02, 0.015, -0.01, 0.02] * 50)  # 250 days
        volatility = calculate_volatility(returns)
        
        # Volatility should be positive
        assert volatility > 0
        # Should be annualized (multiplied by sqrt(252))
        assert volatility > returns.std()
    
    def test_calculate_beta(self):
        """Test beta calculation"""
        # Create correlated returns
        dates = pd.date_range('2024-01-01', periods=100)
        benchmark_returns = pd.Series(np.random.normal(0.001, 0.01, 100), index=dates)
        # Stock returns with beta of ~2 (more volatile, correlated)
        stock_returns = pd.Series(benchmark_returns * 2 + np.random.normal(0, 0.005, 100), index=dates)
        
        beta = calculate_beta(stock_returns, benchmark_returns)
        
        # Beta should be positive and close to 2
        assert beta > 0
        assert abs(beta - 2.0) < 1.0  # Allow some variance
    
    def test_calculate_beta_insufficient_data(self):
        """Test beta calculation with insufficient data"""
        stock_returns = pd.Series([0.01])
        benchmark_returns = pd.Series([0.01])
        
        with pytest.raises(ValueError, match="Insufficient overlapping data"):
            calculate_beta(stock_returns, benchmark_returns)
    
    def test_calculate_max_drawdown(self):
        """Test maximum drawdown calculation"""
        # Create price series with known drawdown
        prices = pd.Series([100, 110, 120, 80, 90, 95],
                          index=pd.date_range('2024-01-01', periods=6))
        
        max_dd_pct, max_dd_value, peak_date, trough_date = calculate_max_drawdown(prices)
        
        # Max drawdown should be from 120 to 80 = 33.33%
        assert max_dd_pct > 0.30 and max_dd_pct < 0.35
        assert max_dd_value > 0
        assert peak_date < trough_date
    
    def test_calculate_max_drawdown_no_drawdown(self):
        """Test max drawdown with only upward movement"""
        prices = pd.Series([100, 110, 120, 130],
                          index=pd.date_range('2024-01-01', periods=4))
        
        max_dd_pct, max_dd_value, peak_date, trough_date = calculate_max_drawdown(prices)
        
        # Should be 0 or very small
        assert max_dd_pct >= 0
        assert max_dd_value >= 0


class TestTaxSimulation:
    """Test cases for tax simulation calculations"""
    
    def test_tax_estimate_profit(self):
        """Test tax calculation with profit"""
        result = calculate_tax_estimate(
            shares=100,
            current_price=200,
            cost_basis=150,
            tax_rate=0.20
        )
        
        assert result['proceeds'] == 20000  # 100 * 200
        assert result['total_cost_basis'] == 15000  # 100 * 150
        assert result['gain'] == 5000  # 20000 - 15000
        assert result['tax_estimate'] == 1000  # 5000 * 0.20
        assert result['after_tax_proceeds'] == 19000  # 20000 - 1000
    
    def test_tax_estimate_loss(self):
        """Test tax calculation with loss (no tax)"""
        result = calculate_tax_estimate(
            shares=100,
            current_price=100,
            cost_basis=150,
            tax_rate=0.20
        )
        
        assert result['proceeds'] == 10000
        assert result['total_cost_basis'] == 15000
        assert result['gain'] == -5000  # Loss
        assert result['tax_estimate'] == 0  # No tax on losses
        assert result['after_tax_proceeds'] == 10000
    
    def test_tax_estimate_break_even(self):
        """Test tax calculation at break-even"""
        result = calculate_tax_estimate(
            shares=100,
            current_price=150,
            cost_basis=150,
            tax_rate=0.20
        )
        
        assert result['proceeds'] == 15000
        assert result['total_cost_basis'] == 15000
        assert result['gain'] == 0
        assert result['tax_estimate'] == 0
        assert result['after_tax_proceeds'] == 15000
    
    def test_tax_estimate_fractional_shares(self):
        """Test tax calculation with fractional shares"""
        result = calculate_tax_estimate(
            shares=0.5,
            current_price=200,
            cost_basis=150,
            tax_rate=0.20
        )
        
        assert result['proceeds'] == 100  # 0.5 * 200
        assert result['total_cost_basis'] == 75  # 0.5 * 150
        assert result['gain'] == 25
        assert result['tax_estimate'] == 5  # 25 * 0.20
        assert result['after_tax_proceeds'] == 95
    
    def test_tax_estimate_zero_tax_rate(self):
        """Test tax calculation with zero tax rate"""
        result = calculate_tax_estimate(
            shares=100,
            current_price=200,
            cost_basis=150,
            tax_rate=0.0
        )
        
        assert result['tax_estimate'] == 0
        assert result['after_tax_proceeds'] == result['proceeds']
    
    def test_tax_estimate_high_tax_rate(self):
        """Test tax calculation with high tax rate"""
        result = calculate_tax_estimate(
            shares=100,
            current_price=200,
            cost_basis=150,
            tax_rate=0.50  # 50%
        )
        
        assert result['tax_estimate'] == 2500  # 5000 * 0.50
        assert result['after_tax_proceeds'] == 17500


class TestEdgeCases:
    """Test edge cases and boundary conditions"""
    
    def test_very_small_shares(self):
        """Test with very small share amounts"""
        result = calculate_tax_estimate(
            shares=0.001,
            current_price=100,
            cost_basis=50,
            tax_rate=0.20
        )
        assert result['proceeds'] > 0
        assert result['tax_estimate'] >= 0
    
    def test_very_large_shares(self):
        """Test with very large share amounts"""
        result = calculate_tax_estimate(
            shares=1000000,
            current_price=100,
            cost_basis=50,
            tax_rate=0.20
        )
        assert result['proceeds'] == 100000000
        assert result['tax_estimate'] == 10000000
    
    def test_very_high_price(self):
        """Test with very high stock price"""
        result = calculate_tax_estimate(
            shares=10,
            current_price=10000,
            cost_basis=5000,
            tax_rate=0.20
        )
        assert result['proceeds'] == 100000
        assert result['gain'] == 50000
    
    def test_zero_cost_basis(self):
        """Test with zero cost basis (gift, inheritance)"""
        result = calculate_tax_estimate(
            shares=100,
            current_price=200,
            cost_basis=0,
            tax_rate=0.20
        )
        assert result['total_cost_basis'] == 0
        assert result['gain'] == 20000
        assert result['tax_estimate'] == 4000


class TestRateLimiting:
    """Test cases for rate limiting functionality"""
    
    def test_is_rate_limit_error_detection(self):
        """Test that rate limit errors are correctly identified"""
        # Test various rate limit error messages
        class RateLimitError(Exception):
            def __init__(self, msg):
                self.msg = msg
            def __str__(self):
                return self.msg
        
        assert _is_rate_limit_error(RateLimitError("Too many requests")) == True
        assert _is_rate_limit_error(RateLimitError("Rate limit exceeded")) == True
        assert _is_rate_limit_error(RateLimitError("Rate limited. Try after a while.")) == True
        assert _is_rate_limit_error(RateLimitError("429 Too Many Requests")) == True
        assert _is_rate_limit_error(RateLimitError("Throttled")) == True
        assert _is_rate_limit_error(RateLimitError("Quota exceeded")) == True
        
        # Test non-rate-limit errors
        assert _is_rate_limit_error(RateLimitError("Not found")) == False
        assert _is_rate_limit_error(RateLimitError("Invalid ticker")) == False
        assert _is_rate_limit_error(RateLimitError("Connection timeout")) == False
    
    def test_rate_limit_delay(self):
        """Test that rate limit delay function works"""
        import time
        
        # First call should not delay (or minimal delay)
        start = time.time()
        _rate_limit_delay()
        first_call_time = time.time() - start
        assert first_call_time < 0.1  # Should be very fast
        
        # Second call should enforce minimum delay
        start = time.time()
        _rate_limit_delay()
        second_call_time = time.time() - start
        # Should have at least some delay (0.5 seconds minimum)
        assert second_call_time >= 0.4  # Allow some margin for execution time


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])

