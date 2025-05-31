import pandas as pd
import pytest
from unittest.mock import patch, MagicMock # Using unittest.mock directly
from src.core.data_manager import fetch_spy_data # Assuming src is in PYTHONPATH or tests are run in a way that finds src

# No longer using a separate fixture for mock_yf_ticker if we apply patch per test method
# If needed globally, it would be defined differently without mocker

@patch('yfinance.Ticker') # Patch the Ticker class in yfinance module
def test_fetch_spy_data_returns_dataframe(mock_yf_ticker_constructor):
    """Test that fetch_spy_data returns a pandas DataFrame."""
    # Configure the mock returned by yfinance.Ticker("SPY")
    mock_ticker_instance = MagicMock()
    mock_history_data = pd.DataFrame({
        'Open': [100, 101, 102],
        'High': [105, 106, 107],
        'Low': [99, 100, 101],
        'Close': [102, 103, 104],
        'Volume': [1000, 1100, 1200],
        'Dividends': [0, 0, 0], # yfinance history includes these
        'Stock Splits': [0, 0, 0] # and these
    }, index=pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03']))
    mock_ticker_instance.history.return_value = mock_history_data
    mock_yf_ticker_constructor.return_value = mock_ticker_instance # Ticker("SPY") returns our mock_ticker_instance

    period = "3d"
    data = fetch_spy_data(ticker_symbol="SPY", period=period, use_cache=False)

    assert isinstance(data, pd.DataFrame)
    mock_yf_ticker_constructor.assert_called_once_with("SPY")
    mock_ticker_instance.history.assert_called_once_with(period=period, interval="1d")

@patch('yfinance.Ticker')
def test_fetch_spy_data_dataframe_not_empty(mock_yf_ticker_constructor):
    """Test that the DataFrame returned by fetch_spy_data is not empty."""
    mock_ticker_instance = MagicMock()
    mock_history_data = pd.DataFrame({
        'Open': [100], 'High': [105], 'Low': [99], 'Close': [102], 'Volume': [1000],
        'Dividends': [0], 'Stock Splits': [0]
    }, index=pd.to_datetime(['2023-01-01']))
    mock_ticker_instance.history.return_value = mock_history_data
    mock_yf_ticker_constructor.return_value = mock_ticker_instance

    period = "1d"
    data = fetch_spy_data(ticker_symbol="SPY", period=period, use_cache=False)
    assert not data.empty

@patch('yfinance.Ticker')
def test_fetch_spy_data_contains_expected_columns(mock_yf_ticker_constructor):
    """Test that the DataFrame contains the expected OHLCV columns after processing."""
    mock_ticker_instance = MagicMock()
    mock_history_data = pd.DataFrame({ # Raw data from yfinance might have different casing
        'open': [100], 'high': [105], 'low': [99], 'close': [102], 'volume': [1000],
        'dividends': [0], 'stock splits': [0], 'adj close': [102]
    }, index=pd.to_datetime(['2023-01-01']))
    mock_ticker_instance.history.return_value = mock_history_data
    mock_yf_ticker_constructor.return_value = mock_ticker_instance

    period = "1d"
    data = fetch_spy_data(ticker_symbol="SPY", period=period, use_cache=False)

    expected_columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close'] # After processing
    assert all(col in data.columns for col in expected_columns)

@patch('yfinance.Ticker')
def test_fetch_spy_data_handles_yf_failure(mock_yf_ticker_constructor):
    """Test how fetch_spy_data handles an empty DataFrame from yfinance."""
    mock_ticker_instance = MagicMock()
    mock_ticker_instance.history.return_value = pd.DataFrame() # Simulate yfinance returning empty DF
    mock_yf_ticker_constructor.return_value = mock_ticker_instance

    period = "3d"
    data = fetch_spy_data(ticker_symbol="SPY", period=period, use_cache=False)
    assert data is None # Expect None when yfinance returns empty or critical processing fails

if __name__ == '__main__':
    # This allows running tests with `python tests/core/test_data_manager.py`
    # However, `pytest` is the recommended way.
    pytest.main([__file__])
