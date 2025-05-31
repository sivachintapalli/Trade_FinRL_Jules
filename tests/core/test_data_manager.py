import pandas as pd
import pytest
from unittest.mock import patch # Using unittest.mock directly as pytest-mock provides it via fixture
from src.core.data_manager import get_spy_data # Assuming src is in PYTHONPATH or tests are run in a way that finds src

@pytest.fixture
def mock_yf_ticker(mocker):
    """Mocks yfinance.Ticker and its history method."""
    mock_ticker = mocker.MagicMock()
    mock_history_data = pd.DataFrame({
        'Open': [100, 101, 102],
        'High': [105, 106, 107],
        'Low': [99, 100, 101],
        'Close': [102, 103, 104],
        'Volume': [1000, 1100, 1200],
        'Dividends': [0, 0, 0],
        'Stock Splits': [0, 0, 0]
    })
    mock_history_data.index = pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03'])
    mock_ticker.history.return_value = mock_history_data
    return mocker.patch('yfinance.Ticker', return_value=mock_ticker)

def test_get_spy_data_returns_dataframe(mock_yf_ticker):
    """Test that get_spy_data returns a pandas DataFrame."""
    start_date = "2023-01-01"
    end_date = "2023-01-03"
    data = get_spy_data(start_date, end_date)
    assert isinstance(data, pd.DataFrame)
    mock_yf_ticker.assert_called_once_with("SPY")
    mock_yf_ticker.return_value.history.assert_called_once_with(start=start_date, end=end_date)

def test_get_spy_data_dataframe_not_empty(mock_yf_ticker):
    """Test that the DataFrame returned by get_spy_data is not empty."""
    start_date = "2023-01-01"
    end_date = "2023-01-03"
    data = get_spy_data(start_date, end_date)
    assert not data.empty

def test_get_spy_data_contains_expected_columns(mock_yf_ticker):
    """Test that the DataFrame contains the expected OHLCV columns."""
    start_date = "2023-01-01"
    end_date = "2023-01-03"
    data = get_spy_data(start_date, end_date)
    expected_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    assert all(col in data.columns for col in expected_columns)

def test_get_spy_data_handles_yf_failure(mocker):
    """Test how get_spy_data handles an empty DataFrame from yfinance (e.g., bad date range)."""
    mock_ticker = mocker.MagicMock()
    mock_ticker.history.return_value = pd.DataFrame() # Simulate yfinance returning empty DF
    mocker.patch('yfinance.Ticker', return_value=mock_ticker)

    start_date = "2023-01-01"
    end_date = "2023-01-03" # Or a date range that would yield no data
    data = get_spy_data(start_date, end_date)
    assert isinstance(data, pd.DataFrame)
    assert data.empty # Expect an empty DataFrame in this scenario

if __name__ == '__main__':
    # This allows running tests with `python tests/core/test_data_manager.py`
    # However, `pytest` is the recommended way.
    pytest.main([__file__])
