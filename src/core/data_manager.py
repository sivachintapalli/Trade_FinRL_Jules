import yfinance as yf
import pandas as pd

def get_spy_data(start_date: str, end_date: str) -> pd.DataFrame:
    """
    Fetches SPY (S&P 500 ETF) data from Yahoo Finance.

    Args:
        start_date: The start date for the data retrieval (YYYY-MM-DD).
        end_date: The end date for the data retrieval (YYYY-MM-DD).

    Returns:
        A Pandas DataFrame containing OHLCV data for SPY.
    """
    spy = yf.Ticker("SPY")
    data = spy.history(start=start_date, end=end_date)
    return data

if __name__ == '__main__':
    # Example usage:
    start = "2023-01-01"
    end = "2024-01-01"
    spy_data = get_spy_data(start, end)
    if not spy_data.empty:
        print("SPY Data Fetched Successfully:")
        print(spy_data.head())
    else:
        print("No data fetched. Check your dates or ticker symbol.")
