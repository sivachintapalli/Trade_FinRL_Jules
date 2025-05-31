import yfinance as yf
import pandas as pd
import os
from datetime import datetime, timedelta

CACHE_DIR = "data/cache"
CACHE_FILE = os.path.join(CACHE_DIR, "SPY_daily.csv")

def fetch_spy_data(ticker_symbol="SPY", period="5y", interval="1d", use_cache=True):
    """
    Fetches historical stock data for a given ticker symbol, using a local cache.

    Args:
        ticker_symbol (str): The stock ticker symbol (default: "SPY").
        period (str): The period for which to download data (e.g., "1y", "5y", "max").
        interval (str): The interval of data points (e.g., "1d", "1wk", "1mo").
        use_cache (bool): Whether to use the caching mechanism.

    Returns:
        pandas.DataFrame: DataFrame containing the stock data, or None if fetching fails.
    """
    if not os.path.exists(CACHE_DIR):
        try:
            os.makedirs(CACHE_DIR)
        except OSError as e:
            print(f"Error creating cache directory {CACHE_DIR}: {e}")
            return None

    if use_cache and os.path.exists(CACHE_FILE):
        try:
            # Check cache file modification time
            mod_time = os.path.getmtime(CACHE_FILE)
            if datetime.now() - datetime.fromtimestamp(mod_time) < timedelta(days=1):
                print(f"Loading SPY data from cache: {CACHE_FILE}")
                return pd.read_csv(CACHE_FILE, index_col='Date', parse_dates=True)
            else:
                print("Cache is older than 1 day. Fetching fresh data.")
        except Exception as e:
            print(f"Error reading from cache or checking mod time: {e}. Fetching fresh data.")

    print(f"Fetching fresh data for {ticker_symbol}...")
    try:
        stock = yf.Ticker(ticker_symbol)
        data = stock.history(period=period, interval=interval)

        if data.empty:
            print(f"No data found for {ticker_symbol} for the period {period}.")
            return None

        # Ensure the index is DatetimeIndex and named 'Date'
        if not isinstance(data.index, pd.DatetimeIndex):
            data.index = pd.to_datetime(data.index)
        data.index.name = 'Date'

        # Standardize column names to TitleCase, paying attention to 'Volume'
        new_columns = {}
        for col in data.columns:
            if col.lower() == 'adj close': # yfinance often includes 'Adj Close'
                new_columns[col] = 'Adj Close'
            else:
                new_columns[col] = col.title()
        data.rename(columns=new_columns, inplace=True)
        # Ensure 'Volume' is exactly 'Volume' if it exists
        if 'volume' in data.columns and 'Volume' not in data.columns:
             data.rename(columns={'volume': 'Volume'}, inplace=True)
        elif 'Volume' not in data.columns and any(c.lower() == 'volume' for c in data.columns):
            # find the actual casing yfinance used for volume if it wasn't 'volume' or 'Volume'
            for c in data.columns:
                if c.lower() == 'volume':
                    data.rename(columns={c: 'Volume'}, inplace=True)
                    break

        # Verify data types (mostly handled by yfinance, but good to ensure)
        # Ensure OHLCV are numeric
        cols_to_check_numeric = ['Open', 'High', 'Low', 'Close']
        # Add Volume, handling case variations
        if 'Volume' in data.columns:
            cols_to_check_numeric.append('Volume')
        elif 'volume' in data.columns: # This case should ideally be handled by the rename above
            cols_to_check_numeric.append('volume')
            # Standardize to 'Volume'
            data.rename(columns={'volume': 'Volume'}, inplace=True) # Ensure it's 'Volume' for the numeric check

        for col in cols_to_check_numeric:
            if col in data.columns: # Check if column exists before trying to convert
                data[col] = pd.to_numeric(data[col], errors='coerce')

        # Handle missing data - forward fill
        initial_nans = data.isna().sum().sum()
        if initial_nans > 0:
            print(f"Found {initial_nans} NaN values. Applying forward fill.")
            data.ffill(inplace=True)
            # If NaNs are still present at the beginning (after ffill), backfill them
            if data.iloc[0].isna().sum() > 0:
                print("NaNs found at the beginning after ffill. Applying backfill for initial rows.")
                data.bfill(inplace=True)

        # Drop any rows that still have NaNs after ffill/bfill (e.g., if entire column was NaN initially)
        rows_before_dropna = len(data)
        data.dropna(inplace=True)
        if len(data) < rows_before_dropna:
            print(f"Dropped {rows_before_dropna - len(data)} rows due to remaining NaNs after fill.")

        if use_cache:
            try:
                data.to_csv(CACHE_FILE)
                print(f"Data cached to {CACHE_FILE}")
            except Exception as e:
                print(f"Error caching data to {CACHE_FILE}: {e}")

        return data
    except Exception as e:
        print(f"Error fetching data for {ticker_symbol} using yfinance: {e}")
        return None

if __name__ == '__main__':
    print("--- Testing Data Manager ---")

    print("\n--- Attempt 1: Fetching SPY data (potentially using cache) ---")
    spy_data_c1 = fetch_spy_data(ticker_symbol="SPY", period="5y")
    if spy_data_c1 is not None:
        print("\nSPY Data (First Fetch) Head:")
        print(spy_data_c1.head())
        print("\nSPY Data (First Fetch) Tail:")
        print(spy_data_c1.tail())
        print(f"\nData dimensions: {spy_data_c1.shape}")
        print(f"\nData types:\n{spy_data_c1.dtypes}")
        print(f"\nIndex type: {type(spy_data_c1.index)}")
        print(f"\nIndex name: {spy_data_c1.index.name}")
        print(f"\nAny remaining NaNs: {spy_data_c1.isna().sum().sum()}")

    print("\n--- Attempt 2: Fetching SPY data again (should use cache if first attempt was successful) ---")
    spy_data_c2 = fetch_spy_data(ticker_symbol="SPY", period="5y")
    if spy_data_c2 is not None:
        print("\nSPY Data (Second Fetch) Head:")
        print(spy_data_c2.head(2)) # Just show a couple of lines to confirm it loaded
        print(f"\nData dimensions: {spy_data_c2.shape}")
        print(f"\nAny remaining NaNs: {spy_data_c2.isna().sum().sum()}")
        if spy_data_c1 is not None and spy_data_c2.equals(spy_data_c1):
            print("\nData from second fetch is identical to the first (as expected from cache).")
        else:
            print("\nData from second fetch differs or first fetch failed.")

    print("\n--- Attempt 3: Fetching 1 month of SPY data without cache ---")
    spy_data_no_cache = fetch_spy_data(ticker_symbol="SPY", period="1mo", use_cache=False)
    if spy_data_no_cache is not None:
        print("\nSPY Data (No Cache - 1mo) Head:")
        print(spy_data_no_cache.head())
        print(f"\nData dimensions: {spy_data_no_cache.shape}")
        print(f"\nAny remaining NaNs: {spy_data_no_cache.isna().sum().sum()}")

    print("\n--- Testing with a non-existent ticker ---")
    non_existent_data = fetch_spy_data(ticker_symbol="NONEXISTENTTICKER", period="1mo")
    if non_existent_data is None:
        print("Correctly returned None for a non-existent ticker.")
    else:
        print(f"Unexpectedly received data for non-existent ticker: {non_existent_data.head()}")

    print("\n--- Data Manager Test Complete ---")
