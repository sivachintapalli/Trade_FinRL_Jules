import yfinance as yf
import pandas as pd
import os
from datetime import datetime, timedelta

CACHE_DIR = "data/cache"

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

    cache_file_path = os.path.join(CACHE_DIR, f"{ticker_symbol.upper()}_daily.csv")

    data = None # Initialize data to None

    if use_cache and os.path.exists(cache_file_path):
        try:
            mod_time = os.path.getmtime(cache_file_path)
            if datetime.now() - datetime.fromtimestamp(mod_time) < timedelta(days=1):
                print(f"Loading {ticker_symbol.upper()} data from cache: {cache_file_path}")
                data = pd.read_csv(cache_file_path, index_col='Date', parse_dates=True)
            else:
                print(f"Cache for {ticker_symbol.upper()} is older than 1 day. Fetching fresh data.")
        except Exception as e:
            print(f"Error reading from cache or checking mod time for {ticker_symbol.upper()}: {e}. Fetching fresh data.")

    was_fetched_freshly = False
    if data is None: # If cache was not used, not valid, or failed to load
        print(f"Fetching fresh data for {ticker_symbol.upper()}...")
        try:
            stock = yf.Ticker(ticker_symbol)
            fetched_data = stock.history(period=period, interval=interval)

            if fetched_data.empty:
                print(f"No data found for {ticker_symbol.upper()} for the period {period}.")
                return None
            data = fetched_data
            was_fetched_freshly = True
        except Exception as e:
            print(f"Error fetching data for {ticker_symbol.upper()} using yfinance: {e}")
            return None

    if data is None: # Should only happen if initial cache load failed AND fetch failed
        print(f"Failed to load or fetch data for {ticker_symbol.upper()}.")
        return None

    # ==== COMMON DATA PROCESSING ====
    # Applied to data whether loaded from cache or freshly fetched.
    try:
        # 1. Index processing: Ensure DatetimeIndex, UTC, and named 'Date'
        if not isinstance(data.index, pd.DatetimeIndex):
            data.index = pd.to_datetime(data.index, utc=True)
        elif data.index.tz is None:
            data.index = data.index.tz_localize('UTC')
        else:
            data.index = data.index.tz_convert('UTC')
        data.index.name = 'Date'

        # 2. Standardize column names
        new_columns = {}
        for col in data.columns: # Iterate over current columns
            if col.lower() == 'adj close':
                new_columns[col] = 'Adj Close' # Preserve 'Adj Close' casing from yfinance
            else:
                new_columns[col] = col.title() # Title case others
        data.rename(columns=new_columns, inplace=True)

        # Ensure 'Volume' is exactly 'Volume' if it exists in any case
        if 'volume' in data.columns and 'Volume' not in data.columns:
            data.rename(columns={'volume': 'Volume'}, inplace=True)
        elif 'Volume' not in data.columns: # Check if other cases of 'volume' exist
            for col_name in list(data.columns): # Iterate over a copy for renaming
                if col_name.lower() == 'volume':
                    data.rename(columns={col_name: 'Volume'}, inplace=True)
                    break

        # 3. Convert OHLCV columns to numeric
        cols_to_numeric = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
        for col in cols_to_numeric:
            if col in data.columns:
                data[col] = pd.to_numeric(data[col], errors='coerce')

        # 4. Round float columns to 5 decimal places for consistency
        float_cols = ['Open', 'High', 'Low', 'Close', 'Adj Close']
        for col in float_cols:
            if col in data.columns and data[col].dtype == 'float64':
                data[col] = data[col].round(5)

        # 5. Handle missing data (NaNs)
        if data.isna().any().any():
            # print(f"NaN values found in data for {ticker_symbol.upper()}. Applying ffill then bfill.")
            data.ffill(inplace=True)
            data.bfill(inplace=True)

        # 6. Drop rows if critical columns still have NaNs
        critical_cols = ['Open', 'High', 'Low', 'Close']
        # Ensure critical_cols only contains columns present in the DataFrame
        cols_to_check_for_nan_drop = [col for col in critical_cols if col in data.columns]
        if cols_to_check_for_nan_drop:
            rows_before_dropna = len(data)
            data.dropna(subset=cols_to_check_for_nan_drop, inplace=True)
            if len(data) < rows_before_dropna:
                print(f"Dropped {rows_before_dropna - len(data)} rows with NaNs in critical columns for {ticker_symbol.upper()}.")

    except Exception as e_process:
        print(f"Error during common data processing for {ticker_symbol.upper()}: {e_process}")
        return None
    # ==== END COMMON DATA PROCESSING ====

    # Cache the fully processed data if it was freshly fetched
    if was_fetched_freshly and use_cache:
        try:
            data.to_csv(cache_file_path)
            print(f"Data cached to {cache_file_path}")
        except Exception as e:
            print(f"Error caching fully processed data to {cache_file_path}: {e}")

    return data

if __name__ == '__main__':
    print("--- Testing Data Manager ---")

    # Test with AAPL
    print("\n--- Attempt 1: Fetching AAPL data (first time, should fetch) ---")
    aapl_data_c1 = fetch_spy_data(ticker_symbol="AAPL", period="1y")
    if aapl_data_c1 is not None:
        print("\nAAPL Data (First Fetch) Head:")
        print(aapl_data_c1.head())
        print(f"Data dimensions: {aapl_data_c1.shape}")

    print("\n--- Attempt 2: Fetching AAPL data again (should use AAPL cache) ---")
    aapl_data_c2 = fetch_spy_data(ticker_symbol="AAPL", period="1y")
    if aapl_data_c2 is not None:
        print("\nAAPL Data (Second Fetch) Head:")
        print(aapl_data_c2.head(2))
        if aapl_data_c1 is not None and aapl_data_c2.equals(aapl_data_c1):
            print("AAPL data from second fetch is identical to the first (cached).")
        else:
            print("AAPL data from second fetch differs or first fetch failed.")

    # Test with SPY to ensure it still works and uses its own cache
    print("\n--- Attempt 3: Fetching SPY data (first time for SPY in this run, should fetch or use existing SPY cache) ---")
    spy_data_c1 = fetch_spy_data(ticker_symbol="SPY", period="1y")
    if spy_data_c1 is not None:
        print("\nSPY Data (First Fetch for SPY) Head:")
        print(spy_data_c1.head())
        print(f"Data dimensions: {spy_data_c1.shape}")

    print("\n--- Attempt 4: Fetching SPY data again (should use SPY cache) ---")
    spy_data_c2 = fetch_spy_data(ticker_symbol="SPY", period="1y")
    if spy_data_c2 is not None:
        print("\nSPY Data (Second Fetch for SPY) Head:")
        print(spy_data_c2.head(2))
        if spy_data_c1 is not None and spy_data_c2.equals(spy_data_c1):
            print("SPY data from second fetch is identical to the first (cached).")
        else:
            print("SPY data from second fetch differs or first fetch failed.")

    print("\n--- Attempt 5: Fetching 1 month of AAPL data without cache ---")
    aapl_data_no_cache = fetch_spy_data(ticker_symbol="AAPL", period="1mo", use_cache=False)
    if aapl_data_no_cache is not None:
        print("\nAAPL Data (No Cache - 1mo) Head:")
        print(aapl_data_no_cache.head())
        print(f"Data dimensions: {aapl_data_no_cache.shape}")

    print("\n--- Testing with a non-existent ticker ---")
    non_existent_data = fetch_spy_data(ticker_symbol="NONEXISTENTTICKERXYZ", period="1mo")
    if non_existent_data is None:
        print("Correctly returned None for a non-existent ticker.")
    else:
        print(f"Unexpectedly received data for non-existent ticker: {non_existent_data.head()}")

    print("\n--- Data Manager Test Complete ---")
