"""
Core Data Manager Module.

This module is responsible for fetching historical stock market data using the
yfinance library. It incorporates a caching mechanism to store and retrieve
data from local CSV files, reducing redundant API calls and speeding up
data loading for frequently accessed tickers.

Key functionalities include:
- Fetching data for specified ticker symbols, periods, and intervals.
- Caching fetched data to `data/cache/{TICKER_SYMBOL}_daily.csv`.
- Reading from cache if data is recent (less than 1 day old) and `use_cache` is True.
- Performing common data preprocessing and cleaning steps on the fetched or
  cached data to ensure consistency. This includes:
    - Standardizing the DatetimeIndex (UTC timezone, named 'Date').
    - Standardizing column names (e.g., 'Open', 'High', 'Low', 'Close', 'Volume').
    - Converting relevant columns to numeric types.
    - Rounding floating-point values.
    - Handling missing data (NaNs) through fill and drop strategies.
"""
import yfinance as yf
import pandas as pd
import os
from datetime import datetime, timedelta

CACHE_DIR = "data/cache"

def fetch_spy_data(ticker_symbol="SPY", period="5y", interval="1d", use_cache=True):
    """
    Fetches historical stock data for a given ticker symbol, applying caching
    and common data processing steps.

    Args:
        ticker_symbol (str, optional): The stock ticker symbol (e.g., "SPY", "AAPL").
                                       Defaults to "SPY".
        period (str, optional): The period for which to download data (e.g., "1y",
                                "5y", "max", "6mo"). Defaults to "5y".
        interval (str, optional): The interval of data points (e.g., "1d", "1wk",
                                  "1mo"). Defaults to "1d".
        use_cache (bool, optional): If True, attempts to load data from a local
                                    cache file and saves fresh data to cache. If False,
                                    always fetches fresh data and does not use or
                                    update the cache. Defaults to True.

    Returns:
        pandas.DataFrame or None: A DataFrame containing the historical stock data,
                                  indexed by date, with columns like 'Open', 'High',
                                  'Low', 'Close', 'Adj Close', and 'Volume'.
                                  Returns None if data fetching or processing encounters
                                  an unrecoverable error.

    Caching Behavior:
        - The function checks for a cache file located at
          `data/cache/{TICKER_SYMBOL.upper()}_daily.csv`.
        - If `use_cache` is True, and a valid cache file exists that was modified
          within the last 24 hours (controlled by `timedelta(days=1)`), the data
          is loaded from this cache file.
        - If `use_cache` is False, the cache is stale (older than 24 hours), the
          cache file doesn't exist, or reading from cache fails, fresh data is
          fetched from yfinance.
        - If fresh data is successfully fetched and `use_cache` is True, the newly
          processed data is saved to the cache file, overwriting any previous version.

    Data Processing Sequence:
        The following steps are applied to the data (whether loaded from cache or
        freshly fetched) to ensure consistency and usability:
        1.  Index Handling: Ensures the DataFrame index is a `pd.DatetimeIndex`,
            converted to or localized as 'UTC' timezone, and named 'Date'.
            Using UTC helps standardize time across different sources and DST changes.
        2.  Column Standardization: Renames columns to a consistent title case format
            (e.g., 'open' becomes 'Open'). 'Adj Close' is specifically cased as such.
            The 'Volume' column is also standardized to 'Volume', regardless of its
            original casing from yfinance.
        3.  Numeric Conversion: Converts 'Open', 'High', 'Low', 'Close', 'Adj Close',
            and 'Volume' columns to numeric types using `pd.to_numeric`.
            Errors during conversion are coerced to NaN.
        4.  Rounding: Rounds floating-point columns ('Open', 'High', 'Low', 'Close',
            'Adj Close') to 5 decimal places for display consistency and to avoid
            potential floating-point inaccuracies in comparisons.
        5.  NaN Filling: Handles missing data (NaNs) by first forward-filling
            (`ffill`) existing values, then backward-filling (`bfill`) any remaining NaNs.
            This approach is chosen to propagate last known values forward, and then
            fill any initial NaNs with subsequent valid data.
        6.  Drop Critical NaNs: After filling, rows where critical columns ('Open',
            'High', 'Low', 'Close') still contain NaN values are dropped. This ensures
            that core data points required for charting and analysis are present.

    Error Handling:
        - If the `CACHE_DIR` cannot be created, the function prints an error and
          returns None.
        - If yfinance fails to fetch data (e.g., invalid ticker, network issue) or
          returns an empty DataFrame, an error is printed, and None is returned.
        - If any exception occurs during the "COMMON DATA PROCESSING" block, an error
          is printed, and None is returned.
    """
    if not os.path.exists(CACHE_DIR):
        try:
            os.makedirs(CACHE_DIR)
        except OSError as e:
            print(f"Error creating cache directory {CACHE_DIR}: {e}")
            return None

    cache_file_path = os.path.join(CACHE_DIR, f"{ticker_symbol.upper()}_daily.csv")
    data = None

    if use_cache and os.path.exists(cache_file_path):
        try:
            mod_time = os.path.getmtime(cache_file_path)
            # Check if cache is less than 1 day old
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

    if data is None:
        print(f"Failed to load or fetch data for {ticker_symbol.upper()}.")
        return None

    # ==== COMMON DATA PROCESSING ====
    # This block standardizes the data format, whether loaded from cache or freshly fetched.
    try:
        # 1. Index processing:
        # Ensure DatetimeIndex for time-series operations.
        # Localize to UTC to remove timezone ambiguity and ensure consistency.
        # Name the index 'Date' for clarity.
        if not isinstance(data.index, pd.DatetimeIndex):
            data.index = pd.to_datetime(data.index, utc=True)
        elif data.index.tz is None: # If datetime but not localized
            data.index = data.index.tz_localize('UTC')
        else: # If already localized, convert to UTC
            data.index = data.index.tz_convert('UTC')
        data.index.name = 'Date'

        # 2. Standardize column names:
        # Ensures consistent column naming (Title Case, specific 'Adj Close', 'Volume').
        new_columns = {}
        for col in data.columns:
            if col.lower() == 'adj close':
                new_columns[col] = 'Adj Close' # yfinance often uses 'Adj Close'
            else:
                new_columns[col] = col.title() # Title case for Open, High, Low, Close
        data.rename(columns=new_columns, inplace=True)

        # Specifically ensure 'Volume' is named 'Volume', handling cases like 'volume'.
        if 'volume' in data.columns and 'Volume' not in data.columns: # common case
            data.rename(columns={'volume': 'Volume'}, inplace=True)
        elif 'Volume' not in data.columns: # Check if other variations of 'volume' exist
            for col_name in list(data.columns): # Iterate over a copy for safe renaming
                if col_name.lower() == 'volume':
                    data.rename(columns={col_name: 'Volume'}, inplace=True)
                    break # Found and renamed volume column

        # 3. Convert OHLCV (Open, High, Low, Close, Volume) and Adj Close columns to numeric types.
        # 'errors=coerce' will turn non-convertible values into NaN.
        cols_to_numeric = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
        for col in cols_to_numeric:
            if col in data.columns:
                data[col] = pd.to_numeric(data[col], errors='coerce')

        # 4. Round float columns to 5 decimal places for consistency and to avoid float precision issues.
        float_cols = ['Open', 'High', 'Low', 'Close', 'Adj Close']
        for col in float_cols:
            if col in data.columns and data[col].dtype == 'float64':
                data[col] = data[col].round(5)

        # 5. Handle missing data (NaNs):
        # First, forward-fill to propagate last known values.
        # Then, backward-fill to handle any NaNs at the beginning of the series.
        if data.isna().any().any():
            # print(f"NaN values found in data for {ticker_symbol.upper()}. Applying ffill then bfill.")
            data.ffill(inplace=True)
            data.bfill(inplace=True)

        # 6. Drop rows if critical columns still have NaNs after filling.
        # This ensures that essential data for OHLC charts is present.
        critical_cols = ['Open', 'High', 'Low', 'Close']
        # Filter critical_cols to only include those actually present in the DataFrame
        cols_to_check_for_nan_drop = [col for col in critical_cols if col in data.columns]
        if cols_to_check_for_nan_drop: # Proceed only if there are relevant columns to check
            rows_before_dropna = len(data)
            data.dropna(subset=cols_to_check_for_nan_drop, inplace=True)
            if len(data) < rows_before_dropna:
                print(f"Dropped {rows_before_dropna - len(data)} rows with NaNs in critical columns for {ticker_symbol.upper()}.")

    except Exception as e_process:
        print(f"Error during common data processing for {ticker_symbol.upper()}: {e_process}")
        return None # Return None if processing fails
    # ==== END COMMON DATA PROCESSING ====

    # Cache the fully processed data if it was freshly fetched and caching is enabled
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
        print(f"Index type: {type(aapl_data_c1.index)}, Index timezone: {aapl_data_c1.index.tz}")


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

    print("\n--- Testing cache refresh due to age (manual simulation needed for full test) ---")
    # To truly test cache refresh, you'd need to:
    # 1. Fetch and cache data.
    # 2. Modify the cache file's modification time to be older than 1 day.
    #    (e.g., `os.utime(cache_file_path, (datetime.now() - timedelta(days=2)).timestamp(), (datetime.now() - timedelta(days=2)).timestamp())`)
    # 3. Fetch again with use_cache=True. It should then re-fetch.
    # This is complex to automate here, but the logic is in place.
    print("Simulate fetching old cache (logic check): If cache was >1 day old, it would re-fetch.")


    print("\n--- Data Manager Test Complete ---")
