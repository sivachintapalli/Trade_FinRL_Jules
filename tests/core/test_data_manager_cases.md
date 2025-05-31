# Test Cases for `data_manager.py`

## Module: `data_manager.py`

### Function: `fetch_spy_data(ticker_symbol, period, interval, use_cache)`

---

## Test Scenarios:

### 1. Basic Data Fetching

*   **Test Case 1.1: Valid Common Ticker and Standard Period**
    *   **Description:** Fetch data for a common, valid ticker (e.g., "AAPL") with a standard period (e.g., "1y").
    *   **Inputs:**
        *   `ticker_symbol`: "AAPL"
        *   `period`: "1y"
        *   `interval`: "1d" (default)
        *   `use_cache`: `True` (default)
    *   **Expected Output:**
        *   Returns a non-empty `pandas.DataFrame`.
        *   DataFrame contains standard columns: 'Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close'.
        *   Index is a `DatetimeIndex` named 'Date' and is timezone-aware (UTC).
        *   Data types for OHLCV columns are numeric.

*   **Test Case 1.2: Valid Less Common Ticker**
    *   **Description:** Fetch data for a less common but still valid ticker (e.g., "SAN.MC" - Santander).
    *   **Inputs:**
        *   `ticker_symbol`: "SAN.MC"
        *   `period`: "6mo"
        *   `interval`: "1d"
        *   `use_cache`: `True`
    *   **Expected Output:**
        *   Returns a non-empty `pandas.DataFrame` with standard columns and properties as in Test Case 1.1.

*   **Test Case 1.3: Invalid/Non-Existent Ticker**
    *   **Description:** Attempt to fetch data for a ticker symbol that does not exist.
    *   **Inputs:**
        *   `ticker_symbol`: "NONEXISTENTTICKERXYZ"
        *   `period`: "1mo"
    *   **Expected Output:**
        *   Returns `None`.
        *   An error message is logged indicating no data was found or the ticker is invalid.

*   **Test Case 1.4: Different Valid Periods**
    *   **Description:** Fetch data for a valid ticker using various supported period strings.
    *   **Inputs:**
        *   `ticker_symbol`: "MSFT"
        *   `period`: Try "1mo", "6mo", "1y", "2y", "5y", "max".
    *   **Expected Output:**
        *   For each period, returns a non-empty `pandas.DataFrame`.
        *   The date range of the DataFrame index should approximately correspond to the specified period. (e.g., "1mo" data should span roughly one month).

*   **Test Case 1.5: Different Valid Intervals (Note on yfinance limitations)**
    *   **Description:** Fetch data using different intervals like weekly ("1wk") or monthly ("1mo").
        *   *Note: `yfinance` behavior with intervals can sometimes be inconsistent or default to daily for certain combinations or tickers. Test results should be interpreted with this in mind.*
    *   **Inputs:**
        *   `ticker_symbol`: "GOOG"
        *   `period`: "1y"
        *   `interval`: Try "1wk", then "1mo".
    *   **Expected Output:**
        *   Returns a non-empty `pandas.DataFrame`.
        *   The data points should reflect the specified interval (e.g., for "1wk", dates should be approximately 7 days apart).

---

### 2. Cache Functionality

*Data for these tests should ideally use a unique ticker not used in other tests to avoid interference, or the cache directory should be cleaned before each specific cache test run.*

*   **Test Case 2.1: First Fetch with `use_cache=True`**
    *   **Description:** Fetch data for a ticker for the very first time with caching enabled.
    *   **Setup:** Ensure no cache file exists for the chosen ticker (e.g., `data/cache/TESTCACHE_daily.csv`).
    *   **Inputs:**
        *   `ticker_symbol`: "TESTCACHE" (or a real ticker like "INTC")
        *   `period`: "3mo"
        *   `use_cache`: `True`
    *   **Expected Output:**
        *   Data is fetched from `yfinance` (indicated by log message: "Fetching fresh data...").
        *   A cache file (e.g., `data/cache/TESTCACHE_daily.csv`) is created and populated with the fetched data.
        *   The function returns a valid `pandas.DataFrame`.
        *   Log message indicates data was cached: "Data cached to ...".

*   **Test Case 2.2: Immediate Second Fetch with `use_cache=True`**
    *   **Description:** Fetch the same data again immediately after Test Case 2.1, with caching still enabled.
    *   **Setup:** Test Case 2.1 must have run successfully.
    *   **Inputs:**
        *   `ticker_symbol`: "TESTCACHE" (same as 2.1)
        *   `period`: "3mo" (same as 2.1)
        *   `use_cache`: `True`
    *   **Expected Output:**
        *   Data is loaded from the cache file (indicated by log message: "Loading ... data from cache...").
        *   The returned DataFrame is identical to the one from Test Case 2.1.
        *   No "Fetching fresh data..." log message for this ticker.

*   **Test Case 2.3: Fetch with `use_cache=False` (Cache Exists)**
    *   **Description:** Fetch data with `use_cache=False` even if a valid cache file exists.
    *   **Setup:** A cache file for the ticker exists (e.g., from Test Case 2.1).
    *   **Inputs:**
        *   `ticker_symbol`: "TESTCACHE"
        *   `period`: "3mo"
        *   `use_cache`: `False`
    *   **Expected Output:**
        *   Data is fetched fresh from `yfinance` (indicated by log message: "Fetching fresh data...").
        *   The existing cache file is potentially overwritten with this newly fetched data (if processing is successful).
        *   The function returns a valid `pandas.DataFrame`.

*   **Test Case 2.4: Cache Expiry (Manual Simulation)**
    *   **Description:** Simulate cache expiry by modifying the cache file's timestamp.
    *   **Setup:**
        1.  Fetch data for a ticker with `use_cache=True` to create a cache file (e.g., "EXPIRYCACHE").
        2.  Manually change the modification timestamp of `data/cache/EXPIRYCACHE_daily.csv` to be older than 1 day (e.g., 2 days ago).
    *   **Inputs:**
        *   `ticker_symbol`: "EXPIRYCACHE"
        *   `period`: "3mo"
        *   `use_cache`: `True`
    *   **Expected Output:**
        *   Log message indicates cache is stale: "Cache for ... is older than 1 day. Fetching fresh data."
        *   Data is fetched fresh from `yfinance`.
        *   The cache file is updated with the new data.
        *   The function returns a valid `pandas.DataFrame`.

*   **Test Case 2.5: Corrupted/Unreadable Cache File (Manual Simulation)**
    *   **Description:** Simulate a scenario where the cache file is corrupted or unreadable.
    *   **Setup:**
        1.  Create a cache file for a ticker (e.g., "CORRUPTCACHE").
        2.  Manually corrupt the content of `data/cache/CORRUPTCACHE_daily.csv` (e.g., make it not a valid CSV, or change permissions if feasible in test environment).
    *   **Inputs:**
        *   `ticker_symbol`: "CORRUPTCACHE"
        *   `period`: "3mo"
        *   `use_cache`: `True`
    *   **Expected Output:**
        *   A log message indicates an error reading from cache (e.g., "Error reading from cache... Fetching fresh data.").
        *   The system handles the error gracefully and attempts a fresh fetch from `yfinance`.
        *   If fresh fetch is successful, returns a valid `pandas.DataFrame` and updates the cache file.

---

### 3. Data Processing (within `fetch_spy_data`)

*These tests verify the common data processing steps applied irrespective of whether data is fresh or from cache.*

*   **Test Case 3.1: Index Processing**
    *   **Description:** Verify the DataFrame's index properties.
    *   **Inputs:** Any successful data fetch (e.g., `ticker_symbol="IBM", period="1mo"`).
    *   **Expected Output:**
        *   The `DataFrame.index` is an instance of `pd.DatetimeIndex`.
        *   The `DataFrame.index.tz` is not `None` and represents UTC (e.g., `pytz.UTC` or `datetime.timezone.utc`).
        *   The `DataFrame.index.name` is 'Date'.

*   **Test Case 3.2: Column Name Standardization**
    *   **Description:** Ensure column names are standardized to title case, with specific handling for 'Adj Close' and 'Volume'.
        *   *Note: `yfinance` usually returns columns like 'Open', 'High', 'Low', 'Close', 'Volume', 'Dividends', 'Stock Splits', 'Adj Close'. This test focuses on ensuring the expected output format.*
    *   **Inputs:** Any successful data fetch.
    *   **Expected Output:**
        *   DataFrame columns include 'Open', 'High', 'Low', 'Close', 'Volume'.
        *   If 'Adj Close' data is provided by `yfinance`, it should be present as 'Adj Close'.
        *   Other columns returned by `yfinance` (like 'Dividends', 'Stock Splits') might be present but are not the primary focus of this standardization check unless explicitly handled.

*   **Test Case 3.3: Numeric Conversion of OHLCV Columns**
    *   **Description:** Verify that Open, High, Low, Close, Volume, and Adj Close columns are of numeric data types.
    *   **Inputs:** Any successful data fetch.
    *   **Expected Output:**
        *   `df['Open'].dtype` is numeric (e.g., `float64`).
        *   `df['High'].dtype` is numeric.
        *   `df['Low'].dtype` is numeric.
        *   `df['Close'].dtype` is numeric.
        *   `df['Adj Close'].dtype` (if present) is numeric.
        *   `df['Volume'].dtype` is numeric (e.g., `int64` or `float64` if `yfinance` provides it as float).
    *   **Sub-Case 3.3.1: Handling Coercion (Conceptual)**
        *   If it were possible to inject non-numeric data via `yfinance` (unlikely for standard tickers), the expectation is that `pd.to_numeric` with `errors='coerce'` would convert these to `NaN`, which are then handled by NaN processing.

*   **Test Case 3.4: Rounding of Float Columns**
    *   **Description:** Verify that float columns (Open, High, Low, Close, Adj Close) are rounded to 5 decimal places.
    *   **Inputs:** Any successful data fetch.
    *   **Expected Output:**
        *   Values in 'Open', 'High', 'Low', 'Close', 'Adj Close' (if present and float) should have at most 5 decimal places. (e.g., `round(value, 5)` should be equal to `value`).

*   **Test Case 3.5: NaN Handling**
    *   **Description:** Test how missing values (NaNs) are handled.
    *   **Setup:**
        *   Fetch data for a ticker that might have leading/internal NaNs (e.g., a very new stock, or a stock with known data gaps if one can be identified, though `yfinance` often provides cleaned data).
        *   Alternatively, one might need to manually inject NaNs into a DataFrame post-fetch but pre-common-processing if testing the fill logic in isolation (this is harder for `fetch_spy_data` as a black box).
    *   **Expected Output:**
        *   The `ffill()` then `bfill()` strategy is applied.
        *   If, after filling, NaNs still exist in critical columns ('Open', 'High', 'Low', 'Close'), rows containing these NaNs are dropped.
        *   A log message indicates if rows were dropped due to NaNs in critical columns.

*   **Test Case 3.6: Empty Data Handling from `yfinance`**
    *   **Description:** Test behavior when `yfinance` returns an empty DataFrame for a request (e.g., valid ticker but period with no data, like future dates).
    *   **Inputs:**
        *   `ticker_symbol`: "AAPL"
        *   `period`: "1d" (for a date far in the future where no data exists - `yfinance` might handle this differently, may need to find a specific case).
        *   *Alternatively, find a ticker that is known to sometimes return empty data for valid reasons.*
    *   **Expected Output:**
        *   The function returns `None` (as per current implementation: `if fetched_data.empty: return None`).
        *   A log message indicates that no data was found.

---

### 4. Error Handling (Conceptual for some cases)

*   **Test Case 4.1: Network Errors during `yfinance` Fetch**
    *   **Description:** Simulate a network error (e.g., no internet connection) during the call to `yf.Ticker().history()`. (Difficult to automate reliably in most CI/test environments).
    *   **Expected Behavior (Conceptual):**
        *   The function should catch the exception (e.g., `requests.exceptions.ConnectionError` or similar).
        *   It should log an error message related to the fetch failure.
        *   It should return `None`.

*   **Test Case 4.2: Errors During Cache Directory Creation**
    *   **Description:** Simulate a scenario where the cache directory (`data/cache`) cannot be created (e.g., due to file system permissions). (Difficult to automate reliably).
    *   **Expected Behavior (Conceptual):**
        *   The function should catch the `OSError`.
        *   It should log an error message about failing to create the cache directory.
        *   It should return `None`.

*   **Test Case 4.3: Errors During Cache File Writing/Reading**
    *   **Description:** Simulate errors during writing to or reading from a cache file (e.g., disk full, permission error on the file itself).
    *   **Setup (Writing):** Potentially make the cache directory read-only after it's created but before data is written.
    *   **Setup (Reading):** Manually create a cache file and then change its permissions to be unreadable by the test process.
    *   **Expected Behavior (Conceptual):**
        *   **Writing:** If caching fails, an error is logged ("Error caching fully processed data..."). The function should still return the fetched and processed data for the current session.
        *   **Reading:** If reading cache fails (Test Case 2.5), an error is logged, and a fresh fetch is attempted.

---
