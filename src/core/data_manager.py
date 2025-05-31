import yfinance as yf
import pandas as pd
import os
from datetime import datetime, timedelta
import yaml # Added

# --- Configuration Loading ---
CONFIG_FILE_PATH = "src/config/settings.yaml" # Path relative to project root

def load_app_config():
    """Loads application configuration from settings.yaml."""
    default_fallback_config = {
        'default_ticker': "SPY",
        'cache_dir': "data/cache/",
        'cache_file_name_template': "{ticker}_daily.csv",
        'cache_expiry_days': 1,
        'default_start_date': "5y", # Using "5y" as period if start_date is not there
        'default_end_date': None
    }
    try:
        # Correct path when script is run from project root
        path_to_check = CONFIG_FILE_PATH
        # If script is run directly from src/core, adjust path
        if not os.path.exists(CONFIG_FILE_PATH) and __name__ == '__main__':
             # This adjustment is tricky because CWD is /app when run by tool
             # but might be different if user runs it.
             # Assuming it's /app for `run_in_bash_session`
             pass # Keep CONFIG_FILE_PATH as is, as CWD is /app

        with open(path_to_check, 'r') as f:
            config = yaml.safe_load(f)
        if not config or 'data_manager' not in config:
            print(f"Warning: '{path_to_check}' is missing or 'data_manager' section is not defined. Using default fallbacks.")
            return default_fallback_config

        dm_config = config['data_manager']
        # Ensure all keys have defaults if not in YAML
        dm_config.setdefault('default_ticker', default_fallback_config['default_ticker'])
        dm_config.setdefault('cache_dir', default_fallback_config['cache_dir'])
        dm_config.setdefault('cache_file_name_template', default_fallback_config['cache_file_name_template'])
        dm_config.setdefault('cache_expiry_days', default_fallback_config['cache_expiry_days'])
        dm_config.setdefault('default_start_date', default_fallback_config['default_start_date'])
        dm_config.setdefault('default_end_date', default_fallback_config['default_end_date'])

        # Handle empty string for dates from YAML as None for yfinance if appropriate
        if not dm_config['default_start_date']: # Handles "" or None
            dm_config['default_start_date'] = default_fallback_config['default_start_date'] # fallback to "5y"
        if not dm_config['default_end_date']: # Handles "" or None
            dm_config['default_end_date'] = default_fallback_config['default_end_date'] # fallback to None (today)

        return dm_config
    except FileNotFoundError:
        print(f"Warning: Configuration file '{CONFIG_FILE_PATH}' not found. Using default fallbacks.")
        return default_fallback_config
    except yaml.YAMLError as e:
        print(f"Error parsing YAML file '{CONFIG_FILE_PATH}': {e}. Using default fallbacks.")
        return default_fallback_config


APP_CONFIG = load_app_config()

# Module-level constants that were derived from APP_CONFIG are removed or commented out.
# Functions will access these values directly from APP_CONFIG to ensure test patches are effective.
# DEFAULT_TICKER = APP_CONFIG['default_ticker']
# CACHE_DIR = APP_CONFIG['cache_dir']
# CACHE_FILE_NAME_TEMPLATE = APP_CONFIG['cache_file_name_template']
# CACHE_EXPIRY_DAYS = APP_CONFIG['cache_expiry_days']
# DEFAULT_START_DATE = APP_CONFIG['default_start_date']
# DEFAULT_END_DATE = APP_CONFIG['default_end_date']

def get_spy_data(ticker: str = None, use_cache: bool = True, start_date: str = None, end_date: str = None) -> pd.DataFrame:
    """
    Fetches historical daily market data for a given ticker.
    Uses a local CSV cache to avoid redundant downloads.
    Configuration for cache and default dates is loaded from settings.yaml.

    Args:
        ticker (str, optional): The stock ticker symbol. Defaults to `default_ticker` from config.
        use_cache (bool): Whether to use the caching mechanism.
        start_date (str, optional): Start date for data fetching (YYYY-MM-DD or period string like "1y", "5y").
                                   Defaults to `default_start_date` from config.
        end_date (str, optional): End date for data fetching (YYYY-MM-DD).
                                 Defaults to `default_end_date` from config (None means today).

    Returns:
        pd.DataFrame: DataFrame with OHLCV data, empty if fetching fails.
    """
    # Access config values directly from the (potentially patched) APP_CONFIG
    cfg_default_ticker = APP_CONFIG['default_ticker']
    cfg_cache_dir = APP_CONFIG['cache_dir']
    cfg_cache_file_name_template = APP_CONFIG['cache_file_name_template']
    cfg_cache_expiry_days = APP_CONFIG['cache_expiry_days']
    cfg_default_start_date = APP_CONFIG['default_start_date']
    cfg_default_end_date = APP_CONFIG['default_end_date']

    effective_ticker = ticker if ticker else cfg_default_ticker
    effective_start_date = start_date if start_date else cfg_default_start_date
    effective_end_date = end_date if end_date else cfg_default_end_date

    history_params = {}
    # yfinance logic:
    # - if `period` is used, `start` and `end` are ignored if also passed (period takes precedence for start date).
    # - if `start` is used, `period` is ignored. `end` can be used with `start`.
    # - if `start_date` from config is like "2020-01-01"
    # - if `start_date` from config is like "5y" (a period)

    is_period_string = isinstance(effective_start_date, str) and any(c.isalpha() for c in effective_start_date)

    if is_period_string:
        history_params['period'] = effective_start_date
        # yfinance usually ignores 'end' if 'period' is set, but if user explicitly provides end_date,
        # it might be for a period calculated from a specific end, not today.
        # However, standard yf behavior is period relative to "today" (or 'end' if 'end' is also provided for period).
        # For simplicity, if period is given, let yf handle its default end (today) unless overridden.
        if effective_end_date:
            history_params['end'] = effective_end_date
    else: # Assumed to be a date string YYYY-MM-DD
        history_params['start'] = effective_start_date
        if effective_end_date:
            history_params['end'] = effective_end_date
        # If no effective_end_date, yfinance fetches up to the latest.

    if not os.path.exists(cfg_cache_dir): # Use dynamically accessed config value
        os.makedirs(cfg_cache_dir)
        print(f"Created cache directory: {cfg_cache_dir}")

    # Use a more specific cache file name if start/end dates are involved,
    # or ensure cache logic considers if the cached data range is sufficient.
    # For now, keeping it simple: ticker only in filename. Re-fetch if stale.
    cache_file_path = os.path.join(cfg_cache_dir, cfg_cache_file_name_template.format(ticker=effective_ticker)) # Use dynamic config

    if use_cache and os.path.exists(cache_file_path):
        file_mod_time = datetime.fromtimestamp(os.path.getmtime(cache_file_path))
        if datetime.now() - file_mod_time < timedelta(days=cfg_cache_expiry_days): # Use dynamic config
            print(f"Loading '{effective_ticker}' data from cache: {cache_file_path}")
            try:
                data = pd.read_csv(cache_file_path, index_col='Date', parse_dates=True)
                if not data.empty:
                    # TODO: Add check: if cached data range matches requested start/end.
                    # For now, if cache exists and is fresh, we assume it's good.
                    # This means if you change start/end for the same ticker, it might give you
                    # the old cached full range until cache expires.
                    return data
                else:
                    print(f"Cache file {cache_file_path} is empty. Fetching new data.")
            except Exception as e:
                print(f"Error reading cache file {cache_file_path}: {e}. Fetching new data.")
        else:
            print(f"Cache file {cache_file_path} is stale. Fetching new data.")

    print(f"Fetching '{effective_ticker}' data from yfinance with params: {history_params}...")
    try:
        stock = yf.Ticker(effective_ticker)
        data = stock.history(**history_params)

        if data.empty:
            print(f"No data returned from yfinance for ticker {effective_ticker} with params {history_params}.")
            return pd.DataFrame()

        if use_cache:
            print(f"Saving '{effective_ticker}' data to cache: {cache_file_path}")
            data.to_csv(cache_file_path)

        return data
    except Exception as e:
        print(f"Error fetching data for {effective_ticker} from yfinance: {e}")
        return pd.DataFrame()

if __name__ == '__main__':
    print(f"Running data_manager.py with loaded config: {APP_CONFIG}")

    # Test with default config loaded from settings.yaml
    print("\n--- Testing with default configuration ---")
    default_data = get_spy_data()
    if not default_data.empty:
        print(f"\nData for {APP_CONFIG['default_ticker']} (using config defaults) Head:")
        print(default_data.head())
        print(f"Shape: {default_data.shape}")
        # Verify data range based on config (e.g., default_start_date)
        if not default_data.index.empty:
            print(f"Data from {default_data.index.min()} to {default_data.index.max()}")
    else:
        print(f"\nFailed to get data for {APP_CONFIG['default_ticker']} with config defaults.")

    # Test overriding ticker and dates
    print("\n--- Testing with overridden parameters ---")
    aapl_data = get_spy_data(ticker="AAPL", start_date="2023-01-01", end_date="2023-03-31")
    if not aapl_data.empty:
        print("\nAAPL Data (custom parameters: 2023-01-01 to 2023-03-31) Head:")
        print(aapl_data.head())
        print(f"Shape: {aapl_data.shape}")
        if not aapl_data.index.empty:
            print(f"Data from {aapl_data.index.min()} to {aapl_data.index.max()}")
    else:
        print("\nFailed to get AAPL data with custom parameters.")

    # Test using a period string for start_date
    print("\n--- Testing with period string ---")
    msft_data_period = get_spy_data(ticker="MSFT", start_date="1y") # Fetch last 1 year for MSFT
    if not msft_data_period.empty:
        print("\nMSFT Data (custom parameter: last 1 year) Head:")
        print(msft_data_period.head())
        print(f"Shape: {msft_data_period.shape}")
        if not msft_data_period.index.empty:
            print(f"Data from {msft_data_period.index.min()} to {msft_data_period.index.max()}")
    else:
        print("\nFailed to get MSFT data with period string.")

    # Test caching behavior (run twice)
    print("\n--- Testing caching ---")
    print("First call (should fetch or use stale):")
    _ = get_spy_data(ticker="GOOG")
    print("Second call (should load from cache if first succeeded and cache is fresh):")
    goog_data_cached = get_spy_data(ticker="GOOG")
    if not goog_data_cached.empty:
        print("\nGOOG Data (cached) Head:")
        print(goog_data_cached.head())
    else:
        print("\nFailed to get GOOG data for caching test.")
