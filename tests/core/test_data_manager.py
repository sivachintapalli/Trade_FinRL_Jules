import pytest
from unittest.mock import patch, MagicMock
import pandas as pd
import os
from datetime import datetime, timedelta
import yaml # For loading config to compare against
from pathlib import Path # Added

# Import the functions/classes to be tested
# Assuming tests are run from the project root, so src.core... is accessible
from src.core.data_manager import get_spy_data, load_app_config

# Helper to create a dummy DataFrame similar to yfinance output
def create_dummy_df(rows=5):
    data = {
        'Open': [150.0 + i for i in range(rows)],
        'High': [152.5 + i for i in range(rows)],
        'Low': [149.5 + i for i in range(rows)],
        'Close': [151.0 + i for i in range(rows)],
        'Volume': [100000 + (i*1000) for i in range(rows)]
    }
    # Ensure dates are unique and cover potential start/end date tests
    dates = pd.to_datetime([f'2023-01-{i+1:02d}' for i in range(rows if rows > 0 else 5)]) # Default 5 if rows is 0
    df = pd.DataFrame(data, index=dates)
    df.index.name = 'Date'
    return df

# Test specific configuration - this will be used to mock APP_CONFIG in data_manager
TEST_APP_CONFIG_FOR_DM_TESTS = {
    'default_ticker': "SPYTEST_DM",
    'cache_dir': "data/cache_test_dm/", # This will be replaced by tmp_path in the fixture
    'cache_file_name_template': "{ticker}_daily_test_dm.csv",
    'cache_expiry_days': 1,
    'default_start_date': "2023-01-01",
    'default_end_date': None # None means today
}

# Fixture to manage the test cache directory based on TEST_APP_CONFIG_FOR_DM_TESTS
@pytest.fixture(autouse=True)
def manage_dm_test_cache_dir(tmp_path, monkeypatch):
    # Use pytest's tmp_path for a truly isolated cache directory per test session for data_manager tests
    dm_test_cache_path = tmp_path / "cache_test_dm_specific"
    dm_test_cache_path.mkdir(parents=True, exist_ok=True)

    # Create a copy of the config to modify for this test run
    current_run_test_config = TEST_APP_CONFIG_FOR_DM_TESTS.copy()
    current_run_test_config['cache_dir'] = str(dm_test_cache_path)

    # Patch the APP_CONFIG loaded at the module level in src.core.data_manager
    # This is crucial for tests of get_spy_data as it uses the module-level APP_CONFIG
    monkeypatch.setattr('src.core.data_manager.APP_CONFIG', current_run_test_config)

    yield current_run_test_config # Provide the modified config to tests if they need it

    # Cleanup is handled by tmp_path fixture automatically for the directory


# --- Test Cases for get_spy_data ---

@patch('yfinance.Ticker') # Mock yfinance.Ticker
def test_get_spy_data_fetch_no_cache(mock_yf_ticker_constructor, manage_dm_test_cache_dir):
    # manage_dm_test_cache_dir provides the config used by get_spy_data via monkeypatch
    test_config = manage_dm_test_cache_dir

    dummy_df = create_dummy_df()
    mock_yf_ticker_instance = MagicMock()
    mock_yf_ticker_instance.history.return_value = dummy_df
    mock_yf_ticker_constructor.return_value = mock_yf_ticker_instance

    ticker_to_test = "FETCHTEST_DM"

    df = get_spy_data(ticker=ticker_to_test, use_cache=True)

    mock_yf_ticker_instance.history.assert_called_once()
    pd.testing.assert_frame_equal(df, dummy_df)

    expected_cache_file = Path(test_config['cache_dir']) / test_config['cache_file_name_template'].format(ticker=ticker_to_test)
    assert expected_cache_file.exists()

@patch('yfinance.Ticker')
def test_get_spy_data_load_from_cache(mock_yf_ticker_constructor, manage_dm_test_cache_dir):
    test_config = manage_dm_test_cache_dir
    dummy_df = create_dummy_df()
    cache_ticker = "CACHELOADTEST_DM"

    current_test_cache_dir = Path(test_config['cache_dir'])
    test_cache_file = current_test_cache_dir / test_config['cache_file_name_template'].format(ticker=cache_ticker)
    dummy_df.to_csv(test_cache_file) # Create the cache file

    df = get_spy_data(ticker=cache_ticker, use_cache=True)

    mock_yf_ticker_constructor.assert_not_called()
    pd.testing.assert_frame_equal(df, dummy_df, check_dtype=False) # dtypes can differ slightly after CSV I/O

@patch('yfinance.Ticker')
def test_get_spy_data_cache_stale(mock_yf_ticker_constructor, manage_dm_test_cache_dir):
    test_config = manage_dm_test_cache_dir
    dummy_df_old = create_dummy_df(rows=2)
    dummy_df_new = create_dummy_df(rows=5) # New data is different
    stale_ticker = "STALETEST_DM"

    current_test_cache_dir = Path(test_config['cache_dir'])
    test_cache_file = current_test_cache_dir / test_config['cache_file_name_template'].format(ticker=stale_ticker)

    dummy_df_old.to_csv(test_cache_file)
    # Make the file stale
    old_time = (datetime.now() - timedelta(days=test_config['cache_expiry_days'] + 1)).timestamp()
    os.utime(test_cache_file, (old_time, old_time))

    mock_yf_ticker_instance = MagicMock()
    mock_yf_ticker_instance.history.return_value = dummy_df_new
    mock_yf_ticker_constructor.return_value = mock_yf_ticker_instance

    df = get_spy_data(ticker=stale_ticker, use_cache=True)

    mock_yf_ticker_instance.history.assert_called_once() # Should have fetched new data
    pd.testing.assert_frame_equal(df, dummy_df_new)
    assert test_cache_file.exists() # Cache file should be updated

@patch('yfinance.Ticker')
def test_get_spy_data_fetch_error(mock_yf_ticker_constructor, manage_dm_test_cache_dir):
    test_config = manage_dm_test_cache_dir # Ensure config is patched
    mock_yf_ticker_instance = MagicMock()
    mock_yf_ticker_instance.history.side_effect = Exception("Simulated yfinance error")
    mock_yf_ticker_constructor.return_value = mock_yf_ticker_instance

    error_ticker="ERRORTEST_DM"
    df = get_spy_data(ticker=error_ticker, use_cache=False) # use_cache=False to ensure fetch attempt

    assert df.empty
    mock_yf_ticker_instance.history.assert_called_once()

@patch('yfinance.Ticker')
def test_get_spy_data_no_cache_option(mock_yf_ticker_constructor, manage_dm_test_cache_dir):
    test_config = manage_dm_test_cache_dir
    dummy_df = create_dummy_df()
    mock_yf_ticker_instance = MagicMock()
    mock_yf_ticker_instance.history.return_value = dummy_df
    mock_yf_ticker_constructor.return_value = mock_yf_ticker_instance

    no_cache_ticker = "NOCACHEOPTIONTEST_DM"
    expected_cache_file = Path(test_config['cache_dir']) / test_config['cache_file_name_template'].format(ticker=no_cache_ticker)

    df = get_spy_data(ticker=no_cache_ticker, use_cache=False) # Key part: use_cache=False

    mock_yf_ticker_instance.history.assert_called_once()
    pd.testing.assert_frame_equal(df, dummy_df)
    assert not expected_cache_file.exists() # Cache file should not be created

@patch('yfinance.Ticker')
def test_get_spy_data_with_specific_dates(mock_yf_ticker_constructor, manage_dm_test_cache_dir):
    test_config = manage_dm_test_cache_dir
    dummy_df = create_dummy_df() # 5 rows, 2023-01-01 to 2023-01-05
    mock_yf_ticker_instance = MagicMock()
    mock_yf_ticker_instance.history.return_value = dummy_df
    mock_yf_ticker_constructor.return_value = mock_yf_ticker_instance

    start_date_param = "2023-01-02"
    end_date_param = "2023-01-04"

    get_spy_data(ticker="DATESPARAMTEST_DM", use_cache=False, start_date=start_date_param, end_date=end_date_param)

    args, kwargs = mock_yf_ticker_instance.history.call_args
    assert kwargs.get('start') == start_date_param
    assert kwargs.get('end') == end_date_param
    assert 'period' not in kwargs # Ensure period isn't being passed

@patch('yfinance.Ticker')
def test_get_spy_data_with_period_string(mock_yf_ticker_constructor, manage_dm_test_cache_dir):
    test_config = manage_dm_test_cache_dir
    dummy_df = create_dummy_df()
    mock_yf_ticker_instance = MagicMock()
    mock_yf_ticker_instance.history.return_value = dummy_df
    mock_yf_ticker_constructor.return_value = mock_yf_ticker_instance

    period_param = "1mo"
    get_spy_data(ticker="PERIODPARAMTEST_DM", use_cache=False, start_date=period_param, end_date=None)

    args, kwargs = mock_yf_ticker_instance.history.call_args
    assert kwargs.get('period') == period_param
    assert 'start' not in kwargs # Ensure start isn't being passed if period is used


# --- Test Cases for load_app_config ---
# These tests do not depend on the manage_dm_test_cache_dir fixture's APP_CONFIG patching.
# They test load_app_config directly by mocking its CONFIG_FILE_PATH or dependencies.

def test_load_app_config_file_not_found(monkeypatch, tmp_path):
    # Make load_app_config look for a non-existent file in tmp_path
    monkeypatch.setattr('src.core.data_manager.CONFIG_FILE_PATH', str(tmp_path / "non_existent_config.yaml"))
    config = load_app_config()
    # Check for fallback defaults
    assert config['default_ticker'] == "SPY"
    assert config['cache_dir'] == "data/cache/"
    assert config['default_start_date'] == "5y"

@patch('yaml.safe_load')
@patch('builtins.open', new_callable=MagicMock)
def test_load_app_config_yaml_error(mock_open, mock_yaml_safe_load, monkeypatch, tmp_path):
    # Ensure builtins.open is mocked correctly for the context manager
    mock_open.return_value.__enter__.return_value = MagicMock() # Mock the file object
    mock_yaml_safe_load.side_effect = yaml.YAMLError("Simulated YAML error")

    # Make load_app_config use a dummy path that open will be called with
    dummy_path_for_yaml_error = str(tmp_path / "dummy_config_for_yaml_error.yaml")
    monkeypatch.setattr('src.core.data_manager.CONFIG_FILE_PATH', dummy_path_for_yaml_error)

    config = load_app_config()
    # Check for fallback defaults
    assert config['default_ticker'] == "SPY"
    assert config['cache_expiry_days'] == 1
    assert config['default_start_date'] == "5y"

def test_load_app_config_loads_actual_file(monkeypatch, tmp_path):
    actual_config_content = {
        'data_manager': {
            'default_ticker': "ACTUALSPY_FROM_FILE",
            'cache_dir': "actual/cache_from_file/",
            'cache_file_name_template': "actual_file_{ticker}.csv",
            'cache_expiry_days': 77,
            'default_start_date': "2022-02-02",
            'default_end_date': "2023-02-02"
        }
    }
    temp_config_file = tmp_path / "temp_actual_config.yaml"
    with open(temp_config_file, 'w') as f:
        yaml.dump(actual_config_content, f)

    # Make load_app_config use this temporary file
    monkeypatch.setattr('src.core.data_manager.CONFIG_FILE_PATH', str(temp_config_file))

    config = load_app_config()
    dm_config = actual_config_content['data_manager']
    assert config['default_ticker'] == dm_config['default_ticker']
    assert config['cache_dir'] == dm_config['cache_dir']
    assert config['cache_expiry_days'] == dm_config['cache_expiry_days']
    assert config['default_start_date'] == dm_config['default_start_date']
    assert config['default_end_date'] == dm_config['default_end_date']

def test_load_app_config_partial_config_uses_defaults(monkeypatch, tmp_path):
    partial_config_content = {
        'data_manager': {
            'default_ticker': "PARTIALSPY",
            # cache_dir is missing, should use fallback
            'cache_expiry_days': 3,
            # default_start_date is missing
        }
    }
    temp_config_file = tmp_path / "temp_partial_config.yaml"
    with open(temp_config_file, 'w') as f:
        yaml.dump(partial_config_content, f)

    monkeypatch.setattr('src.core.data_manager.CONFIG_FILE_PATH', str(temp_config_file))

    config = load_app_config()
    # Check specified values
    assert config['default_ticker'] == "PARTIALSPY"
    assert config['cache_expiry_days'] == 3
    # Check fallback values for missing keys
    assert config['cache_dir'] == "data/cache/" # Fallback default
    assert config['cache_file_name_template'] == "{ticker}_daily.csv" # Fallback default
    assert config['default_start_date'] == "5y" # Fallback default
    assert config['default_end_date'] is None # Fallback default
