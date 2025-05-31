import pytest
import pandas as pd
import os
import shutil
import importlib # For clearing module cache if needed
import sys
from pathlib import Path

# Add src directory to sys.path to allow direct import of src.core modules
PROJECT_ROOT = Path(__file__).resolve().parents[2] # Should point to /app
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.custom_indicator_interface import CustomIndicator
from src.core.indicator_calculator import load_custom_indicators, calculate_indicator

# --- Fixtures ---

@pytest.fixture
def sample_ohlcv_df():
    # Provides a sample DataFrame for testing indicator calculations
    data = {
        'Open': [i for i in range(100, 125)],
        'High': [i + 2 for i in range(100, 125)],
        'Low': [i - 2 for i in range(100, 125)],
        'Close': [i + 0.5 for i in range(100, 125)],
        'Volume': [i * 10 for i in range(100, 125)]
    }
    idx = pd.date_range(start='2023-01-01', periods=25, freq='D')
    return pd.DataFrame(data, index=idx)

@pytest.fixture
def temp_indicators_dir(tmp_path):
    # Creates a temporary directory for indicator files for isolated testing
    indicators_dir = tmp_path / "temp_indicators"
    indicators_dir.mkdir()

    # Valid indicator 1
    with open(indicators_dir / "valid_one.py", "w") as f:
        f.write("""
from src.core.custom_indicator_interface import CustomIndicator
import pandas as pd
class TempSMA(CustomIndicator):
    def __init__(self, period=5, column='Close'):
        super().__init__(period=period, column=column)
        if not isinstance(self.period, int) or self.period <= 0:
            raise ValueError("Period must be a positive integer")
        if not isinstance(self.column, str) or not self.column:
            raise ValueError("Column must be a non-empty string")

    def calculate(self, data_df):
        if self.column not in data_df.columns: raise ValueError(f'Column {self.column} not found')
        return data_df[self.column].rolling(window=self.period).mean().rename(f'TempSMA_{self.period}_{self.column}')
""")

    # Valid indicator 2 & 3 in one file
    with open(indicators_dir / "valid_two_three.py", "w") as f:
        f.write("""
from src.core.custom_indicator_interface import CustomIndicator
import pandas as pd
class AnotherSMA(CustomIndicator):
    def __init__(self, length=10, column='Close'):
        super().__init__(length=length, column=column)
        self.period = length
        if not isinstance(self.period, int) or self.period <= 0:
            raise ValueError("Period must be a positive integer")
        if not isinstance(self.column, str) or not self.column:
            raise ValueError("Column must be a non-empty string")

    def calculate(self, data_df):
        if self.column not in data_df.columns: raise ValueError(f'Column {self.column} not found')
        return data_df[self.column].rolling(window=self.period).mean().rename(f'AnotherSMA_{self.period}_{self.column}')

class YetAnotherIndicator(CustomIndicator):
    def __init__(self, col_high='High', col_low='Low'):
        super().__init__(col_high=col_high, col_low=col_low)
    def calculate(self, data_df):
        if self.col_high not in data_df.columns: raise ValueError(f'Column {self.col_high} not found')
        if self.col_low not in data_df.columns: raise ValueError(f'Column {self.col_low} not found')
        return (data_df[self.col_high] + data_df[self.col_low]) / 2
""")

    # File with no CustomIndicator subclass
    with open(indicators_dir / "not_an_indicator.py", "w") as f:
        f.write("""
class NotAnIndicator:
    pass
""")

    # Empty .py file
    with open(indicators_dir / "empty.py", "w") as f:
        f.write("")

    # Non .py file
    with open(indicators_dir / "some_data.txt", "w") as f:
        f.write("this is not python code")

    yield indicators_dir # Return Path object

    loaded_module_names = [ p.stem for p in indicators_dir.glob("*.py") ]
    for name in loaded_module_names:
        if name in sys.modules:
            del sys.modules[name]


# --- Tests for load_custom_indicators ---

def test_load_from_valid_dir(temp_indicators_dir):
    indicators = load_custom_indicators(temp_indicators_dir)
    assert "TempSMA" in indicators
    assert "AnotherSMA" in indicators
    assert "YetAnotherIndicator" in indicators
    assert len(indicators) == 3
    assert issubclass(indicators["TempSMA"], CustomIndicator)

def test_load_ignores_non_indicators_and_empty_files(temp_indicators_dir, capsys):
    indicators = load_custom_indicators(temp_indicators_dir)
    assert "NotAnIndicator" not in indicators
    # Check that 'empty.py' did not cause an issue or load anything
    # Loading 'empty' module does not print warnings by default
    # but good to ensure it doesn't add anything to indicators.
    assert not any('empty' in key.lower() for key in indicators.keys())


def test_load_from_empty_dir(tmp_path):
    empty_dir = tmp_path / "empty_indicators"
    empty_dir.mkdir()
    indicators = load_custom_indicators(str(empty_dir))
    assert len(indicators) == 0

def test_load_from_non_existent_dir(capsys):
    indicators = load_custom_indicators("non_existent_path_for_sure_123")
    assert len(indicators) == 0
    captured = capsys.readouterr()
    assert "Warning: Indicators directory not found" in captured.out

def test_load_from_file_instead_of_dir(tmp_path, capsys):
    file_path = tmp_path / "not_a_dir.txt"
    file_path.write_text("I am a file")
    indicators = load_custom_indicators(str(file_path))
    assert len(indicators) == 0
    captured = capsys.readouterr()
    assert "Warning: Indicators path is not a directory" in captured.out


def test_load_duplicate_indicator_name(temp_indicators_dir, capsys):
    duplicate_file_path = temp_indicators_dir / "duplicate_indicator.py" # Use Path object directly
    with open(duplicate_file_path, "w") as f:
        f.write("""
from src.core.custom_indicator_interface import CustomIndicator # Corrected import
import pandas as pd
class TempSMA(CustomIndicator): # Duplicate name
    def __init__(self, period=3, column='Open'): # Different params for distinction
        super().__init__(period=period, column=column)
    def calculate(self, data_df): return data_df[self.column].rolling(window=self.period).mean().rename(f'TempSMA_{self.period}_{self.column}_duplicate')
""")
    indicators = load_custom_indicators(str(temp_indicators_dir)) # load_custom_indicators expects str
    assert "TempSMA" in indicators

    captured = capsys.readouterr()
    assert "Warning: Duplicate indicator class name 'TempSMA' found" in captured.out

    # Verify which one was loaded by checking its default parameters.
    # The original TempSMA (valid_one.py) has period=5, column='Close'.
    # The duplicate TempSMA (duplicate_indicator.py) has period=3, column='Open'.
    indicator_class = indicators["TempSMA"]
    indicator_instance = indicator_class()

    if indicator_instance.period == 3:
        # This means duplicate_indicator.py's version won (was loaded last)
        assert indicator_instance.column == 'Open'
        # Warning should mention that duplicate_indicator.py overwrote something or was found as duplicate
        # The current warning is "found in {filename}" where filename is the one causing overwrite.
        # So if duplicate_indicator.py is last, warning is "found in duplicate_indicator.py"
        assert "found in duplicate_indicator.py" in captured.out
    elif indicator_instance.period == 5:
        # This means valid_one.py's version won (was loaded last)
        assert indicator_instance.column == 'Close'
        assert "found in valid_one.py" in captured.out
    else:
        pytest.fail(f"Unexpected period for TempSMA: {indicator_instance.period}")

    # Cleanup
    if "duplicate_indicator" in sys.modules:
        del sys.modules["duplicate_indicator"]
    # valid_one module might also be in sys.modules
    if "valid_one" in sys.modules and "TempSMA" not in sys.modules["valid_one"].__dict__:
         # If TempSMA was truly overwritten and points to duplicate_indicator.TempSMA
         pass # valid_one might still be there but its TempSMA is not the one in use.


# --- Tests for calculate_indicator ---

@pytest.fixture
def loaded_indicators_map(temp_indicators_dir):
    return load_custom_indicators(temp_indicators_dir)

def test_calculate_valid_indicator(sample_ohlcv_df, loaded_indicators_map):
    result = calculate_indicator(sample_ohlcv_df, "TempSMA", loaded_indicators_map, period=5, column='Close')
    assert isinstance(result, pd.Series)
    assert result.name == "TempSMA_5_Close"
    assert not result.isnull().all()
    assert result.index.equals(sample_ohlcv_df.index)
    expected_sma_at_idx_4 = sample_ohlcv_df['Close'].iloc[0:5].mean()
    # pd.testing.assert_series_equal(result.iloc[4:5], expected_series_slice, check_dtype=False, check_index_attributes=False)
    # Simpler check for the specific value and its index, if assert_series_equal is too strict on metadata
    assert result.iloc[4] == pytest.approx(expected_sma_at_idx_4)
    assert result.index[4] == sample_ohlcv_df.index[4]


def test_calculate_indicator_default_params(sample_ohlcv_df, loaded_indicators_map):
    result = calculate_indicator(sample_ohlcv_df, "TempSMA", loaded_indicators_map) # Uses TempSMA defaults: period=5, column='Close'
    assert isinstance(result, pd.Series)
    assert result.name == "TempSMA_5_Close"
    expected_sma_at_idx_4 = sample_ohlcv_df['Close'].iloc[0:5].mean()
    # pd.testing.assert_series_equal(result.iloc[4:5], expected_series_slice, check_dtype=False, check_index_attributes=False)
    assert result.iloc[4] == pytest.approx(expected_sma_at_idx_4)
    assert result.index[4] == sample_ohlcv_df.index[4]


def test_calculate_another_valid_indicator(sample_ohlcv_df, loaded_indicators_map):
    result = calculate_indicator(sample_ohlcv_df, "YetAnotherIndicator", loaded_indicators_map, col_high='High', col_low='Low')
    assert isinstance(result, pd.Series)
    assert not result.isnull().all()
    expected_val_0 = (sample_ohlcv_df['High'].iloc[0] + sample_ohlcv_df['Low'].iloc[0]) / 2
    assert result.iloc[0] == expected_val_0

def test_calculate_indicator_not_found(sample_ohlcv_df, loaded_indicators_map, capsys):
    result = calculate_indicator(sample_ohlcv_df, "NonExistentIndicator", loaded_indicators_map)
    assert result is None
    captured = capsys.readouterr()
    assert "Error: Indicator 'NonExistentIndicator' not found" in captured.out

def test_calculate_indicator_bad_params_name(sample_ohlcv_df, loaded_indicators_map, capsys):
    result = calculate_indicator(sample_ohlcv_df, "TempSMA", loaded_indicators_map, window=5, column='Close') # 'window' is wrong
    assert result is None
    captured = capsys.readouterr()
    assert "Error instantiating or calculating indicator 'TempSMA'" in captured.out
    assert "unexpected keyword argument 'window'" in captured.out # From TypeError

def test_calculate_indicator_bad_params_value(sample_ohlcv_df, loaded_indicators_map, capsys):
    result = calculate_indicator(sample_ohlcv_df, "TempSMA", loaded_indicators_map, period=0, column='Close') # period=0 is invalid
    assert result is None
    captured = capsys.readouterr()
    # This error is now caught by the ValueError block in calculate_indicator as __init__ raises ValueError
    assert "Error during calculation of indicator 'TempSMA'" in captured.out
    assert "Period must be a positive integer" in captured.out # From TempSMA's __init__ ValueError

def test_calculate_indicator_calculation_error_col_not_found(sample_ohlcv_df, loaded_indicators_map, capsys):
    result = calculate_indicator(sample_ohlcv_df, "TempSMA", loaded_indicators_map, period=5, column='NonExistentColumn')
    assert result is None
    captured = capsys.readouterr()
    assert "Error during calculation of indicator 'TempSMA'" in captured.out
    assert "Column NonExistentColumn not found" in captured.out # From TempSMA's calculate ValueError

def test_calculate_with_invalid_df_type(loaded_indicators_map, capsys):
    result = calculate_indicator("not a dataframe", "TempSMA", loaded_indicators_map) # type: ignore
    assert result is None
    captured = capsys.readouterr()
    assert "Error: data_df must be a Pandas DataFrame." in captured.out

def test_calculate_indicator_returns_wrong_type(temp_indicators_dir, sample_ohlcv_df, capsys):
    (temp_indicators_dir / "bad_return_type.py").write_text( # Use Path object directly
"""
from src.core.custom_indicator_interface import CustomIndicator # Corrected import
class BadReturnIndicator(CustomIndicator):
    def calculate(self, data_df): return "not a series"
""")
    indicators = load_custom_indicators(str(temp_indicators_dir)) # load_custom_indicators expects str
    result = calculate_indicator(sample_ohlcv_df, "BadReturnIndicator", indicators)
    assert result is None
    captured = capsys.readouterr()
    assert "Error: Indicator 'BadReturnIndicator' did not return a Pandas Series or DataFrame." in captured.out
    if "bad_return_type" in sys.modules: del sys.modules["bad_return_type"]


def test_calculate_indicator_index_mismatch_reindexes(temp_indicators_dir, sample_ohlcv_df, capsys):
    (temp_indicators_dir / "index_mismatch.py").write_text( # Use Path object directly
"""
from src.core.custom_indicator_interface import CustomIndicator # Corrected import
import pandas as pd
class IndexMismatchIndicator(CustomIndicator):
    def calculate(self, data_df):
        return pd.Series([1,2,3,4,5], index=pd.RangeIndex(start=1000, stop=1005, step=1), name="Mismatch")
""")
    indicators = load_custom_indicators(str(temp_indicators_dir)) # load_custom_indicators expects str
    result = calculate_indicator(sample_ohlcv_df.copy(), "IndexMismatchIndicator", indicators) # Pass copy of df
    assert result is not None
    assert result.index.equals(sample_ohlcv_df.index)
    captured = capsys.readouterr()
    assert "Warning: Index of the result from 'IndexMismatchIndicator' does not match" in captured.out
    assert result.isnull().all() # All values should be NaN after reindexing a completely different index
    if "index_mismatch" in sys.modules: del sys.modules["index_mismatch"]

def test_load_indicator_with_syntax_error(temp_indicators_dir, capsys):
    (temp_indicators_dir / "syntax_error.py").write_text("def broken_function(:") # Use Path object directly
    indicators = load_custom_indicators(str(temp_indicators_dir)) # load_custom_indicators expects str
    assert "syntax_error" not in [m.__name__ for c in indicators.values() for m in [inspect.getmodule(c)] if m] # Ensure no indicator from this module
    captured = capsys.readouterr()
    # The exact error message can vary depending on Python version and importlib behavior
    assert "An unexpected error occurred while loading indicator from" in captured.out or \
           "Error importing module syntax_error from" in captured.out
    if "syntax_error" in sys.modules: del sys.modules["syntax_error"]

# Cleanup for modules that might have been loaded by tests and could interfere if tests are re-run in same session
# This is a bit broad but helpful for interactive sessions. Pytest usually isolates test runs.
import inspect # Added import here for use in tests and cleanup
@pytest.fixture(scope="session", autouse=True)
def cleanup_modules():
    yield
    # This cleanup might be too aggressive or not perfectly targeted for all test runners / scenarios.
    # It's generally better to rely on pytest's test isolation if possible.
    # However, dynamic module loading can sometimes leave traces.
    test_modules_prefixes = ['valid_one', 'valid_two_three', 'not_an_indicator', 'empty',
                             'duplicate_indicator', 'bad_return_type', 'index_mismatch', 'syntax_error']
    modules_to_delete = [m for m in sys.modules if any(m.startswith(prefix) for prefix in test_modules_prefixes)]
    for mod_name in modules_to_delete:
        del sys.modules[mod_name]
