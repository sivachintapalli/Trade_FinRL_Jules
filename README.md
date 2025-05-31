# Trade_FinRL_Jules

## Custom Technical Indicators

This platform provides a flexible framework for defining, loading, and utilizing your own custom technical indicators written in Python.

### Creating a Custom Indicator

To create a custom indicator, you need to write a Python class that inherits from `CustomIndicator`, which is defined in `src.core.custom_indicator_interface.py`.

**Key Requirements:**

1.  **Inheritance**: Your class must inherit from `CustomIndicator`.
2.  **`__init__` Method**:
    *   The constructor of your indicator class should accept any parameters your indicator needs (e.g., `period`, `column_name`).
    *   It **must** call `super().__init__(**kwargs)` with these parameters. These parameters will then be available as attributes of your indicator instance (e.g., `self.period`).
3.  **`calculate` Method**:
    *   You must implement a method with the signature `calculate(self, data_df: pd.DataFrame) -> pd.Series | pd.DataFrame`.
    *   `data_df`: This is a Pandas DataFrame containing OHLCV data.
    *   The method should perform the indicator calculation and return a Pandas `Series` (for single-value indicators) or `DataFrame` (for multi-value indicators like MACD with signal line).
    *   The index of the returned Series/DataFrame should align with the input `data_df`.

**Example: `SimpleMovingAverage`**

Here's an example of a custom Simple Moving Average indicator, typically found in `src/indicators/custom_sma.py`:

```python
import pandas as pd
from src.core.custom_indicator_interface import CustomIndicator

class SimpleMovingAverage(CustomIndicator):
    """
    Calculates the Simple Moving Average (SMA) for a given column and period.
    """

    def __init__(self, period: int = 20, column: str = 'Close'):
        """
        Initializes the SimpleMovingAverage indicator.

        Args:
            period (int): The lookback period for the SMA. Defaults to 20.
            column (str): The name of the column in the DataFrame to calculate SMA on.
                          Defaults to 'Close'.
        """
        super().__init__(period=period, column=column)
        if not isinstance(period, int) or period <= 0:
            raise ValueError("SMA period must be a positive integer.")
        if not isinstance(column, str) or not column:
            raise ValueError("SMA column name must be a non-empty string.")

    def calculate(self, data_df: pd.DataFrame) -> pd.Series:
        """
        Calculates the Simple Moving Average.
        """
        if self.column not in data_df.columns:
            raise ValueError(f"Column '{self.column}' not found in the input DataFrame.")

        sma_series = data_df[self.column].rolling(window=self.period, min_periods=self.period).mean()
        sma_series.name = f"SMA_{self.period}_{self.column}"
        return sma_series
```

### Using Custom Indicators

1.  **Placement**: Place your Python file containing your custom indicator class(es) into the `src/indicators/` directory.
2.  **Loading**: The system can dynamically load these indicators. The `load_custom_indicators()` function from `src.core.indicator_calculator` scans this directory and makes the indicators available.
3.  **Calculation**: Once loaded, you can calculate an indicator using the `calculate_indicator()` function, also from `src.core.indicator_calculator`.

**Example Usage:**

```python
import pandas as pd
from src.core.indicator_calculator import load_custom_indicators, calculate_indicator
# Assume sample_ohlcv_df is your Pandas DataFrame with OHLCV data

# 1. Load available custom indicators
available_indicators = load_custom_indicators() # Scans 'src/indicators/'
print(f"Loaded indicators: {list(available_indicators.keys())}")

# 2. Calculate a specific indicator
if "SimpleMovingAverage" in available_indicators:
    # Create a dummy DataFrame for the example to run
    data = {
        'Open': [10+i for i in range(30)],
        'High': [11+i for i in range(30)],
        'Low': [9+i for i in range(30)],
        'Close': [10.5+i for i in range(30)],
        'Volume': [100+i*10 for i in range(30)]
    }
    sample_ohlcv_df = pd.DataFrame(data, index=pd.date_range(start='2023-01-01', periods=30))

    sma_output = calculate_indicator(
        sample_ohlcv_df,
        "SimpleMovingAverage",
        available_indicators,
        period=10,  # Pass parameters for the indicator
        column='Close'
    )
    if sma_output is not None:
        print("\nSMA (10, Close) Output (first 15):")
        print(sma_output.head(15))
```

This framework allows for easy extension of the platform's analytical capabilities.