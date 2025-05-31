"""
Custom Simple Moving Average (SMA) Indicator.

This file provides an example implementation of a custom technical indicator,
the Simple Moving Average (SMA). It demonstrates how to subclass the
`CustomIndicator` interface to create new indicators that can be dynamically
loaded and utilized by the trading analysis platform.
"""
import pandas as pd
from src.core.custom_indicator_interface import CustomIndicator

class SimpleMovingAverage(CustomIndicator):
    """
    Calculates the Simple Moving Average (SMA) for a specified data column
    over a given period.

    This indicator inherits from `CustomIndicator` and implements the required
    `__init__` and `calculate` methods.

    Parameters:
        period (int): The lookback window for calculating the SMA. For example,
                      a period of 20 means the SMA is calculated over the
                      last 20 data points.
        column (str): The name of the column in the input DataFrame on which
                      the SMA is to be calculated (e.g., 'Close', 'High').
    """

    def __init__(self, period: int = 20, column: str = 'Close'):
        """
        Initializes the SimpleMovingAverage indicator with specified parameters.

        Args:
            period (int, optional): The lookback period for the SMA.
                                    Must be a positive integer. Defaults to 20.
            column (str, optional): The name of the DataFrame column to use for
                                    SMA calculation. Must be a non-empty string.
                                    Defaults to 'Close'.

        Raises:
            ValueError: If `period` is not a positive integer or if `column`
                        is an empty string.
        """
        # Call super().__init__ to store parameters (period, column) as instance attributes
        # and make them available for get_params() and UI generation.
        super().__init__(period=period, column=column)

        # Validate parameters specific to this indicator after they are set by super()
        if not isinstance(self.period, int) or self.period <= 0:
            raise ValueError("SMA 'period' must be a positive integer.")
        if not isinstance(self.column, str) or not self.column: # Check if string is empty
            raise ValueError("SMA 'column' name must be a non-empty string.")


    def calculate(self, data_df: pd.DataFrame) -> pd.Series:
        """
        Calculates the Simple Moving Average based on the initialized parameters.

        Args:
            data_df (pd.DataFrame): A pandas DataFrame containing the financial data.
                                    This DataFrame must include the column specified
                                    as `self.column` during initialization.

        Returns:
            pd.Series: A pandas Series containing the calculated SMA values. The index
                       of this Series will align with the index of the input `data_df`.
                       The Series will be named descriptively (e.g., "SMA_20_Close").
                       The initial `self.period - 1` values will be NaN, as a full
                       lookback window is not yet available.

        Raises:
            ValueError: If the column specified by `self.column` is not found in
                        the input `data_df`.
        """
        # self.period and self.column are already validated in __init__ and available.

        if self.column not in data_df.columns:
            raise ValueError(
                f"Column '{self.column}' not found in the input DataFrame. "
                f"Available columns: {data_df.columns.tolist()}"
            )

        # Calculate SMA using pandas rolling mean.
        # `min_periods=self.period` ensures that the SMA is calculated only when
        # there are enough data points for the full window, resulting in NaNs
        # at the beginning of the series where the window is not yet filled.
        # This is equivalent to the standard SMA calculation where the first
        # `period - 1` values are undefined.
        sma_series = data_df[self.column].rolling(
            window=self.period,
            min_periods=self.period # Ensures enough data for each calculation
        ).mean()

        # Assign a descriptive name to the output series for better identification
        # in charts or when merging with other data.
        sma_series.name = f"SMA_{self.period}_{self.column}"
        return sma_series

if __name__ == '__main__':
    # Example usage of the SimpleMovingAverage indicator

    # Create a dummy DataFrame
    data = {
        'Open': [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25],
        'High': [11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26],
        'Low': [9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24],
        'Close': [10.5, 11.5, 12.5, 13.5, 14.5, 15.5, 16.5, 17.5, 18.5, 19.5, 20.5, 21.5, 22.5, 23.5, 24.5, 25.5],
        'Volume': [100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 210, 220, 230, 240, 250]
    }
    idx = pd.to_datetime([f'2023-01-{i:02d}' for i in range(1, 17)])
    dummy_df = pd.DataFrame(data, index=idx)
    dummy_df.index.name = "Date"

    print("--- Testing SimpleMovingAverage Indicator ---")

    try:
        # Test with default parameters (period=20, column='Close')
        # Since dummy_df has only 16 data points, period 20 will result in all NaNs
        # when min_periods=period. This is expected.
        print(f"\nInstantiating Indicator: SimpleMovingAverage(period=20, column='Close') (default values in class)")
        sma_default = SimpleMovingAverage() # Uses period=20, column='Close'
        print(f"Indicator instance: {sma_default}")
        sma_default_values = sma_default.calculate(dummy_df.copy())
        print(f"SMA with default parameters (period=20) on {len(dummy_df)} data points (series name: {sma_default_values.name}):")
        print(sma_default_values)
        assert sma_default_values.isna().sum() == len(dummy_df), \
            "Expected all NaNs for period 20 on 16 data points with min_periods=20"


        # Test with specific parameters that should yield some results
        print(f"\nInstantiating Indicator: SimpleMovingAverage(period=5, column='Close')")
        sma_5_close = SimpleMovingAverage(period=5, column='Close')
        print(f"Indicator instance: {sma_5_close}")
        sma_5_close_values = sma_5_close.calculate(dummy_df.copy())
        print(f"SMA (period=5, column='Close', series name: {sma_5_close_values.name}):")
        print(sma_5_close_values.head(10))
        assert sma_5_close_values.isna().sum() == 4, "Expected first 4 values to be NaN for period 5"

        print(f"\nInstantiating Indicator: SimpleMovingAverage(period=3, column='High')")
        sma_3_high = SimpleMovingAverage(period=3, column='High')
        print(f"Indicator instance: {sma_3_high}")
        sma_3_high_values = sma_3_high.calculate(dummy_df.copy())
        print(f"SMA (period=3, column='High', series name: {sma_3_high_values.name}):")
        print(sma_3_high_values.head(10))
        assert sma_3_high_values.isna().sum() == 2, "Expected first 2 values to be NaN for period 3"


        # Test error case: column not found during calculate
        print("\n--- Testing Error Case: Column Not Found (during calculate) ---")
        sma_error_col_calc = SimpleMovingAverage(column='NonExistentColumn')
        print(f"Instantiated Indicator: {sma_error_col_calc}")
        try:
            sma_error_col_calc.calculate(dummy_df.copy())
        except ValueError as e:
            print(f"Correctly caught error: {e}")

        # Test error case: invalid period in constructor
        print("\n--- Testing Error Case: Invalid Period (in constructor, period=0) ---")
        try:
            SimpleMovingAverage(period=0)
        except ValueError as e:
            print(f"Correctly caught error: {e}")

        print("\n--- Testing Error Case: Invalid Period (in constructor, period=-5) ---")
        try:
            SimpleMovingAverage(period=-5)
        except ValueError as e:
            print(f"Correctly caught error: {e}")

        # Test error case: invalid column name in constructor
        print("\n--- Testing Error Case: Invalid Column Name (in constructor, column='') ---")
        try:
            SimpleMovingAverage(column="")
        except ValueError as e:
            print(f"Correctly caught error: {e}")

    except Exception as e:
        print(f"An unexpected error occurred during testing: {e}")

    print("\n--- SimpleMovingAverage Indicator Test Complete ---")
    # The prompt in the original file "Reminder: Test this new indicator with 'python -m src.core.indicator_calculator' as well."
    # is slightly misleading, as this is a custom indicator, not part of indicator_calculator.py.
    # It would be tested via the application UI or by the indicator_manager discovery tests.
    print("\nNote: This custom indicator would be discovered by `indicator_manager.py` and usable in the Streamlit UI.")
