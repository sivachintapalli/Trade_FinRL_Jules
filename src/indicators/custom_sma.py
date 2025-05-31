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
        # Call super().__init__ to store parameters
        super().__init__(period=period, column=column)

        # Validate parameters specific to this indicator
        if not isinstance(self.period, int) or self.period <= 0:
            raise ValueError("SMA period must be a positive integer.")
        if not isinstance(self.column, str) or not self.column:
            raise ValueError("SMA column name must be a non-empty string.")


    def calculate(self, data_df: pd.DataFrame) -> pd.Series:
        """
        Calculates the Simple Moving Average.

        Args:
            data_df: Pandas DataFrame with OHLCV data. Must contain the column specified
                     during initialization (e.g., 'Close').

        Returns:
            A Pandas Series containing the calculated SMA values.
            The index will match the input data_df's index.
            The series will have NaN values for the initial part where SMA cannot be computed.

        Raises:
            ValueError: If the specified column is not found in the DataFrame.
        """
        # Parameters (self.period, self.column) are already validated in __init__
        # and set by super().__init__()

        if self.column not in data_df.columns:
            raise ValueError(f"Column '{self.column}' not found in the input DataFrame. Available columns: {data_df.columns.tolist()}")

        # Use min_periods=self.period to ensure enough data for each calculation window
        sma_series = data_df[self.column].rolling(window=self.period, min_periods=self.period).mean()
        sma_series.name = f"SMA_{self.period}_{self.column}" # Assign a descriptive name to the series
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

    print("Testing SimpleMovingAverage indicator...")

    try:
        # Test with default parameters (period=20, column='Close')
        # Since dummy_df has only 16 data points, period 20 will result in all NaNs
        # when min_periods=period. This is expected.
        print(f"\nInstantiating Indicator: SimpleMovingAverage(period=20, column='Close') (default)")
        sma_default = SimpleMovingAverage()
        sma_default_values = sma_default.calculate(dummy_df.copy()) # Pass copy
        print("SMA with default parameters (period=20) on 16 data points:")
        print(sma_default_values)
        print("Note: Default period is 20, so expect all NaNs with this small dataset as min_periods=20.")

        # Test with specific parameters
        print(f"\nInstantiating Indicator: SimpleMovingAverage(period=5, column='Close')")
        sma_5_close = SimpleMovingAverage(period=5, column='Close')
        sma_5_close_values = sma_5_close.calculate(dummy_df.copy()) # Pass copy
        print("SMA (period=5, column='Close'):")
        print(sma_5_close_values.head(10)) # Show first 10

        print(f"\nInstantiating Indicator: SimpleMovingAverage(period=3, column='High')")
        sma_3_high = SimpleMovingAverage(period=3, column='High')
        sma_3_high_values = sma_3_high.calculate(dummy_df.copy()) # Pass copy
        print("SMA (period=3, column='High'):")
        print(sma_3_high_values.head(10)) # Show first 10

        # Test error case: column not found
        print("\nTesting error case: column not found...")
        sma_error_col = SimpleMovingAverage(column='NonExistent') # Instantiation is fine
        print(f"Instantiated Indicator: {sma_error_col}")
        try:
            sma_error_col.calculate(dummy_df.copy()) # Pass copy
        except ValueError as e:
            print(f"Correctly caught error: {e}")

        # Test error case: invalid period in constructor
        print("\nTesting error case: invalid period in constructor (period=0)...")
        try:
            SimpleMovingAverage(period=0)
        except ValueError as e:
            print(f"Correctly caught error: {e}")

        print("\nTesting error case: invalid period in constructor (period=-5)...")
        try:
            SimpleMovingAverage(period=-5)
        except ValueError as e:
            print(f"Correctly caught error: {e}")

        # Test error case: invalid column in constructor
        print("\nTesting error case: invalid column name in constructor (column='')...")
        try:
            SimpleMovingAverage(column="")
        except ValueError as e:
            print(f"Correctly caught error: {e}")

    except Exception as e:
        print(f"An unexpected error occurred during testing: {e}")

    print("\nReminder: Test this new indicator with 'python -m src.core.indicator_calculator' as well.")
