"""
Built-in Technical Indicator Calculation Module.

This module houses Python functions for calculating various common technical
indicators. These functions are designed to take pandas Series or DataFrames
as input and return pandas Series or DataFrames with the calculated
indicator values.

Currently, it includes:
- calculate_rsi: Calculates the Relative Strength Index.
"""
import pandas as pd

def calculate_rsi(data_series: pd.Series, period: int = 14) -> pd.Series:
    """
    Calculates the Relative Strength Index (RSI) for a given data series.

    The RSI is a momentum oscillator that measures the speed and change of
    price movements. It oscillates between zero and 100. Traditionally, RSI
    is considered overbought when above 70 and oversold when below 30.

    Args:
        data_series (pd.Series): A pandas Series containing the data for which
                                 RSI is to be calculated (e.g., 'Close' prices
                                 of a stock). The Series should have a
                                 DatetimeIndex if time-series properties are
                                 important, though not strictly enforced by
                                 this calculation logic itself.
        period (int, optional): The lookback period for calculating RSI.
                                Defaults to 14, a common period used in
                                technical analysis.

    Returns:
        pd.Series: A pandas Series containing the calculated RSI values, indexed
                   identically to the input `data_series`. The first `period-1`
                   values in the returned Series will be NaN, as the calculation
                   requires a sufficient number of preceding data points for the
                   initial rolling means of gains and losses.

    Raises:
        ValueError: If `data_series` is not a pandas Series.

    Specific Behaviors:
        - NaN Handling: The calculation involves a `diff()` operation which creates
          one NaN at the beginning, and then rolling means over `period` data points.
          The initial `diff()` NaN is dropped. Subsequently, the rolling mean
          will result in the first `period-1` values being NaN. These NaNs are
          retained in the output Series as they represent periods where RSI cannot
          be computed due to insufficient historical data.
        - Empty Input: If an empty `data_series` is provided, an empty `pd.Series`
          of dtype 'float64' is returned.
    """
    if not isinstance(data_series, pd.Series):
        raise ValueError("Input 'data_series' must be a pandas Series.")
    if data_series.empty:
        # Return an empty Series with the same index type if possible, or default.
        return pd.Series(dtype='float64', index=data_series.index)

    # Calculate price differences
    delta = data_series.diff(1)
    # Remove the first NaN value created by diff()
    delta = delta.dropna()

    # Separate gains and losses
    # Gains: Positive changes or zero
    # Losses: Absolute value of negative changes, or zero
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

    # Calculate Relative Strength (RS)
    # Avoid division by zero if loss is zero for a period; RS can be infinite.
    # If loss is 0, RS is effectively infinite, making RSI 100.
    # If gain is 0 and loss is 0, RS is 0, making RSI also undefined (often handled as 50 or NaN by libraries)
    # Here, if loss is 0, rs will be inf. (100 / (1 + inf)) -> 0. So RSI -> 100. This is standard.
    rs = gain / loss

    # Calculate RSI
    rsi = 100 - (100 / (1 + rs))

    # The rolling operations will introduce NaNs at the beginning of the series
    # (specifically, period-1 NaNs). These are expected.
    # To align with the original series length and index, we can reindex.
    # However, the current approach returns a series that starts after the initial NaNs from diff(),
    # and then has period-1 NaNs from rolling.
    # For consistency with data_series index, reindex rsi and fill NaNs appropriately.
    # The current rsi series is shorter by 1 than data_series due to diff().dropna().
    # And its first period-1 values are NaN.
    # A common expectation is that the RSI series aligns with the original data_series index.
    final_rsi = pd.Series(index=data_series.index, dtype='float64')
    final_rsi[rsi.index] = rsi

    return final_rsi

if __name__ == '__main__':
    # Example Usage:
    print("--- Testing RSI Calculator ---")

    # Create a sample data series
    sample_prices = pd.Series([
        10, 12, 15, 14, 13, 16, 18, 20, 19, 17, 18, 20, 22, 25, 23, 22, 20, 23, 24, 25
    ], name="Price")
    sample_prices.index = pd.date_range(start="2023-01-01", periods=len(sample_prices), name="Date")


    print("\nSample Price Data:")
    print(sample_prices)

    rsi_period = 7
    rsi_values = calculate_rsi(sample_prices, period=rsi_period)

    print(f"\nRSI (period {rsi_period}) values:")
    print(rsi_values)

    # Expected NaNs at the beginning of the RSI series is `rsi_period`
    # because `diff()` introduces 1 NaN, and `rolling(window=period)` on the result
    # of `diff()` (after dropping its NaN) needs `period-1` values to compute the first mean.
    # The `final_rsi` re-alignment ensures the length matches `sample_prices`.
    # The first `period` values should be NaN in `final_rsi`.
    # (1 from diff not being available for the first price, period-1 from rolling window)
    # Let's trace:
    # sample_prices: N values
    # delta: N values, first is NaN
    # delta.dropna(): N-1 values
    # gain/loss: N-1 values, first period-1 are NaN
    # rsi: N-1 values, first period-1 are NaN
    # final_rsi: N values. The first value of original series has no diff.
    # The next period-1 values don't have enough data for rolling.
    # So total period NaNs.

    print(f"\nNumber of NaNs in RSI: {rsi_values.isna().sum()} (expected for period {rsi_period} is {rsi_period})")
    print(f"Length of original data: {len(sample_prices)}")
    print(f"Length of RSI series: {len(rsi_values)}")


    if not rsi_values.dropna().empty:
        print(f"First valid RSI value is at index: {rsi_values.first_valid_index()}")
        print(f"Value: {rsi_values.dropna().iloc[0]}")
    else:
        print("RSI series is all NaNs (perhaps data too short or period too long).")

    print("\n--- RSI Calculator Test Complete ---")

[end of src/core/indicator_calculator.py]
