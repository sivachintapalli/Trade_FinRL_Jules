import pandas as pd

def calculate_rsi(data_series: pd.Series, period: int = 14):
    """
    Calculates the Relative Strength Index (RSI) for a given data series.

    Args:
        data_series (pd.Series): Pandas Series containing the data (e.g., 'Close' prices).
        period (int): The lookback period for RSI calculation (default: 14).

    Returns:
        pd.Series: Pandas Series containing the RSI values.
    """
    if not isinstance(data_series, pd.Series):
        raise ValueError("Input 'data_series' must be a pandas Series.")
    if data_series.empty:
        return pd.Series(dtype='float64') # Return empty series if input is empty

    delta = data_series.diff(1)
    delta = delta.dropna() # Remove the first NaN value created by diff()

    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))

    # Fill initial NaNs created by rolling mean with np.nan or a specific value if preferred
    # For RSI, it's common to have NaNs for the initial period.
    rsi = rsi.fillna(float('nan'))

    return rsi

if __name__ == '__main__':
    # Example Usage:
    print("--- Testing RSI Calculator ---")

    # Create a sample data series
    sample_prices = pd.Series([
        10, 12, 15, 14, 13, 16, 18, 20, 19, 17, 18, 20, 22, 25, 23, 22, 20, 23, 24, 25
    ])
    sample_prices.index.name = "Day"

    print("\nSample Price Data:")
    print(sample_prices)

    rsi_values = calculate_rsi(sample_prices, period=7) # Using a shorter period for more visible changes

    print(f"\nRSI (period 7) values:")
    print(rsi_values)

    # Test with a common example (e.g., from a known source to verify calculation if possible)
    # For this example, we'll just check for NaNs at the beginning
    expected_nans = 7 # period + (diff(1) which removes 1, then rolling mean removes period-1)
                      # diff() removes 1, so 19 values. rolling(7) needs 7, so first 6 after diff are NaN.
                      # Total NaNs in RSI series = 1 (from diff) + (period - 1) (from rolling)
                      # So for period 7, it's 1 + 6 = 7 NaNs at start.
                      # Actually, delta.dropna() removes the first NaN. So, rolling(period) on delta will produce period-1 NaNs.

    print(f"\nNumber of initial NaNs in RSI: {rsi_values.isna().sum()} (expected for period {7} is {7-1})")
    # The first value of delta is NaN. delta.dropna() removes it.
    # Then rolling(window=period) is applied. The first 'period-1' values of the rolling mean will be NaN.
    # So, for data_series of length N, delta has N-1 values.
    # gain/loss will have (N-1) - (period-1) = N - period valid values.
    # The rsi series will have 'period-1' NaNs at the beginning.

    if not rsi_values.empty:
        print(f"First valid RSI value: {rsi_values.dropna().iloc[0] if not rsi_values.dropna().empty else 'N/A'}")

    print("\n--- RSI Calculator Test Complete ---")
