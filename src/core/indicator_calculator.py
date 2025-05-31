import pandas as pd

def calculate_rsi(data: pd.DataFrame, window: int = 14) -> pd.Series:
    """
    Calculates the Relative Strength Index (RSI).

    Args:
        data: Pandas DataFrame with a 'Close' column.
        window: The period for RSI calculation (default is 14).

    Returns:
        A Pandas Series containing the RSI values.
    """
    if 'Close' not in data.columns:
        raise ValueError("DataFrame must contain a 'Close' column.")
    if not isinstance(data.index, pd.DatetimeIndex):
        # Attempt to convert if it's not a DatetimeIndex, common with yfinance data
        data.index = pd.to_datetime(data.index)

    delta = data['Close'].diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()

    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))

    return rsi

if __name__ == '__main__':
    # Example Usage:
    # Create a dummy DataFrame for testing
    from datetime import datetime, timedelta
    import numpy as np
    date_today = datetime.now()
    days = pd.date_range(date_today - timedelta(days=99), date_today, freq='D')
    np.random.seed(seed=42)
    # Generate somewhat realistic price data
    price_data = np.random.normal(loc=100, scale=2, size=len(days))
    price_data = np.cumsum(price_data) # Create a random walk for close prices

    dummy_df = pd.DataFrame({'Close': price_data}, index=days)

    print("Original Data with Close prices:")
    print(dummy_df.head())

    # Calculate RSI
    rsi_series = calculate_rsi(dummy_df.copy()) # Use .copy() to avoid SettingWithCopyWarning on delta calculation

    print("\nCalculated RSI (first few values will be NaN due to window):")
    print(rsi_series.head(20))

    # Test with a DataFrame that doesn't have 'Close'
    try:
        calculate_rsi(pd.DataFrame({'Open': [1,2,3]}))
    except ValueError as e:
        print(f"\nError when 'Close' column is missing: {e}")

    # Combine data for viewing
    dummy_df['RSI'] = rsi_series
    print("\nData with RSI:")
    print(dummy_df.head(20))
