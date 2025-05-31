import pandas as pd
import numpy as np

# Monkey-patch numpy for pandas-ta compatibility if needed
if not hasattr(np, 'NaN'):
    np.NaN = np.nan

import pandas_ta as ta # For EMA, ATR, STDEV
from src.core.custom_indicator_interface import CustomIndicator

class SatyPhaseOscillator(CustomIndicator):
    """
    Saty Phase Oscillator.
    A range-based signal to monitor various phases of the market.
    """

    def __init__(self,
                 ema_period: int = 21,
                 atr_period: int = 14,
                 stdev_period: int = 21,
                 signal_smoothing_period: int = 3,
                 atr_multiplier: float = 3.0,
                 bb_atr_compression_multiplier: float = 2.0,
                 bb_atr_expansion_multiplier: float = 1.854,
                 show_label: bool = True, # Will be used by plotting logic later
                 show_zone_crosses: bool = True, # For plotting logic
                 show_extreme_crosses: bool = True): # For plotting logic
        """
        Initializes the SatyPhaseOscillator indicator.

        Args:
            ema_period (int): Period for the pivot EMA. Default 21.
            atr_period (int): Period for ATR calculation. Default 14.
            stdev_period (int): Period for StdDev in Bollinger Bands. Default 21.
            signal_smoothing_period (int): Period for smoothing the raw signal. Default 3.
            atr_multiplier (float): Multiplier for ATR in raw signal calc. Default 3.0.
            bb_atr_compression_multiplier (float): ATR multiplier for compression threshold. Default 2.0.
            bb_atr_expansion_multiplier (float): ATR multiplier for expansion threshold. Default 1.854.
            show_label (bool): Flag to show labels (handled by plotting). Default True.
            show_zone_crosses (bool): Flag to show zone crosses (handled by plotting). Default True.
            show_extreme_crosses (bool): Flag to show extreme crosses (handled by plotting). Default True.
        """
        super().__init__(
            ema_period=ema_period,
            atr_period=atr_period,
            stdev_period=stdev_period,
            signal_smoothing_period=signal_smoothing_period,
            atr_multiplier=atr_multiplier,
            bb_atr_compression_multiplier=bb_atr_compression_multiplier,
            bb_atr_expansion_multiplier=bb_atr_expansion_multiplier,
            show_label=show_label,
            show_zone_crosses=show_zone_crosses,
            show_extreme_crosses=show_extreme_crosses
        )

        # Parameter validation
        if not isinstance(self.ema_period, int) or self.ema_period <= 0:
            raise ValueError("EMA period must be a positive integer.")
        if not isinstance(self.atr_period, int) or self.atr_period <= 0:
            raise ValueError("ATR period must be a positive integer.")
        if not isinstance(self.stdev_period, int) or self.stdev_period <= 0:
            raise ValueError("StdDev period must be a positive integer.")
        if not isinstance(self.signal_smoothing_period, int) or self.signal_smoothing_period <= 0:
            raise ValueError("Signal smoothing period must be a positive integer.")
        if not isinstance(self.atr_multiplier, (int, float)) or self.atr_multiplier <= 0:
            raise ValueError("ATR multiplier must be a positive number.")
        if not isinstance(self.bb_atr_compression_multiplier, (int, float)) or self.bb_atr_compression_multiplier <= 0:
            raise ValueError("Bollinger Band ATR compression multiplier must be a positive number.")
        if not isinstance(self.bb_atr_expansion_multiplier, (int, float)) or self.bb_atr_expansion_multiplier <= 0:
            raise ValueError("Bollinger Band ATR expansion multiplier must be a positive number.")


    def calculate(self, data_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates the Saty Phase Oscillator and related signals.

        Args:
            data_df: Pandas DataFrame with 'High', 'Low', 'Close' columns.
                       It is expected to have a DatetimeIndex.

        Returns:
            A Pandas DataFrame containing:
            - 'oscillator': The Saty Phase Oscillator values.
            - 'compression_tracker': Boolean series for compression state.
            - 'extended_up_zone': Constant line at 100.0.
            - 'distribution_zone': Constant line at 61.8.
            - 'neutral_up_zone': Constant line at 23.6.
            - 'neutral_down_zone': Constant line at -23.6.
            - 'accumulation_zone': Constant line at -61.8.
            - 'extended_down_zone': Constant line at -100.0.
            - 'leaving_accumulation_signal': Values for plotting accumulation exit.
            - 'leaving_extreme_down_signal': Values for plotting extreme down exit.
            - 'leaving_distribution_signal': Values for plotting distribution exit.
            - 'leaving_extreme_up_signal': Values for plotting extreme up exit.
        """
        if not all(col in data_df.columns for col in ['High', 'Low', 'Close']):
            raise ValueError("Input DataFrame must contain 'High', 'Low', and 'Close' columns.")

        close = data_df['Close']
        high = data_df['High']
        low = data_df['Low']

        # ATR calculation using pandas_ta
        atr = ta.atr(high=high, low=low, close=close, length=self.atr_period)
        if atr is None or atr.isnull().all(): # Handle case where ATR calculation fails or returns all NaNs
            atr = pd.Series(np.nan, index=data_df.index)


        # Pivot Data
        pivot = ta.ema(close, length=self.ema_period)
        above_pivot = close >= pivot

        # Bollinger Band Compression Signal
        bband_stdev = ta.stdev(close, length=self.stdev_period) # Renamed to avoid conflict
        if bband_stdev is None: bband_stdev = pd.Series(np.nan, index=data_df.index)

        bband_offset = 2.0 * bband_stdev
        bband_up = pivot + bband_offset
        bband_down = pivot - bband_offset

        compression_threshold_up = pivot + (self.bb_atr_compression_multiplier * atr)
        compression_threshold_down = pivot - (self.bb_atr_compression_multiplier * atr)
        expansion_threshold_up = pivot + (self.bb_atr_expansion_multiplier * atr)
        expansion_threshold_down = pivot - (self.bb_atr_expansion_multiplier * atr)

        # Calculate compression based on above_pivot condition
        compression = pd.Series(np.nan, index=data_df.index)
        # Ensure operands are aligned and handle potential all-NaN series from ta
        for ser in [bband_up, compression_threshold_up, compression_threshold_down, bband_down, pivot, atr, bband_stdev]:
            if ser is None: # Should not happen if initialized above, but as safeguard
                ser = pd.Series(np.nan, index=data_df.index)

        compression.loc[above_pivot] = bband_up.loc[above_pivot] - compression_threshold_up.loc[above_pivot]
        compression.loc[~above_pivot] = compression_threshold_down.loc[~above_pivot] - bband_down.loc[~above_pivot]

        # Calculate in_expansion_zone based on above_pivot condition
        in_expansion_zone = pd.Series(np.nan, index=data_df.index)
        in_expansion_zone.loc[above_pivot] = bband_up.loc[above_pivot] - expansion_threshold_up.loc[above_pivot]
        in_expansion_zone.loc[~above_pivot] = expansion_threshold_down.loc[~above_pivot] - bband_down.loc[~above_pivot]

        # Initialize compression_tracker series
        # compression_tracker = pd.Series(False, index=data_df.index) # Initial default state

        # Expansion logic - Pine Script: expansion = compression[1] <= compression[0]
        expansion_condition = (compression.shift(1) <= compression) & (compression.shift(1).notna()) & (compression.notna())

        # Fill NaNs for boolean conditions to avoid issues if a comparison involves NaN
        _atr = atr.fillna(method='bfill').fillna(method='ffill')
        _compression = compression.fillna(method='bfill').fillna(method='ffill') # Fill NaNs for the conditions
        _in_expansion_zone = in_expansion_zone.fillna(method='bfill').fillna(method='ffill') # Fill NaNs
        _expansion_condition = expansion_condition.fillna(False) # Default to False if cannot be determined

        # compression_tracker logic based on re-evaluation of Pine Script:
        # Default to false, set to true if (compression <= 0), set to false if (expansion and in_expansion_zone > 0)
        # The second condition (set to false) takes precedence if both are met.

        compression_tracker = pd.Series(False, index=data_df.index) # Default value

        # Condition for setting to True: (NOT (expansion AND in_expansion_zone > 0)) AND (compression <= 0)
        # Condition for setting to False (primary): expansion AND in_expansion_zone > 0
        # Condition for setting to True (secondary): compression <= 0
        # Else: False (already default)

        cond_set_false_primary = _expansion_condition & (_in_expansion_zone > 0)
        cond_set_true_secondary = _compression <= 0

        compression_tracker.loc[cond_set_true_secondary] = True
        compression_tracker.loc[cond_set_false_primary] = False # This overrides if both were true

        # Saty Phase Oscillator Signal
        safe_atr = _atr.replace(0, np.nan).fillna(method='ffill') # Replace 0 with NaN, then fill forward
        if safe_atr.isnull().all(): # If ATR is still all NaNs (e.g. not enough data)
            safe_atr = pd.Series(1, index=data_df.index) # Avoid division by zero; result will be large if pivot varies
                                                       # Or handle as error / return NaNs for oscillator

        raw_signal = ((close - pivot) / (self.atr_multiplier * safe_atr)) * 100
        # Fill NaNs in raw_signal before EMA, common practice to prevent all NaNs from EMA if one input is NaN
        oscillator = ta.ema(raw_signal.fillna(method='bfill').fillna(0), length=self.signal_smoothing_period)


        # Output DataFrame
        output_df = pd.DataFrame(index=data_df.index)
        output_df['oscillator'] = oscillator
        output_df['compression_tracker'] = compression_tracker

        # Phase Zones (constant lines for plotting)
        output_df['extended_up_zone'] = 100.0
        output_df['distribution_zone'] = 61.8
        output_df['neutral_up_zone'] = 23.6
        output_df['neutral_down_zone'] = -23.6
        output_df['accumulation_zone'] = -61.8
        output_df['extended_down_zone'] = -100.0

        # Mean Reversion PO Crossover Signals
        oscillator_shifted = oscillator.shift(1)

        leaving_accumulation_cond = (oscillator_shifted <= -61.8) & (oscillator > -61.8)
        output_df['leaving_accumulation_signal'] = np.where(leaving_accumulation_cond, oscillator_shifted - 30, np.nan)

        leaving_extreme_down_cond = (oscillator_shifted <= -100) & (oscillator > -100)
        output_df['leaving_extreme_down_signal'] = np.where(leaving_extreme_down_cond, oscillator_shifted - 30, np.nan)

        leaving_distribution_cond = (oscillator_shifted >= 61.8) & (oscillator < 61.8)
        output_df['leaving_distribution_signal'] = np.where(leaving_distribution_cond, oscillator_shifted + 30, np.nan)

        leaving_extreme_up_cond = (oscillator_shifted >= 100) & (oscillator < 100)
        output_df['leaving_extreme_up_signal'] = np.where(leaving_extreme_up_cond, oscillator_shifted + 30, np.nan)

        return output_df

if __name__ == '__main__':
    # Create a dummy DataFrame similar to what FinRL might use
    data_size = 200
    np.random.seed(42) # for reproducibility

    # Generate more realistic price data
    start_price = 100
    drift = 0.0005
    volatility = 0.015
    returns = np.random.normal(loc=drift, scale=volatility, size=data_size-1)
    price_path = np.zeros(data_size)
    price_path[0] = start_price
    price_path[1:] = start_price * np.exp(np.cumsum(returns))

    # Corrected date generation: convert scalar string to Timestamp before adding TimedeltaIndex
    dates = pd.to_datetime(f'2023-01-01') + pd.to_timedelta(np.arange(data_size), 'D')

    dummy_ohlc_df = pd.DataFrame(index=dates)
    # Ensure High is max and Low is min of Open, Close, and their own random variation
    open_prices = price_path * (1 + np.random.normal(0, 0.005, size=data_size))
    close_prices = price_path * (1 + np.random.normal(0, 0.005, size=data_size))

    dummy_ohlc_df['Open'] = open_prices
    dummy_ohlc_df['Close'] = close_prices
    dummy_ohlc_df['High'] = np.maximum(dummy_ohlc_df['Open'], dummy_ohlc_df['Close']) * (1 + np.random.uniform(0, 0.01, size=data_size))
    dummy_ohlc_df['Low'] = np.minimum(dummy_ohlc_df['Open'], dummy_ohlc_df['Close']) * (1 - np.random.uniform(0, 0.01, size=data_size))
    dummy_ohlc_df['Volume'] = np.random.randint(100000, 1000000, size=data_size)


    print("Dummy OHLC Data (first 5 rows):")
    print(dummy_ohlc_df.head())
    print("\n")

    # Instantiate the indicator
    saty_osc = SatyPhaseOscillator(
        ema_period=21,
        atr_period=14,
        stdev_period=21,
        signal_smoothing_period=3
    )
    print(f"Instantiated Indicator: {saty_osc}")
    print(f"Parameters: {saty_osc.get_params()}")
    print("\n")

    try:
        # Calculate the indicator values
        indicator_results_df = saty_osc.calculate(dummy_ohlc_df.copy()) # Pass a copy

        print("Saty Phase Oscillator Results (first 15 rows):")
        print(indicator_results_df.head(15))
        print("\n")

        print("Saty Phase Oscillator Results (last 15 rows):")
        print(indicator_results_df.tail(15))
        print("\n")

        print("Checking for NaN values in key columns:")
        print(f"Oscillator NaNs: {indicator_results_df['oscillator'].isnull().sum()} out of {len(indicator_results_df)}")
        print(f"Compression Tracker NaNs: {indicator_results_df['compression_tracker'].isnull().sum()} out of {len(indicator_results_df)}")

        # Check one of the signal columns for non-NaN values (indicating crosses)
        print(f"Non-NaN in 'leaving_accumulation_signal': {indicator_results_df['leaving_accumulation_signal'].notna().sum()}")
        print(f"Non-NaN in 'leaving_extreme_down_signal': {indicator_results_df['leaving_extreme_down_signal'].notna().sum()}")
        print(f"Non-NaN in 'leaving_distribution_signal': {indicator_results_df['leaving_distribution_signal'].notna().sum()}")
        print(f"Non-NaN in 'leaving_extreme_up_signal': {indicator_results_df['leaving_extreme_up_signal'].notna().sum()}")


    except Exception as e:
        print(f"An error occurred during SatyPhaseOscillator calculation: {e}")
        import traceback
        traceback.print_exc()

    # Test parameter validation
    print("\nTesting parameter validation...")
    try:
        SatyPhaseOscillator(ema_period=0)
    except ValueError as e:
        print(f"Correctly caught: {e}")
    try:
        SatyPhaseOscillator(atr_multiplier=-1)
    except ValueError as e:
        print(f"Correctly caught: {e}")
