import pandas as pd
import numpy as np
from src.core.custom_indicator_interface import CustomIndicator

class SatyPhaseOscillator(CustomIndicator):
    """
    Saty Phase Oscillator.
    An adaptation of a TradingView script to monitor various market phases.
    """

    def __init__(self,
                 ema_period_pivot: int = 21,
                 atr_period: int = 14,
                 stdev_period_bollinger: int = 21,
                 signal_smoothing_period: int = 3,
                 atr_multiplier_raw_signal: float = 3.0,
                 atr_multiplier_compression: float = 2.0,
                 atr_multiplier_expansion: float = 1.854,
                 # For CustomIndicator base class & plotting hints
                 show_label: bool = True,
                 show_zone_crosses: bool = True,
                 show_extreme_crosses: bool = True,
                 **kwargs
                ):
        super().__init__(
            ema_period_pivot=ema_period_pivot,
            atr_period=atr_period,
            stdev_period_bollinger=stdev_period_bollinger,
            signal_smoothing_period=signal_smoothing_period,
            atr_multiplier_raw_signal=atr_multiplier_raw_signal,
            atr_multiplier_compression=atr_multiplier_compression,
            atr_multiplier_expansion=atr_multiplier_expansion,
            show_label=show_label, # Passed for potential use by plotting logic
            show_zone_crosses=show_zone_crosses,
            show_extreme_crosses=show_extreme_crosses,
            **kwargs
        )
        self.ema_period_pivot = ema_period_pivot
        self.atr_period = atr_period
        self.stdev_period_bollinger = stdev_period_bollinger
        self.signal_smoothing_period = signal_smoothing_period
        self.atr_multiplier_raw_signal = atr_multiplier_raw_signal
        self.atr_multiplier_compression = atr_multiplier_compression
        self.atr_multiplier_expansion = atr_multiplier_expansion

        # Parameter validation (can be expanded)
        if not isinstance(self.ema_period_pivot, int) or self.ema_period_pivot <= 0:
            raise ValueError("EMA period for pivot must be a positive integer.")
        if not isinstance(self.atr_period, int) or self.atr_period <= 0:
            raise ValueError("ATR period must be a positive integer.")
        # ... add more validation as needed

        self.colors_map = {
            'green': '#00ff00', 'red': '#ff0000', 'magenta': '#ff00ff',
            'light_gray': '#c8c8c8', 'gray': '#969696',
            'dark_gray': '#646464', 'yellow': '#ffff00', 'lime': '#00ff00' # lime often same as green
        }

    def _calculate_ema(self, data: pd.Series, period: int) -> pd.Series:
        """Calculates Exponential Moving Average using pandas."""
        if period <= 0: return pd.Series(np.nan, index=data.index)
        # Standard EMA formula: alpha = 2 / (period + 1)
        # Pandas .ewm(span=period, adjust=False) is equivalent to common EMA definition.
        return data.ewm(span=period, adjust=False).mean()

    def _calculate_stdev(self, data: pd.Series, period: int) -> pd.Series:
        """Calculates Standard Deviation using pandas."""
        if period <= 0: return pd.Series(np.nan, index=data.index)
        return data.rolling(window=period).std()

    def _calculate_atr(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
        """Calculates Average True Range using pandas."""
        if period <= 0: return pd.Series(np.nan, index=high.index)
        prev_close = close.shift(1)
        tr1 = high - low
        tr2 = np.abs(high - prev_close)
        tr3 = np.abs(low - prev_close)
        true_range = pd.DataFrame({'tr1': tr1, 'tr2': tr2, 'tr3': tr3}).max(axis=1)
        # ATR is typically an EMA of True Range, but user script used SMA (rolling mean)
        # To align with common ATR definition, using EMA for ATR:
        atr = true_range.ewm(span=period, adjust=False).mean()
        # If strictly following user script's "rolling(window=period).mean()":
        # atr = true_range.rolling(window=period).mean()
        return atr

    def calculate(self, data_df: pd.DataFrame) -> dict:
        if not all(col in data_df.columns for col in ['High', 'Low', 'Close']):
            raise ValueError("Input DataFrame must contain 'High', 'Low', and 'Close' columns.")

        high = data_df['High']
        low = data_df['Low']
        close = data_df['Close']

        pivot = self._calculate_ema(close, self.ema_period_pivot)
        above_pivot = close >= pivot

        bband_stdev = self._calculate_stdev(close, self.stdev_period_bollinger)
        bband_offset = 2.0 * bband_stdev
        bband_up = pivot + bband_offset
        bband_down = pivot - bband_offset

        atr = self._calculate_atr(high, low, close, self.atr_period)
        atr = atr.bfill().ffill() # Handle potential leading NaNs

        safe_atr = atr.replace(0, np.nan).ffill()
        # If safe_atr is still NaN (e.g., ATR was zero everywhere or data too short),
        # calculations involving division by safe_atr will result in NaN, which is acceptable.

        compression_threshold_up = pivot + (self.atr_multiplier_compression * atr)
        compression_threshold_down = pivot - (self.atr_multiplier_compression * atr)
        expansion_threshold_up = pivot + (self.atr_multiplier_expansion * atr)
        expansion_threshold_down = pivot - (self.atr_multiplier_expansion * atr)

        compression = pd.Series(np.where(above_pivot,
                                         bband_up - compression_threshold_up,
                                         compression_threshold_down - bband_down),
                                index=data_df.index)

        in_expansion_zone = pd.Series(np.where(above_pivot,
                                               bband_up - expansion_threshold_up,
                                               expansion_threshold_down - bband_down),
                                      index=data_df.index)

        expansion_shifted_compression = compression.shift(1)
        expansion = expansion_shifted_compression <= compression # Boolean series

        compression_tracker = pd.Series(False, index=data_df.index)
        # Iterative calculation for compression_tracker
        # Ensure indices are aligned; direct iteration over series values for logic.
        # This loop is kept for direct translation of the logic.
        # Vectorized alternatives could be explored but might obscure the original script's logic.
        _compression_np = compression.to_numpy()
        _expansion_np = expansion.to_numpy()
        _in_expansion_zone_np = in_expansion_zone.to_numpy()
        _compression_tracker_np = compression_tracker.to_numpy()

        for i in range(1, len(compression)): # Start from 1 due to shift/previous values
            if _expansion_np[i] and _in_expansion_zone_np[i] > 0:
                _compression_tracker_np[i] = False
            elif _compression_np[i] <= 0:
                _compression_tracker_np[i] = True
            # else: _compression_tracker_np[i] remains its previous value or initial False
            # This is tricky: original script: compression_tracker = nz(compression_tracker[1], false)
            # This implies dependency on previous state of compression_tracker itself if not meeting other cond.
            # The current loop implements: if condA then false, elif condB then true, else (implicitly) keep current value from loop.
            # For a direct pandas series, it should be: else: compression_tracker.iloc[i] = compression_tracker.iloc[i-1]
            # This makes it a stateful calculation.
            # Given the provided logic, the original PineScript `compression_tracker` is:
            # `compression_tracker = expansion and in_expansion_zone > 0 ? false : compression <= 0 ? true : nz(compression_tracker[1], false)`
            # This means if neither of the first two conditions are met, it carries forward its own previous state.
            # The loop needs to reflect this statefulness.
            # Let's re-evaluate the loop for correct state carry-forward:
            # Previous value of compression_tracker_np[i] is actually compression_tracker_np[i-1]
            # if no conditions are met.
            else:
                 _compression_tracker_np[i] = _compression_tracker_np[i-1]


        compression_tracker = pd.Series(_compression_tracker_np, index=data_df.index)

        # Phase Oscillator Calculation
        # Using safe_atr for division; if safe_atr is NaN, raw_signal will be NaN.
        raw_signal = ((close - pivot) / (self.atr_multiplier_raw_signal * safe_atr)) * 100

        # EMA of raw_signal. NaNs in raw_signal will propagate to oscillator.
        oscillator = self._calculate_ema(raw_signal, self.signal_smoothing_period)

        # Zone Crosses (Signals)
        oscillator_shifted = oscillator.shift(1)
        leaving_accumulation = (oscillator_shifted <= -61.8) & (oscillator > -61.8)
        leaving_extreme_down = (oscillator_shifted <= -100.0) & (oscillator > -100.0) # Corrected -100.0
        leaving_distribution = (oscillator_shifted >= 61.8) & (oscillator < 61.8)
        leaving_extreme_up = (oscillator_shifted >= 100.0) & (oscillator < 100.0) # Corrected 100.0

        # Colors Array
        plot_colors_np = np.where(compression_tracker,
                                  self.colors_map['magenta'],
                                  np.where(oscillator >= 0.0,
                                           self.colors_map['green'],
                                           self.colors_map['red']))

        return {
            'oscillator': oscillator,
            'colors': pd.Series(plot_colors_np, index=data_df.index), # Return as Series
            'compression_tracker': compression_tracker,
            'zones': {
                'extended_up': 100.0, 'distribution': 61.8, 'neutral_up': 23.6,
                'neutral_down': -23.6, 'accumulation': -61.8, 'extended_down': -100.0
            },
            'signals': {
                'leaving_accumulation': leaving_accumulation,
                'leaving_extreme_down': leaving_extreme_down,
                'leaving_distribution': leaving_distribution,
                'leaving_extreme_up': leaving_extreme_up
            },
            'colors_map': self.colors_map, # Add colors_map to the output
            # Include raw components for potential debugging or advanced plotting if needed by UI later
            # 'pivot': pivot,
            # 'atr': atr,
            # 'bband_up': bband_up,
            # 'bband_down': bband_down,
            # 'raw_signal': raw_signal
        }

if __name__ == '__main__':
    # Create a dummy DataFrame
    data_size = 250 # Increased size for better indicator calculation stability
    np.random.seed(42)

    dates = pd.date_range(start='2023-01-01', periods=data_size, freq='B')
    price_path = np.zeros(data_size)
    price_path[0] = 100
    returns = np.random.normal(loc=0.0001, scale=0.015, size=data_size-1)
    price_path[1:] = price_path[0] * np.exp(np.cumsum(returns))

    dummy_df = pd.DataFrame(index=dates)
    dummy_df['Open'] = price_path * (1 + np.random.normal(0, 0.002, size=data_size))
    dummy_df['Close'] = price_path * (1 + np.random.normal(0, 0.002, size=data_size))
    dummy_df['High'] = np.maximum(dummy_df['Open'], dummy_df['Close']) * (1 + np.random.uniform(0, 0.005, size=data_size))
    dummy_df['Low'] = np.minimum(dummy_df['Open'], dummy_df['Close']) * (1 - np.random.uniform(0, 0.005, size=data_size))
    dummy_df['Volume'] = np.random.randint(100000, 1000000, size=data_size)

    print("Dummy OHLC Data (first 5 rows):")
    print(dummy_df.head())
    print("\n")

    saty_osc = SatyPhaseOscillator()
    print(f"Instantiated Indicator: {saty_osc}")
    print(f"Parameters: {saty_osc.get_params()}") # get_params() is from CustomIndicator base
    print("\n")

    try:
        results = saty_osc.calculate(dummy_df.copy())
        print("Saty Phase Oscillator Results (first 25 rows of oscillator):")
        print(results['oscillator'].head(25))
        print("\nSaty Phase Oscillator Colors (first 25 rows):")
        print(results['colors'].head(25))
        print("\nSaty Phase Oscillator Compression Tracker (first 25 rows):")
        print(results['compression_tracker'].head(25))

        print("\nChecking for NaN values in key outputs:")
        print(f"Oscillator NaNs: {results['oscillator'].isnull().sum()} out of {len(results['oscillator'])}")
        print(f"Colors NaNs: {results['colors'].isnull().sum()} (should be 0 if oscillator has no NaNs after smoothing window)")
        print(f"Compression Tracker NaNs: {results['compression_tracker'].isnull().sum()} (should be 0)")

        print("\nChecking for signals (sum of True values):")
        for sig_name, sig_series in results['signals'].items():
            print(f"{sig_name}: {sig_series.sum()} crosses")

    except Exception as e:
        print(f"An error occurred during SatyPhaseOscillator calculation: {e}")
        import traceback
        traceback.print_exc()

    print("\n--- Test Complete ---")
