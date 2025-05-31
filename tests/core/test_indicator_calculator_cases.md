# Test Cases for `indicator_calculator.py`

## Module: `indicator_calculator.py`

### Function: `calculate_rsi(data_series: pd.Series, period: int = 14)`

---

## Test Scenarios for `calculate_rsi`:

### 1. Valid Inputs

*   **Test Case 1.1: Typical Prices and Standard Period**
    *   **Description:** Calculate RSI for a typical pandas Series of closing prices with the default period of 14.
    *   **Inputs:**
        *   `data_series`: A `pd.Series` of ~30-50 float values representing daily closing prices (e.g., `pd.Series([44.34, 44.09, ..., 45.00])`).
        *   `period`: 14
    *   **Expected Output:**
        *   Returns a `pd.Series` of RSI values, indexed the same as `data_series`.
        *   The first `period` (i.e., 14) values in the returned Series should be `NaN` (due to `diff(1)` and `rolling(window=period)` requiring `period` data points for the first valid calculation after the diff).
        *   Subsequent RSI values should be floats, typically between 0 and 100.
        *   **Verification:** For a small, defined `data_series`, manually calculate or use a trusted online calculator/library (e.g., TA-Lib, `pandas_ta`) to verify the correctness of the first few non-NaN RSI values.

*   **Test Case 1.2: Different Valid Periods**
    *   **Description:** Test RSI calculation with different valid lookback periods.
    *   **Inputs:**
        *   `data_series`: Same as Test Case 1.1.
        *   `period`: Try 7, then 21.
    *   **Expected Output:**
        *   Returns a `pd.Series` for each period.
        *   The number of initial `NaN` values in the RSI series should correspond to the `period` used (e.g., 7 NaNs for `period=7`, 21 NaNs for `period=21`).
        *   RSI values will differ based on the period (shorter periods result in more volatile RSI). All valid RSI values should be between 0 and 100.

*   **Test Case 1.3: Data Series with Plateaus**
    *   **Description:** Test with a `data_series` that includes sequences of consecutive identical prices.
    *   **Inputs:**
        *   `data_series`: e.g., `pd.Series([20.0, 20.5, 20.5, 20.5, 20.5, 21.0, 21.5])`
        *   `period`: 3
    *   **Expected Output:**
        *   RSI calculation proceeds without error.
        *   During sustained plateaus where price change (delta) is zero for `period` data points, `gain` and `loss` can both become zero. If `loss` is zero, `rs` becomes `inf`, and RSI becomes 100. If both `gain` and `loss` are zero, `rs` becomes `NaN` (0/0), and RSI becomes `NaN`. The exact behavior (often trending towards 50 if it was previously oscillating, or NaN if strictly flat) should be documented based on observed output for such edge cases within the rolling window. *Current implementation detail: if `loss` is 0 and `gain` is >0, RSI is 100. If `loss` is 0 and `gain` is 0, `rs` is NaN, so RSI is NaN.*

*   **Test Case 1.4: Steadily Increasing Data Series**
    *   **Description:** Test with a `data_series` where prices are consistently increasing.
    *   **Inputs:**
        *   `data_series`: e.g., `pd.Series([10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25])` (length > period)
        *   `period`: 5
    *   **Expected Output:**
        *   After initial NaNs, RSI values should be high, trending towards or at 100 (as losses will be zero or very small).

*   **Test Case 1.5: Steadily Decreasing Data Series**
    *   **Description:** Test with a `data_series` where prices are consistently decreasing.
    *   **Inputs:**
        *   `data_series`: e.g., `pd.Series([25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10])` (length > period)
        *   `period`: 5
    *   **Expected Output:**
        *   After initial NaNs, RSI values should be low, trending towards or at 0 (as gains will be zero or very small).

---

### 2. Edge Cases

*   **Test Case 2.1: Empty `data_series`**
    *   **Description:** Test with an empty `pd.Series`.
    *   **Inputs:**
        *   `data_series`: `pd.Series([], dtype=float)`
        *   `period`: 14
    *   **Expected Output:**
        *   Returns an empty `pd.Series` with `dtype='float64'` and the same index as the input (if any). (Matches current implementation).

*   **Test Case 2.2: `data_series` Shorter than `period`**
    *   **Description:** Test with a `data_series` whose length is less than the specified `period`.
    *   **Inputs:**
        *   `data_series`: `pd.Series([10, 11, 12])`
        *   `period`: 5
    *   **Expected Output:**
        *   The returned `pd.Series` should have the same index as `data_series`.
        *   All values in the returned RSI series should be `NaN`. This is because `data_series.diff(1)` results in `N-1` values. `dropna()` on this results in `N-1` values if `N>0`. The `rolling(window=period)` cannot compute a mean if it has fewer than `period` values available after the diff. The `final_rsi` reindexing step ensures output length matches input.

*   **Test Case 2.3: `data_series` Length Equal to `period`**
    *   **Description:** Test with a `data_series` whose length is exactly equal to `period`.
    *   **Inputs:**
        *   `data_series`: `pd.Series([10, 11, 12, 13, 14])`
        *   `period`: 5
    *   **Expected Output:**
        *   All values in the returned RSI series should be `NaN`.
        *   Rationale: `data_series.diff(1)` results in 4 non-NaN values. `rolling(window=5)` on these 4 values will produce all NaNs as it needs 5 values for the first calculation. The `final_rsi` reindexing makes the output series length 5, all NaNs.

*   **Test Case 2.4: `data_series` with All Same Values**
    *   **Description:** Test with a `data_series` where all price values are identical.
    *   **Inputs:**
        *   `data_series`: `pd.Series([20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20])` (length >= period)
        *   `period`: 7
    *   **Expected Output:**
        *   After the initial `period` NaNs, subsequent RSI values should be `NaN`.
        *   Rationale: `delta` will be all zeros. `gain` and `loss` will be all zeros. `rs = gain / loss` will be `0 / 0 = NaN`. Thus, RSI will be `NaN`.

*   **Test Case 2.5: `data_series` Containing `NaN` Values**
    *   **Description:** Test with a `data_series` that itself contains `NaN` values.
    *   **Inputs:**
        *   `data_series`: `pd.Series([10, 11, np.nan, 13, 14, 15, np.nan, 17])`
        *   `period`: 3
    *   **Expected Output:**
        *   `NaN` values in the input `data_series` will propagate through `diff()` and `rolling().mean()`.
        *   The output RSI series will likely have `NaN` values where influenced by the input `NaNs`, in addition to the initial `period` NaNs. The exact propagation should be documented. (e.g., an input NaN will make `delta` NaN for two spots, which then affects rolling means over the window).

---

### 3. Invalid Inputs

*   **Test Case 3.1: `data_series` Not a `pd.Series`**
    *   **Description:** Test with `data_series` input that is not a pandas Series.
    *   **Inputs:**
        *   `data_series`: `[10, 11, 12, 13, 14]` (a Python list)
        *   `period`: 5
    *   **Expected Output:**
        *   Raises `ValueError` with a message like "Input 'data_series' must be a pandas Series." (Matches current implementation).

*   **Test Case 3.2: Invalid `period` Value**
    *   **Description:** Test with `period` values that are non-integer, zero, or negative.
        *   The current `calculate_rsi` function does not explicitly validate `period` for type or positivity. This relies on `pandas.Series.rolling(window=period)` to handle it.
    *   **Inputs & Expected Behavior (based on pandas `rolling`):**
        *   `period`: 0
            *   `pd.Series.rolling(window=0)` raises `ValueError: window must be > 0`. So, `calculate_rsi` should raise this `ValueError`.
        *   `period`: -1
            *   `pd.Series.rolling(window=-1)` raises `ValueError: window must be > 0`. So, `calculate_rsi` should raise this `ValueError`.
        *   `period`: 5.5 (float)
            *   `pd.Series.rolling(window=5.5)` raises `ValueError: window must be an integer`. So, `calculate_rsi` should raise this `ValueError`.
        *   `period`: None
            *   `pd.Series.rolling(window=None)` raises `ValueError: window must be an integer or None`. If `None` is passed, it typically means an expanding window, which is not the intent for RSI `period`. `calculate_rsi` type hint is `int`. Test with `None` if allowed by type checker, otherwise this is a type violation. `rolling(window=None)` would error if `None` is not explicitly handled or if it's not an offset.
    *   **Recommendation:** Consider adding explicit validation for `period` at the beginning of `calculate_rsi` (e.g., `if not isinstance(period, int) or period <= 0: raise ValueError("period must be a positive integer.")`) for clearer error messages, although pandas handles it.

---
