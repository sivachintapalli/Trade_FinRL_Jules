# Test Cases for `plot_generator.py`

## Module: `plot_generator.py`

### Function: `create_candlestick_chart(df, ticker_symbol, rsi_series, rsi_period, overbought_level, oversold_level, custom_indicators_data, ml_signals)`

*Note: These test cases primarily focus on the successful generation of a Plotly `Figure` object and the expected structure of this figure (e.g., number of subplots, presence of traces). The exact visual output (colors, specific marker placements beyond general correctness) is typically verified manually or with more advanced image comparison techniques, which are outside the scope of these markdown cases.*

---

## Test Scenarios for `create_candlestick_chart`:

### 1. Basic Chart

*   **Test Case 1.1: Valid DataFrame, No Optional Features**
    *   **Description:** Generate a basic candlestick chart with only OHLC data and a ticker symbol. All optional arguments (`rsi_series`, `custom_indicators_data`, `ml_signals`) are `None`.
    *   **Inputs:**
        *   `df`: A `pandas.DataFrame` with 'Open', 'High', 'Low', 'Close' columns and a `DatetimeIndex`. (e.g., 50 rows of data).
        *   `ticker_symbol`: "TESTICKER"
        *   `rsi_series`: `None`
        *   `custom_indicators_data`: `None`
        *   `ml_signals`: `None`
    *   **Expected Output:**
        *   Returns a `plotly.graph_objects.Figure` object.
        *   No errors are raised during chart generation.
        *   The figure should contain one primary subplot (trace type: candlestick).
        *   The chart title should include "TESTICKER Candlestick Chart".
        *   X-axis should be 'Date', Y-axis should be 'Price'.

---

### 2. With RSI

*   **Test Case 2.1: Valid RSI Series**
    *   **Description:** Generate a chart with a candlestick plot and an RSI subplot.
    *   **Inputs:**
        *   `df`: Valid OHLC DataFrame.
        *   `ticker_symbol`: "RSI_TEST"
        *   `rsi_series`: A `pd.Series` of RSI values, index-aligned with `df`.
        *   `rsi_period`: 14 (for legend/title)
        *   `overbought_level`: 70
        *   `oversold_level`: 30
        *   `custom_indicators_data`: `None`
        *   `ml_signals`: `None`
    *   **Expected Output:**
        *   Returns a `plotly.graph_objects.Figure` object.
        *   The figure should have two subplots (candlestick on top, RSI below).
        *   The RSI subplot should contain a trace for `rsi_series` values.
        *   Horizontal lines for `overbought_level` (70) and `oversold_level` (30) should be present in the RSI subplot.
        *   Y-axis title for RSI subplot should be "RSI" and range [0, 100].

*   **Test Case 2.2: RSI Series All NaNs**
    *   **Description:** Test with an `rsi_series` that contains all NaN values (e.g., if data was too short for any RSI calculation).
    *   **Inputs:**
        *   `df`: Valid OHLC DataFrame.
        *   `ticker_symbol`: "RSI_NAN_TEST"
        *   `rsi_series`: A `pd.Series` of the same length as `df.index`, but all values are `np.nan`.
        *   `rsi_period`: 14
    *   **Expected Output:**
        *   Returns a `plotly.graph_objects.Figure` object.
        *   Handles gracefully; chart generates.
        *   The RSI subplot is present but the RSI trace will show no points (or a gapped line if `connectgaps=True` was used, though typically it won't connect all NaNs).
        *   Overbought/oversold lines should still be visible in the RSI subplot.

---

### 3. With Custom Indicators

*   **Test Case 3.1: One Custom Indicator (Simple Series)**
    *   **Description:** Generate a chart with one custom indicator (e.g., a Simple Moving Average) plotted as a simple line series.
    *   **Inputs:**
        *   `df`: Valid OHLC DataFrame.
        *   `ticker_symbol`: "CUSTOM_SMA_TEST"
        *   `rsi_series`: `None`
        *   `custom_indicators_data`: `[{'name': 'SMA_20', 'series': pd.Series(...), 'plot_type': 'simple_series'}]` (series aligned with `df`).
        *   `ml_signals`: `None`
    *   **Expected Output:**
        *   Returns a `plotly.graph_objects.Figure` object.
        *   Two subplots: candlestick and one for "SMA_20".
        *   The custom indicator subplot contains a trace for the SMA series.
        *   Y-axis title for the custom indicator subplot should be "SMA_20" (or truncated if long).

*   **Test Case 3.2: Multiple Custom Indicators (Simple Series)**
    *   **Description:** Generate a chart with two distinct custom indicators.
    *   **Inputs:**
        *   `df`: Valid OHLC DataFrame.
        *   `ticker_symbol`: "MULTI_CUSTOM_TEST"
        *   `custom_indicators_data`:
            ```python
            [
                {'name': 'SMA_10', 'series': pd.Series(...), 'plot_type': 'simple_series'},
                {'name': 'EMA_30', 'series': pd.Series(...), 'plot_type': 'simple_series'}
            ]
            ```
    *   **Expected Output:**
        *   Returns a `plotly.graph_objects.Figure` object.
        *   Three subplots: candlestick, one for "SMA_10", one for "EMA_30".

*   **Test Case 3.3: Custom Indicator as DataFrame (e.g., SatyPhaseOscillator)**
    *   **Description:** Test with a custom indicator that returns a DataFrame, specifically testing the 'saty_phase_oscillator' plot type.
    *   **Inputs:**
        *   `df`: Valid OHLC DataFrame.
        *   `ticker_symbol`: "SPO_TEST"
        *   `custom_indicators_data`:
            ```python
            [{
                'name': 'Saty Phase Osc.',
                'plot_type': 'saty_phase_oscillator',
                'data_df': spo_results_df, # DataFrame from SatyPhaseOscillator.calculate()
                'params': {'show_zone_crosses': True, 'show_extreme_crosses': True} # Example params
            }]
            ```
    *   **Expected Output:**
        *   Returns a `plotly.graph_objects.Figure` object.
        *   Two subplots: candlestick and one for "Saty Phase Osc.".
        *   The SPO subplot should contain multiple traces: 'oscillator' line, zone lines, and marker traces for crossover signals if present in `spo_results_df` and enabled by `params`.
        *   Y-axis for SPO subplot should be fixed (e.g., -150 to 150).

*   **Test Case 3.4: Custom Indicator Series of Different Length**
    *   **Description:** Test how the plot handles a custom indicator series whose index/length does not perfectly align with `df`.
    *   **Inputs:**
        *   `df`: Valid OHLC DataFrame (e.g., 100 rows).
        *   `custom_indicators_data`: `[{'name': 'Short_SMA', 'series': pd.Series(...) }]` where the series is shorter (e.g., 50 rows) or longer and misaligned.
    *   **Expected Output:**
        *   The `create_candlestick_chart` function uses `series.reindex(main_df_index)`. This means:
            *   If the indicator series is shorter, it will be plotted, and the missing parts will be NaNs (gaps).
            *   If the indicator series is longer, it will be truncated to match `df.index`.
            *   If indices are misaligned but some overlap, only overlapping parts will plot.
        *   No error should be raised if reindexing can be performed. The plot will show data where available.

*   **Test Case 3.5: Custom Indicator Series All NaNs**
    *   **Description:** Test with a custom indicator series that contains only NaN values.
    *   **Inputs:**
        *   `df`: Valid OHLC DataFrame.
        *   `custom_indicators_data`: `[{'name': 'All_NaN_Indicator', 'series': pd.Series([np.nan]*len(df), index=df.index)}]`
    *   **Expected Output:**
        *   Returns a `plotly.graph_objects.Figure` object.
        *   The subplot for "All_NaN_Indicator" is present, but its trace will show no points (or a gapped line).

---

### 4. With ML Signals

*   **Test Case 4.1: Valid ML Signals**
    *   **Description:** Generate a chart with buy and sell signals from an ML model.
    *   **Inputs:**
        *   `df`: Valid OHLC DataFrame.
        *   `ticker_symbol`: "ML_SIG_TEST"
        *   `ml_signals`: A `pd.Series` aligned with `df.index`, containing values (e.g., 1 for buy, 0 for sell). Assume default signal values in function are 1 (buy) and 0 (sell).
    *   **Expected Output:**
        *   Returns a `plotly.graph_objects.Figure` object.
        *   The main candlestick chart should have additional scatter traces for buy signals (e.g., green triangles below lows) and sell signals (e.g., red triangles above highs) at the correct dates.

*   **Test Case 4.2: ML Signals All Zeros/No Actionable Signals**
    *   **Description:** Test with `ml_signals` that contain no values corresponding to buy or sell actions (e.g., all are hold signals, or a value not mapped to buy/sell).
    *   **Inputs:**
        *   `df`: Valid OHLC DataFrame.
        *   `ml_signals`: A `pd.Series` containing only values like `np.nan` or a value not equal to the function's `buy_signal_val` or `sell_signal_val`.
    *   **Expected Output:**
        *   Returns a `plotly.graph_objects.Figure` object.
        *   No buy or sell marker traces should be added to the candlestick chart.

*   **Test Case 4.3: ML Signals of Different Length/Alignment**
    *   **Description:** Test how `ml_signals` of a different length or misaligned index are handled.
    *   **Inputs:**
        *   `df`: Valid OHLC DataFrame (e.g., 100 rows).
        *   `ml_signals`: A `pd.Series` that is shorter, longer, or has a non-overlapping index with `df`.
    *   **Expected Output:**
        *   The `create_candlestick_chart` function uses `ml_signals.reindex(df.index)`. This means:
            *   Signals outside the date range of `df` will be ignored.
            *   Dates in `df` not covered by `ml_signals` will have NaN signals after reindexing, so no markers.
            *   The plot should generate without error, only showing markers for valid, aligned signals.

---

### 5. Combinations

*   **Test Case 5.1: RSI, Custom Indicator, and ML Signals Active**
    *   **Description:** Test a complex chart with all major optional features enabled.
    *   **Inputs:**
        *   `df`: Valid OHLC DataFrame.
        *   `ticker_symbol`: "COMPLEX_CHART"
        *   `rsi_series`: Valid RSI data.
        *   `custom_indicators_data`: A list with one or two valid custom indicator definitions.
        *   `ml_signals`: Valid ML signal data.
    *   **Expected Output:**
        *   Returns a `plotly.graph_objects.Figure` object.
        *   The figure should contain:
            *   Main candlestick chart with ML signal markers.
            *   A subplot for RSI.
            *   One or more subplots for the custom indicator(s).
        *   All components should be rendered correctly according to their individual specifications.

---

### 6. Invalid/Empty Inputs

*   **Test Case 6.1: Empty DataFrame `df`**
    *   **Description:** Test with `df` being an empty DataFrame.
    *   **Inputs:**
        *   `df`: `pd.DataFrame()`
        *   `ticker_symbol`: "EMPTY_DF_TEST"
    *   **Expected Output:**
        *   Raises `ValueError` with a message like "DataFrame is missing required columns..." (as per current implementation's initial check).

*   **Test Case 6.2: `df` Missing Required OHLC Columns**
    *   **Description:** Test with `df` that does not have one or more of 'Open', 'High', 'Low', 'Close'.
    *   **Inputs:**
        *   `df`: `pd.DataFrame({'Open': [...], 'High': [...]})` (missing 'Low', 'Close')
        *   `ticker_symbol`: "MISSING_COLS_TEST"
    *   **Expected Output:**
        *   Raises `ValueError` with a message indicating missing columns (from the function's explicit check).

*   **Test Case 6.3: Malformed `custom_indicators_data` Entries**
    *   **Description:** Test with entries in `custom_indicators_data` that are malformed (e.g., a dictionary missing the 'name' or 'series' key for a 'simple_series' type, or missing 'data_df' for 'saty_phase_oscillator').
    *   **Inputs:**
        *   `df`: Valid OHLC DataFrame.
        *   `custom_indicators_data`: `[{'series': pd.Series(...)}]` (missing 'name')
        *   OR `[{'name': 'Bad SPO', 'plot_type': 'saty_phase_oscillator', 'params': {}}]` (missing 'data_df')
    *   **Expected Output:**
        *   The function should handle this gracefully. Observed behavior:
            *   It prints a warning message to the console (e.g., "Warning: Custom indicator ... data is missing or invalid. Skipping plot row.").
            *   The malformed indicator is skipped, and the rest of the chart (if any other valid components exist) is generated.
            *   The function does not raise an unhandled error.

*   **Test Case 6.4: `rsi_series` Not a Pandas Series**
    *   **Description:** Test providing an `rsi_series` argument that is not a `pd.Series`.
    *   **Inputs:**
        *   `df`: Valid OHLC DataFrame.
        *   `rsi_series`: `[10, 20, 30]` (a Python list)
    *   **Expected Output:**
        *   Raises `ValueError` with message "rsi_series must be a pandas Series." (from the function's explicit check).

---
