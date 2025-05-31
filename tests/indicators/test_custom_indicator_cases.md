# Test Cases for Custom Technical Indicators

This document outlines test cases for custom technical indicators developed by inheriting from the `CustomIndicator` base class, as well as specific test cases for the `SimpleMovingAverage` indicator found in `src/indicators/custom_sma.py`.

---

## Part 1: General Test Cases for Custom Indicators

These test cases apply to any custom indicator class that inherits from `src.core.custom_indicator_interface.CustomIndicator`.

### 1.1. Initialization (`__init__`)

*   **Test Case 1.1.1: Successful Instantiation with Valid Parameters**
    *   **Description:** Verify that a custom indicator can be instantiated successfully when provided with valid parameters as defined by its `__init__` method.
    *   **Inputs:** For a hypothetical `MyCustomIndicator(param1: int, param2: str = 'default')`:
        *   `MyCustomIndicator(param1=10, param2='test')`
        *   `MyCustomIndicator(param1=5)` (using default for `param2`)
    *   **Expected Output:**
        *   Object is created successfully without errors.
        *   Instance attributes `self.param1` and `self.param2` are set to the provided values (or defaults).

*   **Test Case 1.1.2: Parameters Correctly Stored**
    *   **Description:** Ensure that parameters passed during instantiation are correctly stored and accessible.
    *   **Inputs:** Instantiate an indicator with known parameters.
    *   **Expected Output:**
        *   Parameters are accessible as instance attributes (e.g., `indicator_instance.param_name`).
        *   `indicator_instance.get_params()` returns a dictionary containing these parameters and their values.

*   **Test Case 1.1.3: Instantiation with Invalid Parameter Types**
    *   **Description:** Test instantiation when a parameter of an incorrect type is provided (e.g., a string where an integer is expected by type hint and internal logic).
    *   **Inputs:** For `MyCustomIndicator(param1: int, ...)`:
        *   `MyCustomIndicator(param1='not_an_int')`
    *   **Expected Output:**
        *   Should raise `TypeError` if type checking is strictly enforced by the indicator's `__init__` before `super().__init__`.
        *   If validation happens after `super().__init__` (where attributes are set), it might be a `ValueError` or other custom error depending on the indicator's validation logic.
        *   If no explicit validation, behavior depends on how the parameter is used. The goal is to ensure type hints are respected or validated.

*   **Test Case 1.1.4: Instantiation with Invalid Parameter Values**
    *   **Description:** Test instantiation when a parameter has a valid type but an invalid value according to the indicator's logic (e.g., a negative period for a moving average).
    *   **Inputs:** For an indicator like `SMA(period: int)`:
        *   `SMA(period=-5)`
        *   `SMA(period=0)`
    *   **Expected Output:**
        *   Should raise `ValueError` (or a custom domain-specific error) from the indicator's own validation logic within `__init__`.

*   **Test Case 1.1.5: Instantiation with Missing Required Parameters**
    *   **Description:** Test instantiation when a required parameter (one without a default value in `__init__`) is not provided.
    *   **Inputs:** For `MyCustomIndicator(required_param, optional_param='default')`:
        *   `MyCustomIndicator(optional_param='value')` (missing `required_param`)
    *   **Expected Output:**
        *   Raises `TypeError` (e.g., `__init__() missing 1 required positional argument: 'required_param'`).

*   **Test Case 1.1.6: Type Hinting for Streamlit UI (Conceptual)**
    *   **Description:** Note the importance of type hints in the indicator's `__init__` signature. While not a direct functional test case for `data_manager.py`, it's a critical design aspect for UI generation.
    *   **Concept:** The Streamlit UI in the main application relies on `inspect.signature()` and parameter type hints (`param.annotation`) from the indicator's `__init__` method to dynamically generate appropriate input widgets (e.g., `st.number_input` for `int` or `float`, `st.selectbox` for `bool`, `st.text_input` for `str`).
    *   **Expected Behavior (Design Principle):** Custom indicators should use clear and accurate type hints for all parameters intended to be configurable in the UI.

### 1.2. Calculation (`calculate(self, data_df: pd.DataFrame)`)

*   **Test Case 1.2.1: Calculation with Valid `data_df`**
    *   **Description:** Test the `calculate` method with a typical, valid `pd.DataFrame` (e.g., OHLCV data).
    *   **Inputs:**
        *   `data_df`: A `pandas.DataFrame` with appropriate columns (e.g., 'Open', 'High', 'Low', 'Close', 'Volume') and a `DatetimeIndex`.
    *   **Expected Output:**
        *   Returns a `pandas.Series` or `pandas.DataFrame` containing the calculated indicator values.
        *   The index of the output Series/DataFrame should align with the index of the input `data_df`.
        *   Values should be of a numeric type (float or int).

*   **Test Case 1.2.2: `data_df` Missing a Required Column**
    *   **Description:** Test `calculate` when `data_df` is missing a column that the indicator expects (e.g., the 'source_column' specified in `__init__`).
    *   **Inputs:**
        *   Indicator initialized with `source_column='Close'`.
        *   `data_df`: A DataFrame that does *not* contain a 'Close' column.
    *   **Expected Output:**
        *   Raises `ValueError` or `KeyError` from within the `calculate` method when it tries to access the missing column.

*   **Test Case 1.2.3: Empty `data_df`**
    *   **Description:** Test `calculate` with an empty `pd.DataFrame`.
    *   **Inputs:**
        *   `data_df`: `pd.DataFrame()`
    *   **Expected Output:**
        *   The method should handle this gracefully. Common behaviors:
            *   Return an empty `pd.Series` or `pd.DataFrame` with a compatible dtype.
            *   Raise a specific `ValueError` indicating that input data is empty.
        *   The exact behavior should be documented by the specific indicator if not returning an empty Series/DataFrame.

*   **Test Case 1.2.4: `data_df` Containing NaNs in Source Columns**
    *   **Description:** Test `calculate` when the source column(s) in `data_df` contain `NaN` values.
    *   **Inputs:**
        *   `data_df`: A DataFrame where the column(s) used by the indicator have some `NaN` entries.
    *   **Expected Output:**
        *   The indicator calculation should handle NaNs robustly. This typically means:
            *   NaNs in the input propagate to the output for the affected calculation windows.
            *   Some indicators might have specific logic to fill or ignore NaNs, which should be documented.
        *   The method should not crash due to unexpected NaNs.

*   **Test Case 1.2.5: Output Series/DataFrame Naming Convention**
    *   **Description:** Verify that the output `pd.Series` (or columns of `pd.DataFrame`) from `calculate` has a descriptive name.
    *   **Inputs:** Any valid `data_df`.
    *   **Expected Output:**
        *   If a Series is returned, `output_series.name` should be descriptive (e.g., "SMA_20_Close", "MyIndicator_CustomName").
        *   If a DataFrame is returned, its columns should have meaningful names.

*   **Test Case 1.2.6: `data_df` of Various Lengths**
    *   **Description:** Test `calculate` with `data_df` of lengths shorter than, equal to, and longer than any lookback period defined by the indicator.
    *   **Inputs:**
        *   Indicator with a lookback period `L` (e.g., `SMA(period=L)`).
        *   `data_df` of length `N < L`.
        *   `data_df` of length `N = L`.
        *   `data_df` of length `N > L`.
    *   **Expected Output:**
        *   Correct handling of initial `NaN` values in the output due to the lookback period.
            *   If `N < L`, the output might be all NaNs or partially NaNs depending on `min_periods` logic.
            *   If `N = L`, the first `L-1` (or `L` depending on `diff` usage) values are typically NaN.
            *   If `N > L`, initial values are NaN, followed by calculated values.

### 1.3. `get_params(self)` Method

*   **Test Case 1.3.1: Returns a Dictionary**
    *   **Description:** Verify that `get_params()` returns a dictionary.
    *   **Inputs:** Call `get_params()` on an instantiated indicator.
    *   **Expected Output:** The return type is `dict`.

*   **Test Case 1.3.2: Dictionary Contains Correct Parameters and Values**
    *   **Description:** Ensure the dictionary from `get_params()` accurately reflects the parameters and their values with which the indicator was initialized.
    *   **Inputs:** Instantiate an indicator with specific parameters (e.g., `MyIndicator(param1=10, param2='test')`).
    *   **Expected Output:**
        *   `get_params()` returns `{'param1': 10, 'param2': 'test'}` (or similar, excluding attributes starting with `_`).

---

## Part 2: Specific Test Cases for `SimpleMovingAverage`

### Module: `custom_sma.py`
### Class: `SimpleMovingAverage(period: int, column: str)`

### 2.1. Initialization (`__init__`)

*   **Test Case 2.1.1: Valid `period` and `column`**
    *   **Description:** Instantiate `SimpleMovingAverage` with valid parameters.
    *   **Inputs:** `period=20`, `column='Close'`
    *   **Expected Output:**
        *   Successful instantiation.
        *   `self.period` is 20.
        *   `self.column` is 'Close'.
        *   `get_params()` returns `{'period': 20, 'column': 'Close'}`.

*   **Test Case 2.1.2: Invalid `period` (Zero or Negative)**
    *   **Description:** Test instantiation with `period = 0` or a negative integer.
    *   **Inputs:**
        *   `period=0`, `column='Close'`
        *   `period=-5`, `column='Close'`
    *   **Expected Output:**
        *   Raises `ValueError` with a message like "SMA 'period' must be a positive integer." (due to validation in `SimpleMovingAverage.__init__`).

*   **Test Case 2.1.3: Invalid `column` (Empty or Non-String)**
    *   **Description:** Test instantiation with an empty string or a non-string type for `column`.
    *   **Inputs:**
        *   `period=20`, `column=""` (empty string)
        *   `period=20`, `column=123` (non-string)
    *   **Expected Output:**
        *   For `column=""`: Raises `ValueError` ("SMA 'column' name must be a non-empty string.").
        *   For `column=123`: Raises `ValueError` ("SMA 'column' name must be a non-empty string.") because the check is `isinstance(self.column, str)`. If it were `TypeError`, that would also be acceptable if the check was different.

### 2.2. Calculation (`calculate(self, data_df: pd.DataFrame)`)

*Let `data_df` be a sample DataFrame with 'Date' index and columns 'Open', 'High', 'Low', 'Close', 'Volume'.*

*   **Test Case 2.2.1: Standard Calculation**
    *   **Description:** Calculate SMA with typical valid inputs.
    *   **Inputs:**
        *   `indicator = SimpleMovingAverage(period=5, column='Close')`
        *   `data_df`: A DataFrame with at least 10 rows of 'Close' prices.
    *   **Expected Output:**
        *   Returns a `pd.Series` of SMA values.
        *   The first 4 values of the Series are `NaN`.
        *   The 5th value (index 4) is the mean of `data_df['Close'][0:5]`.
        *   The 6th value (index 5) is the mean of `data_df['Close'][1:6]`, and so on.
        *   The `name` attribute of the returned Series is `SMA_5_Close`.
        *   Compare a few calculated values with manual calculation or a trusted library like `pandas.Series.rolling(window=5).mean()`.

*   **Test Case 2.2.2: `period` Larger than `len(data_df)`**
    *   **Description:** Test when the lookback `period` is greater than the length of the input DataFrame.
    *   **Inputs:**
        *   `indicator = SimpleMovingAverage(period=20, column='Close')`
        *   `data_df`: A DataFrame with 10 rows of 'Close' prices.
    *   **Expected Output:**
        *   All values in the returned `pd.Series` are `NaN`.
        *   The Series name is `SMA_20_Close`.

*   **Test Case 2.2.3: Different Valid `column`**
    *   **Description:** Calculate SMA using a different valid column from `data_df`, like 'Volume'.
    *   **Inputs:**
        *   `indicator = SimpleMovingAverage(period=10, column='Volume')`
        *   `data_df`: A DataFrame with 'Volume' data.
    *   **Expected Output:**
        *   Returns a `pd.Series` of SMA values calculated based on the 'Volume' column.
        *   The `name` attribute of the returned Series is `SMA_10_Volume`.
        *   Initial `period-1` (i.e., 9) values are `NaN`.

*   **Test Case 2.2.4: `column` Not Present in `data_df`**
    *   **Description:** Test `calculate` when the specified `column` for SMA does not exist in `data_df`.
    *   **Inputs:**
        *   `indicator = SimpleMovingAverage(period=5, column='NonExistentColumn')`
        *   `data_df`: Standard OHLCV DataFrame.
    *   **Expected Output:**
        *   Raises `ValueError` with a message indicating the column was not found (as per `SimpleMovingAverage.calculate` implementation).

*   **Test Case 2.2.5: `data_df` with NaNs in Target `column`**
    *   **Description:** Test `calculate` when the target column in `data_df` contains `NaN` values.
    *   **Inputs:**
        *   `indicator = SimpleMovingAverage(period=3, column='Close')`
        *   `data_df['Close']`: `pd.Series([10, 11, np.nan, 13, 14, 15])`
    *   **Expected Output:**
        *   The pandas `rolling().mean()` function, by default, propagates NaNs. If a window includes a NaN, the result for that window is NaN.
        *   Expected SMA Series (for period 3):
            *   Index 0: NaN (lookback)
            *   Index 1: NaN (lookback)
            *   Index 2: NaN (due to `data_df['Close'][2]` being NaN, affecting window `[data_df['Close'][0:3]]`)
            *   Index 3: NaN (due to `data_df['Close'][2]` being NaN, affecting window `[data_df['Close'][1:4]]`)
            *   Index 4: NaN (due to `data_df['Close'][2]` being NaN, affecting window `[data_df['Close'][2:5]]`)
            *   Index 5: `(13 + 14 + 15) / 3 = 14.0`
        *   The output Series name should be `SMA_3_Close`.

---
