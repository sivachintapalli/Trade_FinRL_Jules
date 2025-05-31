"""
Custom Technical Indicator Interface Module.

This module defines the abstract base class `CustomIndicator`, which serves as
the foundational interface for all custom technical indicators integrated into
the trading analysis platform. By subclassing `CustomIndicator` and implementing
its abstract methods, developers can create new indicators that can be
dynamically discovered, configured, and utilized by the application.
"""
from abc import ABC, abstractmethod
import pandas as pd

class CustomIndicator(ABC):
    """
    Abstract Base Class for creating custom technical indicators.

    Purpose:
        To provide a standardized structure for all custom technical indicators,
        ensuring they can be seamlessly integrated into the analysis platform.
        Subclasses will be discoverable by the `IndicatorManager` and their
        parameters will be configurable via the Streamlit UI.

    How to Inherit:
        1. Create a new Python class that inherits from `CustomIndicator`.
        2. Implement the `__init__` method:
           - Define parameters your indicator requires (e.g., `period`, `column_name`).
           - **Crucially, use type hints for these parameters** (e.g., `period: int`,
             `column: str`). The Streamlit UI uses these hints to generate
             appropriate input fields.
           - Call `super().__init__(**kwargs)` within your subclass's `__init__`,
             passing all parameters as keyword arguments. This allows the base
             class to automatically store these parameters as instance attributes.
        3. Implement the `calculate` abstract method:
           - This method will contain the core logic for computing your indicator.
           - It receives a pandas DataFrame of financial data.
           - It must return a pandas Series or DataFrame with the calculated
             indicator values, ensuring the index aligns with the input DataFrame.

    Example of subclass `__init__`:
    ```python
    class MyIndicator(CustomIndicator):
        def __init__(self, period: int = 20, source_column: str = 'Close'):
            # Parameters 'period' and 'source_column' will be configurable in the UI.
            # Type hints (int, str) guide the UI element generation.
            super().__init__(period=period, source_column=source_column)
            # You can add more initialization logic here if needed,
            # but parameters for UI should be in __init__ signature and passed to super.
    ```

    Parameter Storage:
        Parameters passed to `super().__init__(**kwargs)` are automatically set as
        instance attributes. For example, if `MyIndicator(period=10)` is called,
        `self.period` will be `10` within the instance.
    """

    def __init__(self, **kwargs):
        """
        Initializes the custom indicator and stores its parameters.

        This constructor is called by subclasses. The keyword arguments (`kwargs`)
        passed to it (which should correspond to the parameters defined in the
        subclass's `__init__` signature) are automatically set as attributes
        of the instance. This allows for generic parameter handling and retrieval.

        Args:
            **kwargs: Arbitrary keyword arguments representing the indicator's
                      parameters. For example, if a subclass calls
                      `super().__init__(period=20, column='Close')`, then
                      `self.period` will be set to `20` and `self.column`
                      to `'Close'`.
        """
        for key, value in kwargs.items():
            setattr(self, key, value)

    @abstractmethod
    def calculate(self, data_df: pd.DataFrame) -> pd.Series | pd.DataFrame:
        """
        Abstract method to calculate the indicator's values.

        This method **must be implemented by all subclasses**. It defines the
        core logic of the technical indicator.

        Args:
            data_df (pd.DataFrame): A pandas DataFrame containing the input financial
                                    data, typically including columns like 'Open',
                                    'High', 'Low', 'Close', and 'Volume'. The
                                    DataFrame is expected to have a DatetimeIndex.

        Returns:
            pd.Series | pd.DataFrame: A pandas Series or DataFrame containing the
                                      calculated indicator values. The index of the
                                      returned Series/DataFrame should align with the
                                      index of the input `data_df` to ensure proper
                                      plotting and analysis. If the indicator produces
                                      multiple output lines (e.g., MACD with signal line),
                                      a DataFrame can be returned with appropriately
                                      named columns.
        """
        pass

    def get_params(self) -> dict:
        """
        Retrieves the parameters with which the indicator instance was configured.

        This method is useful for inspecting the indicator's settings or for
        display purposes. It returns a dictionary of parameter names and their
        values, excluding any internal attributes (those starting with an underscore).

        Returns:
            dict: A dictionary where keys are parameter names (str) and values are
                  their corresponding values.
        """
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}

    def __repr__(self):
        """
        Provides an unambiguous string representation of the indicator instance,
        including its class name and configured parameters.

        Example: `SimpleMovingAverage(period=20, column='Close')`
        """
        params_str = ', '.join(f'{k}={v!r}' for k, v in self.get_params().items()) # Use !r for repr of values
        return f"{self.__class__.__name__}({params_str})"

if __name__ == '__main__':
    # Example of how a user might define and use a subclass (for testing/demonstration)
    class MySimpleMovingAverage(CustomIndicator):
        """
        A simple moving average indicator example.
        """
        def __init__(self, period: int = 20, column: str = 'Close'):
            """
            Initializes the Simple Moving Average indicator.

            Args:
                period (int): The lookback period for the SMA.
                column (str): The DataFrame column on which to calculate the SMA.
            """
            super().__init__(period=period, column=column)
            # self.period and self.column are now set by the super().__init__ call

        def calculate(self, data_df: pd.DataFrame) -> pd.Series:
            """
            Calculates the Simple Moving Average.
            """
            if self.column not in data_df.columns:
                raise ValueError(f"Column '{self.column}' not found in DataFrame. Available: {data_df.columns.tolist()}")
            if not isinstance(self.period, int) or self.period <= 0:
                raise ValueError(f"Period must be a positive integer, got {self.period}")

            # Name the series for better identification in plots/dataframes
            series_name = f"SMA_{self.column}_{self.period}"
            return data_df[self.column].rolling(window=self.period).mean().rename(series_name)

    # Example usage:
    data = {
        'Open': [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
        'High': [11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21],
        'Low': [9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
        'Close': [10.5, 11.5, 12.5, 13.5, 14.5, 15.5, 16.5, 17.5, 18.5, 19.5, 20.5],
        'Volume': [100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200]
    }
    dummy_df = pd.DataFrame(data, index=pd.to_datetime([f'2023-01-{i:02d}' for i in range(1, 12)]))
    dummy_df.index.name = "Date"

    print("--- Testing CustomIndicator Interface ---")
    try:
        # Instantiate with default parameters (defined in MySimpleMovingAverage)
        sma_default = MySimpleMovingAverage()
        print(f"Instantiated Indicator (default): {sma_default}") # Uses __repr__
        print(f"Parameters (default): {sma_default.get_params()}")
        sma_default_values = sma_default.calculate(dummy_df)
        print("\nSMA Values (default period 20 on 'Close'):")
        print(sma_default_values)

        # Instantiate with custom parameters
        sma_custom = MySimpleMovingAverage(period=5, column='High')
        print(f"\nInstantiated Indicator (custom): {sma_custom}")
        print(f"Parameters (custom): {sma_custom.get_params()}")
        sma_custom_values = sma_custom.calculate(dummy_df)
        print("\nSMA Values (custom period 5 on 'High'):")
        print(sma_custom_values)

        # Test error case: invalid column (should be caught by MySimpleMovingAverage.calculate)
        print("\n--- Testing Error Case (Invalid Column) ---")
        try:
            sma_error_col = MySimpleMovingAverage(column='InvalidColumn')
            sma_error_col.calculate(dummy_df)
        except ValueError as ve:
            print(f"Caught expected error: {ve}")

        # Test error case: invalid period (should be caught by MySimpleMovingAverage.calculate)
        print("\n--- Testing Error Case (Invalid Period) ---")
        try:
            sma_error_period = MySimpleMovingAverage(period=0) # Period must be positive
            sma_error_period.calculate(dummy_df)
        except ValueError as ve:
            print(f"Caught expected error: {ve}")

    except Exception as e:
        print(f"An unexpected error occurred during testing: {e}")

    print("\n--- CustomIndicator Interface Test Complete ---")
