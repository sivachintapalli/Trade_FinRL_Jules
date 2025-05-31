from abc import ABC, abstractmethod
import pandas as pd

class CustomIndicator(ABC):
    """
    Abstract base class for custom technical indicators.

    Users should subclass this class to create their own custom indicators.
    Parameters for the indicator can be passed during instantiation.
    """

    def __init__(self, **kwargs):
        """
        Initializes the custom indicator with given parameters.
        Parameters are stored as attributes of the instance.

        Args:
            **kwargs: Arbitrary keyword arguments representing indicator parameters.
                      These will be set as attributes of the instance.
                      Example: CustomIndicator(period=20) will set self.period = 20.
        """
        for key, value in kwargs.items():
            setattr(self, key, value)

    @abstractmethod
    def calculate(self, data_df: pd.DataFrame) -> pd.Series | pd.DataFrame:
        """
        Calculate the indicator values.

        This method must be implemented by subclasses.

        Args:
            data_df: Pandas DataFrame with OHLCV data.
                       It is expected to have columns like 'Open', 'High', 'Low', 'Close', 'Volume'.

        Returns:
            A Pandas Series or DataFrame containing the calculated indicator values.
            The index should align with the input data_df's index.
        """
        pass

    def get_params(self) -> dict:
        """
        Returns the parameters of the indicator.

        Returns:
            A dictionary of the indicator's parameters.
        """
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}

    def __repr__(self):
        """
        String representation of the indicator, showing its name and parameters.
        """
        params_str = ', '.join(f'{k}={v}' for k, v in self.get_params().items())
        return f"{self.__class__.__name__}({params_str})"

if __name__ == '__main__':
    # Example of how a user might define a subclass (for testing purposes)
    class MySimpleMovingAverage(CustomIndicator):
        def __init__(self, period: int = 20, column: str = 'Close'):
            super().__init__(period=period, column=column)

        def calculate(self, data_df: pd.DataFrame) -> pd.Series:
            if self.column not in data_df.columns:
                raise ValueError(f"Column '{self.column}' not found in DataFrame.")
            if not isinstance(self.period, int) or self.period <= 0:
                raise ValueError("Period must be a positive integer.")

            return data_df[self.column].rolling(window=self.period).mean()

    # Example usage:
    # Create a dummy DataFrame
    data = {
        'Open': [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
        'High': [11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21],
        'Low': [9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
        'Close': [10.5, 11.5, 12.5, 13.5, 14.5, 15.5, 16.5, 17.5, 18.5, 19.5, 20.5],
        'Volume': [100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200]
    }
    dummy_df = pd.DataFrame(data, index=pd.to_datetime([f'2023-01-{i:02d}' for i in range(1, 12)]))

    try:
        sma_indicator = MySimpleMovingAverage(period=5, column='Close')
        print(f"Instantiated Indicator: {sma_indicator}")
        sma_values = sma_indicator.calculate(dummy_df)
        print("\nSMA Values:")
        print(sma_values)

        # Test with a different column
        sma_high = MySimpleMovingAverage(period=3, column='High')
        print(f"\nInstantiated Indicator: {sma_high}")
        sma_high_values = sma_high.calculate(dummy_df)
        print("\nSMA High Values:")
        print(sma_high_values)

        # Test error case: invalid column
        # sma_error = MySimpleMovingAverage(column='Invalid')
        # sma_error.calculate(dummy_df)

    except Exception as e:
        print(f"An error occurred: {e}")
