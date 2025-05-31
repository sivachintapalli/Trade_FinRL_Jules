import pandas as pd
from src.core.custom_indicator_interface import CustomIndicator

class MyDummySMA(CustomIndicator):
    def __init__(self, period: int = 10, column: str = 'Close'):
        super().__init__(period=period, column=column)

    def calculate(self, data_df: pd.DataFrame) -> pd.Series:
        if self.column not in data_df.columns:
            raise ValueError(f"Column '{self.column}' not found in DataFrame.")
        if not isinstance(self.period, int) or self.period <= 0:
            raise ValueError("Period must be a positive integer.")
        return data_df[self.column].rolling(window=self.period).mean()
