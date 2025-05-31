import pandas as pd
from typing import Union # For type hint pd.Series | pd.DataFrame

# Attempt to import the indicator calculation logic and custom indicator example
# This is for the __main__ block demonstration
try:
    from src.core.indicator_calculator import load_custom_indicators, calculate_indicator
    # We will use SimpleMovingAverage for demonstration, ensure it's discoverable
    # If src.indicators.custom_sma defines SimpleMovingAverage, load_custom_indicators should find it.
except ImportError:
    print("Warning: Could not import from src.core.indicator_calculator or src.indicators.custom_sma for __main__ demo in plot_generator.py. Create dummy versions if needed for isolated testing.")
    # Fallback for isolated testing if imports fail (e.g. if structure is not fully set up)
    calculate_indicator = None
    load_custom_indicators = None

class PlotGenerator: # Wrapping functions in a class if that's the intended structure
    """
    Handles the generation of plots and adding indicators to charts.
    This is a placeholder and will be expanded with actual plotting logic (e.g., using Plotly).
    """

    def __init__(self, ohlcv_df: pd.DataFrame):
        if not isinstance(ohlcv_df, pd.DataFrame):
            raise ValueError("OHLCV data must be a Pandas DataFrame.")
        self.ohlcv_df = ohlcv_df
        self.fig = None # Placeholder for a Plotly figure or similar
        print(f"PlotGenerator initialized with OHLCV data covering dates from {self.ohlcv_df.index.min()} to {self.ohlcv_df.index.max()}.")

    def add_indicator_trace(self, indicator_series: Union[pd.Series, pd.DataFrame], name: str = "Indicator", panel: str = "main"):
        """
        Simulates adding an indicator trace to a chart.
        In a real implementation, this would add a trace to a Plotly figure.

        Args:
            indicator_series: Pandas Series or DataFrame containing the indicator values.
            name: Name of the indicator trace.
            panel: The panel to plot on (e.g., 'main' for overlay, 'sub' for separate panel).
        """
        if not isinstance(indicator_series, (pd.Series, pd.DataFrame)):
            print(f"Error: Indicator data for '{name}' must be a Pandas Series or DataFrame. Got: {type(indicator_series)}")
            return

        if not indicator_series.index.equals(self.ohlcv_df.index):
            print(f"Warning: Index of indicator '{name}' does not perfectly match OHLCV data index. "
                  f"Indicator length: {len(indicator_series)}, OHLCV length: {len(self.ohlcv_df)}."
                  "Attempting to align by reindexing indicator to OHLCV index.")
            # This ensures that the indicator aligns with the main chart's X-axis.
            # Missing values in the indicator (e.g., due to initial calculation period) will be NaNs.
            # Extra values in the indicator not in ohlcv_df will be dropped.
            indicator_series = indicator_series.reindex(self.ohlcv_df.index)


        print(f"Plotting indicator '{name}' on panel '{panel}'.")
        if isinstance(indicator_series, pd.Series):
            print(f"Data for '{name}' (first 5 values):\n{indicator_series.head()}")
        elif isinstance(indicator_series, pd.DataFrame):
            print(f"Data for '{name}' (first 5 values):\n{indicator_series.head()}")
            # If DataFrame, it might have multiple columns (e.g. MACD with signal line)
            # A real plotting function would iterate over columns or expect specific column names.

        # In a real scenario:
        # if self.fig:
        #    for col in indicator_series.columns if isinstance(indicator_series, pd.DataFrame) else [indicator_series.name or name]:
        #        self.fig.add_trace(go.Scatter(x=self.ohlcv_df.index, y=indicator_series[col], name=col), row=..., col=...)
        print(f"Indicator '{name}' trace added (simulated).")


    def show_chart(self):
        """ Simulates showing the chart. """
        if self.fig:
            # self.fig.show()
            print("Chart display simulated.")
        else:
            print("No figure to display. Add base chart and traces first.")

if __name__ == '__main__':
    print("Demonstrating PlotGenerator and add_indicator_trace...")

    # Create a sample OHLCV DataFrame
    data = {
        'Open': [100+i for i in range(25)],
        'High': [102+i for i in range(25)],
        'Low': [98+i for i in range(25)],
        'Close': [100.5+i for i in range(25)],
        'Volume': [1000+i*10 for i in range(25)]
    }
    index = pd.date_range(start='2023-01-01', periods=25, freq='D')
    sample_ohlcv_df = pd.DataFrame(data, index=index)

    print("\n--- Initializing PlotGenerator ---")
    plot_gen = PlotGenerator(sample_ohlcv_df.copy())

    # Simulate calculating an indicator
    # For this demo, we'll use the SimpleMovingAverage if available
    indicator_output = None
    if load_custom_indicators and calculate_indicator:
        print("\n--- Loading and Calculating SMA Indicator ---")
        # Ensure src/indicators and src/core are in PYTHONPATH or use relative imports carefully
        # This assumes 'src' is a package or current working directory allows these imports.
        # For testing, it's often better to run `python -m src.charts.plot_generator` from project root.

        # Load indicators (this will print loading messages)
        # The path might need adjustment if not running from project root.
        # Assuming 'src' is in python path.
        try:
            # Adjust path if necessary, this assumes running from project root
            # For `python -m src.charts.plot_generator`, `src` is implicitly in path for `src.core`
            loaded_indicators = load_custom_indicators() # Default path "src/indicators"
            if "SimpleMovingAverage" in loaded_indicators:
                print("Calculating SimpleMovingAverage(period=5, column='Close') via calculate_indicator...")
                indicator_output = calculate_indicator(
                    sample_ohlcv_df,
                    "SimpleMovingAverage",
                    loaded_indicators,
                    period=5,
                    column='Close'
                )
                if indicator_output is not None:
                    print("SimpleMovingAverage calculated successfully.")
                else:
                    print("Failed to calculate SimpleMovingAverage.")
            else:
                print("SimpleMovingAverage not found in loaded indicators.")
                print(f"Available indicators: {list(loaded_indicators.keys())}")
        except Exception as e:
            print(f"Error during indicator loading/calculation for demo: {e}")
            print("You might need to ensure dummy_sma.py and custom_sma.py are in src/indicators/")

    else:
        print("\n--- Indicator calculation skipped (core modules not imported) ---")
        print("Creating a dummy indicator series for plot_generator demo purposes.")
        indicator_output = pd.Series([100 + i*0.5 for i in range(25)], index=index, name="DummyIndicator")


    if indicator_output is not None:
        print("\n--- Adding Indicator to Chart (Simulated) ---")
        plot_gen.add_indicator_trace(indicator_output, name=str(indicator_output.name) or "Calculated Indicator")
    else:
        print("\n--- Skipping adding indicator to chart as calculation failed or was skipped ---")

    print("\n--- Simulating Show Chart ---")
    plot_gen.show_chart() # Will say no figure, as we haven't created one.

    print("\nPlotGenerator demonstration finished.")
