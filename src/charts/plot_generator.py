import plotly.graph_objects as go
import pandas as pd
from plotly.subplots import make_subplots # Import make_subplots

def create_candlestick_chart(df: pd.DataFrame, ticker_symbol: str = "SPY",
                             rsi_series: pd.Series = None, rsi_period: int = 14,
                             overbought_level: int = 70, oversold_level: int = 30):
    """
    Generates a Plotly candlestick chart from a DataFrame, optionally including an RSI subplot.

    Args:
        df (pd.DataFrame): DataFrame with 'Open', 'High', 'Low', 'Close' columns
                           and a DatetimeIndex.
        ticker_symbol (str): Name of the ticker for chart title.
        rsi_series (pd.Series, optional): Pandas Series containing RSI values, indexed by Date.
        rsi_period (int): The period used for RSI calculation (for labeling).
        overbought_level (int): The RSI overbought threshold.
        oversold_level (int): The RSI oversold threshold.

    Returns:
        plotly.graph_objects.Figure: Plotly figure object.
    """

    # Ensure required columns exist in the main DataFrame
    required_cols = ['Open', 'High', 'Low', 'Close']
    if not all(col in df.columns for col in required_cols):
        missing_cols = [col for col in required_cols if col not in df.columns]
        raise ValueError(f"DataFrame is missing required columns for candlestick: {missing_cols}")

    if rsi_series is not None:
        # Create a figure with subplots: 1 for candlestick, 1 for RSI
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                           row_heights=[0.7, 0.3],  # Candlestick gets 70% height, RSI 30%
                           vertical_spacing=0.05) # Space between subplots
    else:
        fig = go.Figure()

    # Candlestick Trace
    fig.add_trace(go.Candlestick(x=df.index,
                               open=df['Open'],
                               high=df['High'],
                               low=df['Low'],
                               close=df['Close'],
                               name=f'{ticker_symbol} OHLC'),
                  row=1 if rsi_series is not None else None,
                  col=1 if rsi_series is not None else None)

    # RSI Trace (if provided)
    if rsi_series is not None:
        if not isinstance(rsi_series, pd.Series):
            raise ValueError("rsi_series must be a pandas Series.")
        if not rsi_series.empty:
            # Align rsi_series with df.index to ensure correct plotting
            aligned_rsi = rsi_series.reindex(df.index)

            fig.add_trace(go.Scatter(x=aligned_rsi.index, y=aligned_rsi,
                                     mode='lines', name=f'RSI ({rsi_period})',
                                     line=dict(color='cyan')),
                          row=2, col=1)

            # Add overbought and oversold lines for RSI
            fig.add_hline(y=overbought_level, line_dash="dash", line_color="red",
                          annotation_text=f"Overbought ({overbought_level})", annotation_position="bottom right",
                          row=2, col=1)
            fig.add_hline(y=oversold_level, line_dash="dash", line_color="green",
                          annotation_text=f"Oversold ({oversold_level})", annotation_position="top right",
                          row=2, col=1)
            fig.update_yaxes(title_text="RSI", range=[0, 100], row=2, col=1) # Set RSI y-axis range

    # Update layout
    fig.update_layout(
        title=f'{ticker_symbol} Candlestick Chart' + (' with RSI' if rsi_series is not None else ''),
        xaxis_title='Date',
        yaxis_title='Price', # Main y-axis title for candlestick
        xaxis_rangeslider_visible=False,
        template="plotly_dark",
        height=600 if rsi_series is None else 800 # Adjust height if RSI is present
    )

    # Style adjustments for axes
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)', row=1, col=1) # Apply to candlestick y-axis


    # Hide the x-axis title for the top plot if there's an RSI plot below
    if rsi_series is not None:
        fig.update_xaxes(title_text="", row=1, col=1)


    return fig

if __name__ == '__main__':
    import sys
    import os
    # Add the project root to the Python path
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    sample_df = None
    try:
        from src.core.data_manager import fetch_spy_data
        print("Fetching sample SPY data for chart generation test...")
        sample_df = fetch_spy_data(period="6mo") # Fetch 6 months for better RSI
    except ImportError as e:
        print(f"Could not import data_manager: {e}. Using dummy data for chart generation test.")
        # Use a slightly larger dummy dataset for RSI
        dates_rng = pd.date_range(start='2023-01-01', periods=60, freq='B') # Approx 3 months of business days
        data = {
            'Open': [100 + i + (i%5) for i in range(60)],
            'High': [100 + i + (i%5) + 2 for i in range(60)],
            'Low': [100 + i + (i%5) - 2 for i in range(60)],
            'Close': [100 + i + (i%5) + 1 for i in range(60)],
        }
        sample_df = pd.DataFrame(data, index=pd.Index(dates_rng, name="Date"))


    if sample_df is not None and not sample_df.empty:
        print("\nSample DataFrame Head for Charting:")
        print(sample_df.head())

        # Test 1: Candlestick chart without RSI
        print("\nCreating candlestick chart WITHOUT RSI...")
        fig_no_rsi = create_candlestick_chart(sample_df, ticker_symbol="TEST (No RSI)")
        print("Figure created. If interactive, call fig_no_rsi.show()")
        # fig_no_rsi.show() # Uncomment to display if in interactive environment

        # Test 2: Candlestick chart WITH RSI
        print("\nCreating candlestick chart WITH RSI...")
        rsi_values = None
        try:
            from src.core.indicator_calculator import calculate_rsi
            if 'Close' in sample_df.columns:
                rsi_values = calculate_rsi(sample_df['Close'], period=14)
                print("RSI calculated.")
            else:
                print("Warning: 'Close' column not found in sample_df for RSI calculation.")
        except ImportError as e:
            print(f"Could not import calculate_rsi: {e}. Cannot test RSI integration.")

        fig_with_rsi = create_candlestick_chart(sample_df, ticker_symbol="TEST (With RSI)",
                                                rsi_series=rsi_values, rsi_period=14)
        print("Figure with RSI created. If interactive, call fig_with_rsi.show()")
        # fig_with_rsi.show() # Uncomment to display if in interactive environment

    else:
        print("Failed to get sample data for chart testing.")
