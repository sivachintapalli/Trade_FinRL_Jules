import plotly.graph_objects as go
import pandas as pd
from plotly.subplots import make_subplots # Import make_subplots

def create_candlestick_chart(df: pd.DataFrame, ticker_symbol: str = "SPY",
                             rsi_series: pd.Series = None, rsi_period: int = 14,
                             overbought_level: int = 70, oversold_level: int = 30,
                             custom_indicators_data: list = None,
                             ml_signals: pd.Series = None):
    """
    Generates a Plotly candlestick chart from a DataFrame, optionally including RSI,
    custom indicators, and ML-generated buy/sell signals.

    Args:
        df (pd.DataFrame): DataFrame with 'Open', 'High', 'Low', 'Close' columns
                           and a DatetimeIndex.
        ticker_symbol (str): Name of the ticker for chart title.
        rsi_series (pd.Series, optional): Pandas Series containing RSI values.
        rsi_period (int): The period used for RSI calculation (for labeling).
        overbought_level (int): The RSI overbought threshold.
        oversold_level (int): The RSI oversold threshold.
        custom_indicators_data (list, optional): A list of dictionaries, where each
            dictionary contains 'name' (str) and 'series' (pd.Series) for a custom indicator.
            Example: [{'name': 'SMA_20', 'series': pd.Series(...) }, ...]
        ml_signals (pd.Series, optional): Pandas Series containing ML-generated signals
            (e.g., 1 for buy, 0 for sell/hold), indexed by Date.

    Returns:
        plotly.graph_objects.Figure: Plotly figure object.
    """

    # Ensure required columns exist in the main DataFrame
    required_cols = ['Open', 'High', 'Low', 'Close']
    if not all(col in df.columns for col in required_cols):
        missing_cols = [col for col in required_cols if col not in df.columns]
        raise ValueError(f"DataFrame is missing required columns for candlestick: {missing_cols}")

    num_custom_indicators = len(custom_indicators_data) if custom_indicators_data else 0
    num_rows = 1 + (1 if rsi_series is not None else 0) + num_custom_indicators

    row_heights = [0.6] # Main candlestick plot height
    if rsi_series is not None:
        row_heights.append(0.2) # RSI plot height
    if num_custom_indicators > 0:
        custom_indicator_height = 0.2 / num_custom_indicators if num_custom_indicators <=2 else 0.15 # smaller if many
        row_heights.extend([custom_indicator_height] * num_custom_indicators)

    # Normalize row_heights if they don't sum to 1 (or close to it)
    # This is a simple normalization, more complex logic might be needed if sum is far off
    total_height_ratio = sum(row_heights)
    if total_height_ratio > 1.5 and num_rows > 1 : # Heuristic, if sum is too large and we have multiple plots
         row_heights = [h / total_height_ratio for h in row_heights]


    fig = make_subplots(rows=num_rows, cols=1, shared_xaxes=True,
                        row_heights=row_heights if num_rows > 1 else None,
                        vertical_spacing=0.03)

    current_row = 1

    # Candlestick Trace
    fig.add_trace(go.Candlestick(x=df.index,
                               open=df['Open'],
                               high=df['High'],
                               low=df['Low'],
                               close=df['Close'],
                               name=f'{ticker_symbol} OHLC'),
                  row=current_row, col=1)

    # Add ML Signals to the candlestick plot if provided
    if ml_signals is not None and isinstance(ml_signals, pd.Series) and not ml_signals.empty:
        aligned_signals = ml_signals.reindex(df.index) # Align signals with main df index

        buy_signals = df.loc[aligned_signals == 1]
        sell_signals = df.loc[aligned_signals == 0] # Assuming 0 is sell/hold for now

        if not buy_signals.empty:
            fig.add_trace(go.Scatter(
                x=buy_signals.index,
                y=buy_signals['Low'] * 0.98, # Place marker below the low
                mode='markers',
                marker_symbol='triangle-up',
                marker_size=10,
                marker_color='green',
                name='Buy Signal'
            ), row=current_row, col=1)

        if not sell_signals.empty:
            fig.add_trace(go.Scatter(
                x=sell_signals.index,
                y=sell_signals['High'] * 1.02, # Place marker above the high
                mode='markers',
                marker_symbol='triangle-down',
                marker_size=10,
                marker_color='red',
                name='Sell Signal'
            ), row=current_row, col=1)

    fig.update_yaxes(title_text="Price", row=current_row, col=1)
    if num_rows > 1: # If there are subplots below, hide x-axis title for this row
        fig.update_xaxes(title_text="", row=current_row, col=1)
    current_row += 1

    # RSI Trace (if provided)
    if rsi_series is not None:
        if not isinstance(rsi_series, pd.Series):
            raise ValueError("rsi_series must be a pandas Series.")
        if not rsi_series.empty:
            aligned_rsi = rsi_series.reindex(df.index)
            fig.add_trace(go.Scatter(x=aligned_rsi.index, y=aligned_rsi,
                                     mode='lines', name=f'RSI ({rsi_period})',
                                     line=dict(color='cyan')),
                          row=current_row, col=1)
            fig.add_hline(y=overbought_level, line_dash="dash", line_color="red", row=current_row, col=1)
            fig.add_hline(y=oversold_level, line_dash="dash", line_color="green", row=current_row, col=1)
            fig.update_yaxes(title_text="RSI", range=[0, 100], row=current_row, col=1)
            if num_rows > current_row: # If there are more subplots below
                 fig.update_xaxes(title_text="", row=current_row, col=1)
            current_row += 1

    # Custom Indicators Traces
    if custom_indicators_data:
        for indicator in custom_indicators_data:
            name = indicator.get('name', f'CustomInd_{current_row-1}')
            series = indicator.get('series')
            if isinstance(series, pd.Series) and not series.empty:
                aligned_series = series.reindex(df.index)
                fig.add_trace(go.Scatter(x=aligned_series.index, y=aligned_series,
                                         mode='lines', name=name,
                                         connectgaps=True), # Connect gaps for indicators like SMA
                              row=current_row, col=1)
                fig.update_yaxes(title_text=name[:15], row=current_row, col=1) # Shorten long names for y-axis title
                if num_rows > current_row: # If there are more subplots below
                    fig.update_xaxes(title_text="", row=current_row, col=1)
                current_row += 1
            else:
                print(f"Warning: Custom indicator '{name}' data is missing or not a Series. Skipping.")


    chart_title = f'{ticker_symbol} Candlestick Chart'
    if rsi_series is not None: chart_title += ' with RSI'
    if num_custom_indicators > 0: chart_title += f' & {num_custom_indicators} Custom Indicator(s)'

    fig.update_layout(
        title=chart_title,
        xaxis_title='Date' if num_rows == 1 else '', # Only show main x-axis title if no subplots or it's the last one
        xaxis_rangeslider_visible=False,
        template="plotly_dark",
        height=350 + (num_rows * 150) # Base height + per_row_height
    )

    # Ensure the bottom-most x-axis has the title "Date"
    if num_rows > 1:
        fig.update_xaxes(title_text="Date", row=num_rows, col=1)


    # General style adjustments for all axes
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)')
    for i in range(1, num_rows + 1):
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)', row=i, col=1)

    return fig

if __name__ == '__main__':
    import sys
    import os
    # Add the project root to the Python path
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    sample_df = None
    rsi_values = None
    custom_sma_series = None
    custom_ema_series = None
    dummy_ml_signals = None

    try:
        from src.core.data_manager import fetch_spy_data
        print("Fetching sample SPY data for chart generation test...")
        sample_df = fetch_spy_data(period="6mo")

        if sample_df is not None and not sample_df.empty:
            from src.core.indicator_calculator import calculate_rsi # Assuming this is still available
            if 'Close' in sample_df.columns:
                rsi_values = calculate_rsi(sample_df['Close'], period=14)
                print("RSI calculated.")

            custom_sma_series = sample_df['Close'].rolling(window=20).mean().rename("SMA_20_Close")
            custom_ema_series = sample_df['Close'].ewm(span=10, adjust=False).mean().rename("EMA_10_Close")
            print("Custom SMA and EMA series created.")

            # Create dummy ML signals for testing
            signal_values = np.random.choice([0, 1, np.nan], size=len(sample_df)) # Mix of 0, 1, and NaN
            dummy_ml_signals = pd.Series(signal_values, index=sample_df.index, name="ML_Signals_Test")
            # Ensure some non-NaN signals for plotting
            if pd.isna(dummy_ml_signals.iloc[5:15]).all(): # Ensure some signals exist
                 dummy_ml_signals.iloc[5:10] = 1
                 dummy_ml_signals.iloc[10:15] = 0
            print("Dummy ML signals created.")
        else:
            raise ValueError("Fetched sample_df is None or empty.")

    except Exception as e:
        print(f"Could not fetch/process real data due to: {e}. Using generated dummy data.")
        dates_rng = pd.date_range(start='2023-01-01', periods=100, freq='B')
        data_vals = [100 + i + (i%10) - (i%5) for i in range(100)]
        sample_df = pd.DataFrame({
            'Open': [v - 0.5 for v in data_vals],
            'High': [v + 1 for v in data_vals],
            'Low': [v - 1 for v in data_vals],
            'Close': data_vals
        }, index=pd.Index(dates_rng, name="Date"))

        if 'Close' in sample_df.columns:
            rsi_values = sample_df['Close'].rolling(window=14).apply(lambda x: x.sum() / 14, raw=True).rename("RSI_14")
            custom_sma_series = sample_df['Close'].rolling(window=20).mean().rename("SMA_20_Close")
            custom_ema_series = sample_df['Close'].ewm(span=10, adjust=False).mean().rename("EMA_10_Close")
            signal_values = np.random.choice([0, 1], size=len(sample_df))
            dummy_ml_signals = pd.Series(signal_values, index=sample_df.index, name="ML_Signals_Test")


    if sample_df is not None and not sample_df.empty:
        print("\nSample DataFrame Head for Charting:")
        print(sample_df.head())

        custom_indicators_multi = []
        if custom_sma_series is not None: custom_indicators_multi.append({'name': custom_sma_series.name, 'series': custom_sma_series})
        if custom_ema_series is not None: custom_indicators_multi.append({'name': custom_ema_series.name, 'series': custom_ema_series})


        print("\nCreating candlestick chart WITH ALL features (RSI, Custom Indicators, ML Signals)...")
        fig_with_all = create_candlestick_chart(sample_df.copy(), ticker_symbol="TEST (All Features)",
                                                rsi_series=rsi_values, rsi_period=14,
                                                custom_indicators_data=custom_indicators_multi,
                                                ml_signals=dummy_ml_signals)
        print("Figure with all features created. If interactive, call fig_with_all.show()")
        # fig_with_all.show()

        print("\nCreating candlestick chart WITH only ML Signals...")
        fig_ml_only = create_candlestick_chart(sample_df.copy(), ticker_symbol="TEST (ML Signals Only)",
                                               ml_signals=dummy_ml_signals)
        # fig_ml_only.show()

    else:
        print("Failed to get or generate sample data for chart testing.")
