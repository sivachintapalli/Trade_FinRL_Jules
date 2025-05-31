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
        main_df_index = df.index # Ensure main_df_index is defined for reindexing
        for indicator_info in custom_indicators_data:
            plot_type = indicator_info.get('plot_type', 'simple_series') # Default to simple_series
            indicator_name = indicator_info.get('name', f'CustomInd_{current_row-1}')

            if plot_type == 'saty_phase_oscillator':
                spo_df = indicator_info.get('data_df')
                spo_params = indicator_info.get('params', {})

                if spo_df is not None and isinstance(spo_df, pd.DataFrame) and not spo_df.empty:
                    # Plot Oscillator
                    fig.add_trace(go.Scatter(x=main_df_index, y=spo_df['oscillator'].reindex(main_df_index),
                                             mode='lines', name='Oscillator', connectgaps=True,
                                             legendgroup=f"group{current_row}", legendgrouptitle_text=indicator_name),
                                  row=current_row, col=1)

                    # Zone Lines
                    zone_lines = {
                        'extended_up_zone': {'y': 100.0, 'color': 'rgba(200,200,200,0.5)'},
                        'distribution_zone': {'y': 61.8, 'color': 'rgba(150,150,150,0.5)'},
                        'neutral_up_zone': {'y': 23.6, 'color': 'rgba(100,100,100,0.5)'},
                        'neutral_down_zone': {'y': -23.6, 'color': 'rgba(100,100,100,0.5)'},
                        'accumulation_zone': {'y': -61.8, 'color': 'rgba(150,150,150,0.5)'},
                        'extended_down_zone': {'y': -100.0, 'color': 'rgba(200,200,200,0.5)'}
                    }
                    for name, props in zone_lines.items():
                        fig.add_hline(y=props['y'], line_dash="dot", line_color=props['color'], row=current_row, col=1)

                    # Crossover Signals
                    show_zones_sig = spo_params.get('show_zone_crosses', True)
                    show_extremes_sig = spo_params.get('show_extreme_crosses', True)
                    crossover_signals = {
                        'leaving_accumulation_signal': {'color': 'yellow', 'symbol': 'circle', 'visible': show_zones_sig},
                        'leaving_distribution_signal': {'color': 'yellow', 'symbol': 'circle', 'visible': show_zones_sig},
                        'leaving_extreme_down_signal': {'color': 'lime', 'symbol': 'circle', 'visible': show_extremes_sig},
                        'leaving_extreme_up_signal': {'color': 'lime', 'symbol': 'circle', 'visible': show_extremes_sig}
                    }
                    for sig_col, props in crossover_signals.items():
                        if props['visible'] and sig_col in spo_df:
                            series_to_plot = spo_df[sig_col].reindex(main_df_index).dropna() # Changed variable name from 'series'
                            if not series_to_plot.empty:
                                fig.add_trace(go.Scatter(x=series_to_plot.index, y=series_to_plot, mode='markers',
                                                         name=sig_col.replace('_signal','').replace('_', ' ').title(),
                                                         marker_symbol=props['symbol'], marker_color=props['color'], marker_size=8,
                                                         legendgroup=f"group{current_row}"),
                                              row=current_row, col=1)

                    fig.update_yaxes(title_text=indicator_name[:15], row=current_row, col=1, autorange=False, range=[-150, 150]) # Fixed range for SPO
                    if num_rows > current_row:
                        fig.update_xaxes(title_text="", row=current_row, col=1)
                    current_row += 1
                else:
                    print(f"Warning: Saty Phase Oscillator data for '{indicator_name}' is missing or not a DataFrame. Skipping.")
                    # If an empty plot is preferred, increment current_row here too. For now, it skips the row.

            elif plot_type == 'simple_series': # Existing logic adapted
                series_data = indicator_info.get('series') # Changed variable name from 'series'
                if isinstance(series_data, pd.Series) and not series_data.empty:
                    aligned_series = series_data.reindex(main_df_index)
                    fig.add_trace(go.Scatter(x=aligned_series.index, y=aligned_series,
                                             mode='lines', name=indicator_name, connectgaps=True,
                                             legendgroup=f"group{current_row}", legendgrouptitle_text=indicator_name),
                                  row=current_row, col=1)
                    fig.update_yaxes(title_text=indicator_name[:15], row=current_row, col=1)
                    if num_rows > current_row:
                        fig.update_xaxes(title_text="", row=current_row, col=1)
                    current_row += 1
                else:
                    print(f"Warning: Custom indicator '{indicator_name}' data is missing or not a Series. Skipping.")
            # else: # Placeholder for other plot types
            #     print(f"Warning: Unknown plot_type '{plot_type}' for indicator '{indicator_name}'. Skipping.")
            #     current_row +=1 # Or handle appropriately


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

    # Must import after sys.path modification for src.
    from src.indicators.saty_phase_oscillator import SatyPhaseOscillator
    import numpy as np # Ensure numpy is available for __main__

    # Monkey-patch numpy for pandas-ta compatibility if needed
    if not hasattr(np, 'NaN'):
        np.NaN = np.nan

    sample_df = None
    rsi_values = None
    custom_sma_series = None
    custom_ema_series = None
    dummy_ml_signals = None
    spo_results_df = None # For Saty Phase Oscillator

    try:
        from src.core.data_manager import fetch_spy_data
        print("Fetching sample SPY data for chart generation test...")
        sample_df = fetch_spy_data(period="1y") # Longer period for better indicator display

        if sample_df is not None and not sample_df.empty:
            from src.core.indicator_calculator import calculate_rsi
            if 'Close' in sample_df.columns:
                rsi_values = calculate_rsi(sample_df['Close'], period=14)
                print("RSI calculated.")

            custom_sma_series = sample_df['Close'].rolling(window=20).mean().rename("SMA_20_Close")
            custom_ema_series = sample_df['Close'].ewm(span=10, adjust=False).mean().rename("EMA_10_Close")
            print("Custom SMA and EMA series created.")

            # Saty Phase Oscillator calculation
            saty_osc = SatyPhaseOscillator()
            spo_results_df = saty_osc.calculate(sample_df.copy())
            print("Saty Phase Oscillator calculated.")

            # Create dummy ML signals
            signal_values = np.random.choice([0, 1, np.nan], size=len(sample_df))
            dummy_ml_signals = pd.Series(signal_values, index=sample_df.index, name="ML_Signals_Test")
            if pd.isna(dummy_ml_signals.iloc[5:25]).all(): # Ensure some signals exist
                 dummy_ml_signals.iloc[5:15] = 1
                 dummy_ml_signals.iloc[15:25] = 0
            print("Dummy ML signals created.")
        else:
            raise ValueError("Fetched sample_df is None or empty.")

    except Exception as e:
        print(f"Could not fetch/process real data due to: {e}. Using generated dummy data.")
        dates_rng = pd.date_range(start='2023-01-01', periods=200, freq='B') # More data for SPO
        data_vals = [100 + i + (i%10) - (i%5) + np.sin(i/20)*5 for i in range(200)] # Some oscillation
        sample_df = pd.DataFrame({
            'Open': [v - 0.5 + np.random.randn()*0.2 for v in data_vals],
            'High': [v + 1 + np.abs(np.random.randn()*0.5) for v in data_vals],
            'Low': [v - 1 - np.abs(np.random.randn()*0.5) for v in data_vals],
            'Close': [v + np.random.randn()*0.3 for v in data_vals]
        }, index=pd.Index(dates_rng, name="Date"))
        sample_df['High'] = np.maximum(sample_df['High'], sample_df[['Open', 'Close']].max(axis=1))
        sample_df['Low'] = np.minimum(sample_df['Low'], sample_df[['Open', 'Close']].min(axis=1))


        if 'Close' in sample_df.columns:
            rsi_values = sample_df['Close'].rolling(window=14).apply(lambda x: x.sum() / 14, raw=True).rename("RSI_14")
            custom_sma_series = sample_df['Close'].rolling(window=20).mean().rename("SMA_20_Close")
            custom_ema_series = sample_df['Close'].ewm(span=10, adjust=False).mean().rename("EMA_10_Close")

            saty_osc_dummy = SatyPhaseOscillator() # On dummy data
            spo_results_df = saty_osc_dummy.calculate(sample_df.copy())

            signal_values = np.random.choice([0, 1], size=len(sample_df))
            dummy_ml_signals = pd.Series(signal_values, index=sample_df.index, name="ML_Signals_Test")


    if sample_df is not None and not sample_df.empty:
        print("\nSample DataFrame Head for Charting:")
        print(sample_df.head())

        custom_indicators_multi = []
        if custom_sma_series is not None:
            custom_indicators_multi.append({'name': custom_sma_series.name,
                                             'series': custom_sma_series,
                                             'plot_type': 'simple_series'})
        if custom_ema_series is not None:
            custom_indicators_multi.append({'name': custom_ema_series.name,
                                             'series': custom_ema_series,
                                             'plot_type': 'simple_series'})
        if spo_results_df is not None:
            custom_indicators_multi.append({
                'name': 'Saty Phase Osc.',
                'plot_type': 'saty_phase_oscillator',
                'data_df': spo_results_df,
                'params': SatyPhaseOscillator().get_params() # Use default params for plotting example
            })


        print("\nCreating candlestick chart WITH ALL features (RSI, Custom Indicators including SPO, ML Signals)...")
        fig_with_all = create_candlestick_chart(sample_df.copy(), ticker_symbol="TEST (All Features)",
                                                rsi_series=rsi_values, rsi_period=14,
                                                custom_indicators_data=custom_indicators_multi,
                                                ml_signals=dummy_ml_signals)
        print("Figure with all features created. If interactive, call fig_with_all.show()")
        # fig_with_all.show() # In a script, this would display the chart.

        # Example of saving to HTML (optional)
        # output_html_path = "candlestick_chart_with_spo.html"
        # fig_with_all.write_html(output_html_path)
        # print(f"Chart saved to {output_html_path}")


        print("\nCreating candlestick chart WITH only ML Signals...")
        fig_ml_only = create_candlestick_chart(sample_df.copy(), ticker_symbol="TEST (ML Signals Only)",
                                               ml_signals=dummy_ml_signals)
        # fig_ml_only.show()

    else:
        print("Failed to get or generate sample data for chart testing.")
