"""
Financial Chart Generation Module.

This module is dedicated to creating interactive financial charts using the
Plotly library. It primarily focuses on generating candlestick charts that can
be augmented with various technical indicators (such as RSI and custom-defined
indicators) as subplots, and machine learning-generated trading signals as
overlays on the main price chart.

The main function, `create_candlestick_chart`, orchestrates the assembly of these
chart components into a single, cohesive Plotly Figure object.
"""
import plotly.graph_objects as go
import pandas as pd
from plotly.subplots import make_subplots # Import make_subplots

def create_candlestick_chart(df: pd.DataFrame, ticker_symbol: str = "SPY",
                             rsi_series: pd.Series = None, rsi_period: int = 14,
                             overbought_level: int = 70, oversold_level: int = 30,
                             custom_indicators_data: list = None,
                             ml_signals: pd.Series = None):
    """
    Generates an interactive Plotly candlestick chart with optional overlays for
    RSI, custom technical indicators, and machine learning-generated trading signals.

    The chart is structured with a main candlestick plot and optional subplots
    for each provided indicator. ML signals are overlaid directly onto the
    candlestick plot.

    Args:
        df (pd.DataFrame): The primary DataFrame containing financial data,
                           specifically 'Open', 'High', 'Low', 'Close' (OHLC) columns.
                           It must have a DatetimeIndex for time-series plotting.
        ticker_symbol (str, optional): The ticker symbol of the financial instrument
                                       (e.g., "SPY", "AAPL"). Used for chart titles.
                                       Defaults to "SPY".
        rsi_series (pd.Series, optional): A pandas Series containing pre-calculated
                                          RSI values. If provided, RSI is plotted
                                          in a separate subplot below the main chart.
                                          Defaults to None.
        rsi_period (int, optional): The period used for the RSI calculation. This is
                                    primarily for labeling purposes in the chart legend.
                                    Defaults to 14.
        overbought_level (int, optional): The RSI level considered as overbought.
                                          A horizontal line is drawn at this level on
                                          the RSI subplot. Defaults to 70.
        oversold_level (int, optional): The RSI level considered as oversold.
                                        A horizontal line is drawn at this level on
                                        the RSI subplot. Defaults to 30.
        custom_indicators_data (list, optional): A list of dictionaries, where each
            dictionary defines a custom indicator to be plotted in its own subplot.
            Each dictionary should conform to one of the supported structures:
            1. For simple line series indicators:
               {
                   'name': str,  # Name for legend and subplot title
                   'series': pd.Series,  # Indicator values
                   'plot_type': 'simple_series' # Optional, defaults to this
               }
            2. For specific complex indicators like 'SatyPhaseOscillator':
               {
                   'name': str, # e.g., "Saty Phase Oscillator"
                   'plot_type': 'saty_phase_oscillator',
                   'data_df': pd.DataFrame, # The DataFrame output from SatyPhaseOscillator.calculate()
                   'params': dict # Parameters from SatyPhaseOscillator.get_params()
               }
            Defaults to None (no custom indicators plotted).
        ml_signals (pd.Series, optional): A pandas Series containing machine
                                          learning-generated trading signals, indexed
                                          by Date. Typically, a value of 1 indicates
                                          a "buy" signal, and 0 indicates a "sell"
                                          or "hold" signal. These are plotted as
                                          markers on the main candlestick chart.
                                          Defaults to None.

    Returns:
        plotly.graph_objects.Figure: A Plotly Figure object representing the
                                     generated financial chart. This figure can be
                                     displayed (e.g., `fig.show()`) or saved to a file.

    Raises:
        ValueError: If the input `df` is missing any of the required OHLC columns
                    ('Open', 'High', 'Low', 'Close').
        ValueError: If `rsi_series` is provided but is not a pandas Series.

    Subplot Handling:
        - The chart is constructed using `plotly.subplots.make_subplots` to accommodate
          multiple plots (main candlestick + indicators).
        - The main candlestick chart always occupies the first (top) row.
        - If `rsi_series` is provided, it is plotted in a dedicated subplot directly
          below the main candlestick chart.
        - Each indicator specified in `custom_indicators_data` is plotted in its own
          subsequent subplot.
        - Row heights are dynamically adjusted to give more space to the main
          candlestick chart while accommodating the indicator subplots.

    ML Signal Overlay:
        - If `ml_signals` are provided:
            - "Buy" signals (value 1) are marked with green, upward-pointing triangles
              placed slightly below the 'Low' price of the corresponding candle.
            - "Sell" signals (value 0) are marked with red, downward-pointing triangles
              placed slightly above the 'High' price of the corresponding candle.
    """

    # Ensure required columns exist in the main DataFrame
    required_cols = ['Open', 'High', 'Low', 'Close']
    if not all(col in df.columns for col in required_cols):
        missing_cols = [col for col in required_cols if col not in df.columns]
        raise ValueError(f"DataFrame is missing required columns for candlestick: {missing_cols}")

    num_custom_indicators = len(custom_indicators_data) if custom_indicators_data else 0
    num_rows = 1 + (1 if rsi_series is not None else 0) + num_custom_indicators

    row_heights = [0.6] # Main candlestick plot height (gets more space)
    if rsi_series is not None:
        row_heights.append(0.2)
    if num_custom_indicators > 0:
        # Distribute remaining height among custom indicators, with a cap on individual height
        custom_indicator_height_share = 0.2 # Max proportion for all custom indicators together
        if num_rows == 2 and rsi_series is None : # Only candlestick and custom indicators
             custom_indicator_height_share = 0.4 # Give more if only custom indicators

        individual_custom_height = custom_indicator_height_share / num_custom_indicators if num_custom_indicators > 0 else 0
        # Ensure a minimum reasonable height for custom indicators if many are present
        min_height_per_custom = 0.1 # if num_custom_indicators < 3 else 0.07
        individual_custom_height = max(individual_custom_height, min_height_per_custom)

        row_heights.extend([individual_custom_height] * num_custom_indicators)

    # Normalize row_heights to sum to 1, only if there are subplots.
    if num_rows > 1:
        total_height_ratio = sum(row_heights)
        if total_height_ratio > 0: # Avoid division by zero if all heights are zero (though unlikely)
            row_heights = [h / total_height_ratio for h in row_heights]
        else: # Fallback if total_height_ratio is 0
            row_heights = None


    fig = make_subplots(rows=num_rows, cols=1, shared_xaxes=True,
                        row_heights=row_heights if num_rows > 1 else None,
                        vertical_spacing=0.03) # Small vertical spacing between subplots

    current_row = 1

    # --- Candlestick Trace (Main Plot) ---
    fig.add_trace(go.Candlestick(x=df.index,
                               open=df['Open'],
                               high=df['High'],
                               low=df['Low'],
                               close=df['Close'],
                               name=f'{ticker_symbol} OHLC'),
                  row=current_row, col=1)

    # --- Add ML Signals to the Candlestick Plot (if provided) ---
    if ml_signals is not None and isinstance(ml_signals, pd.Series) and not ml_signals.empty:
        aligned_signals = ml_signals.reindex(df.index) # Align signals with main df index

        buy_signals = df.loc[aligned_signals == 1]
        sell_signals = df.loc[aligned_signals == 0] # Assuming 0 is sell

        if not buy_signals.empty:
            fig.add_trace(go.Scatter(
                x=buy_signals.index,
                y=buy_signals['Low'] * 0.98, # Place marker below the low for buys
                mode='markers',
                marker_symbol='triangle-up',
                marker_size=10,
                marker_color='green',
                name='Buy Signal'
            ), row=current_row, col=1)

        if not sell_signals.empty:
            fig.add_trace(go.Scatter(
                x=sell_signals.index,
                y=sell_signals['High'] * 1.02, # Place marker above the high for sells
                mode='markers',
                marker_symbol='triangle-down',
                marker_size=10,
                marker_color='red',
                name='Sell Signal'
            ), row=current_row, col=1)

    fig.update_yaxes(title_text="Price", row=current_row, col=1)
    if num_rows > 1: # If there are subplots below, hide x-axis labels for this row
        fig.update_xaxes(showticklabels=False, row=current_row, col=1)
    current_row += 1

    # --- RSI Trace (Subplot, if provided) ---
    if rsi_series is not None:
        if not isinstance(rsi_series, pd.Series):
            raise ValueError("rsi_series must be a pandas Series.")
        if not rsi_series.empty:
            aligned_rsi = rsi_series.reindex(df.index)
            fig.add_trace(go.Scatter(x=aligned_rsi.index, y=aligned_rsi,
                                     mode='lines', name=f'RSI ({rsi_period})',
                                     line=dict(color='cyan')),
                          row=current_row, col=1)
            # Add overbought and oversold lines
            fig.add_hline(y=overbought_level, line_dash="dash", line_color="red", row=current_row, col=1)
            fig.add_hline(y=oversold_level, line_dash="dash", line_color="green", row=current_row, col=1)
            fig.update_yaxes(title_text="RSI", range=[0, 100], row=current_row, col=1)
            if num_rows > current_row: # If more subplots below, hide x-axis labels
                 fig.update_xaxes(showticklabels=False, row=current_row, col=1)
            current_row += 1

    # --- Custom Indicators Traces (Subplots, if provided) ---
    if custom_indicators_data:
        main_df_index = df.index
        for indicator_info in custom_indicators_data:
            plot_type = indicator_info.get('plot_type', 'simple_series')
            indicator_name = indicator_info.get('name', f'CustomInd_{current_row-1}')

            if plot_type == 'saty_phase_oscillator':
                spo_df = indicator_info.get('data_df')
                spo_params = indicator_info.get('params', {})

                if spo_df is not None and isinstance(spo_df, pd.DataFrame) and not spo_df.empty:
                    fig.add_trace(go.Scatter(x=main_df_index, y=spo_df['oscillator'].reindex(main_df_index),
                                             mode='lines', name='Oscillator', connectgaps=True,
                                             legendgroup=f"group{current_row}", legendgrouptitle_text=indicator_name),
                                  row=current_row, col=1)
                    zone_lines = {
                        'extended_up_zone': {'y': 100.0, 'color': 'rgba(200,200,200,0.5)'},
                        'distribution_zone': {'y': 61.8, 'color': 'rgba(150,150,150,0.5)'},
                        'neutral_up_zone': {'y': 23.6, 'color': 'rgba(100,100,100,0.5)'},
                        'neutral_down_zone': {'y': -23.6, 'color': 'rgba(100,100,100,0.5)'},
                        'accumulation_zone': {'y': -61.8, 'color': 'rgba(150,150,150,0.5)'},
                        'extended_down_zone': {'y': -100.0, 'color': 'rgba(200,200,200,0.5)'}
                    }
                    for _, props in zone_lines.items():
                        fig.add_hline(y=props['y'], line_dash="dot", line_color=props['color'], row=current_row, col=1)

                    crossover_signals = {
                        'leaving_accumulation_signal': {'color': 'yellow', 'symbol': 'circle', 'visible': spo_params.get('show_zone_crosses', True)},
                        'leaving_distribution_signal': {'color': 'yellow', 'symbol': 'circle', 'visible': spo_params.get('show_zone_crosses', True)},
                        'leaving_extreme_down_signal': {'color': 'lime', 'symbol': 'circle', 'visible': spo_params.get('show_extreme_crosses', True)},
                        'leaving_extreme_up_signal': {'color': 'lime', 'symbol': 'circle', 'visible': spo_params.get('show_extreme_crosses', True)}
                    }
                    for sig_col, props in crossover_signals.items():
                        if props['visible'] and sig_col in spo_df:
                            sig_series_to_plot = spo_df[sig_col].reindex(main_df_index).dropna()
                            if not sig_series_to_plot.empty:
                                fig.add_trace(go.Scatter(x=sig_series_to_plot.index, y=sig_series_to_plot, mode='markers',
                                                         name=sig_col.replace('_signal','').replace('_', ' ').title(),
                                                         marker_symbol=props['symbol'], marker_color=props['color'], marker_size=8,
                                                         legendgroup=f"group{current_row}"),
                                              row=current_row, col=1)
                    fig.update_yaxes(title_text=indicator_name[:15], row=current_row, col=1, autorange=False, range=[-150, 150])
                    if num_rows > current_row: fig.update_xaxes(showticklabels=False, row=current_row, col=1)
                    current_row += 1
                else:
                    print(f"Warning: Saty Phase Oscillator data for '{indicator_name}' is missing or invalid. Skipping plot row.")
            elif plot_type == 'simple_series':
                series_data = indicator_info.get('series')
                if isinstance(series_data, pd.Series) and not series_data.empty:
                    aligned_series = series_data.reindex(main_df_index)
                    fig.add_trace(go.Scatter(x=aligned_series.index, y=aligned_series,
                                             mode='lines', name=indicator_name, connectgaps=True,
                                             legendgroup=f"group{current_row}", legendgrouptitle_text=indicator_name),
                                  row=current_row, col=1)
                    fig.update_yaxes(title_text=indicator_name[:15], row=current_row, col=1) # Truncate long names
                    if num_rows > current_row: fig.update_xaxes(showticklabels=False, row=current_row, col=1)
                    current_row += 1
                else:
                    print(f"Warning: Custom indicator '{indicator_name}' (simple_series) data is missing or invalid. Skipping plot row.")
            # else: # Placeholder for other plot types
            #     print(f"Warning: Unknown plot_type '{plot_type}' for indicator '{indicator_name}'. Skipping plot row.")


    # --- Final Layout Updates ---
    chart_title = f'{ticker_symbol} Candlestick Chart'
    if rsi_series is not None: chart_title += ' with RSI'
    if num_custom_indicators > 0: chart_title += f' & Custom Indicator(s)'

    fig.update_layout(
        title=chart_title,
        xaxis_rangeslider_visible=False, # Hide the default rangeslider
        template="plotly_dark", # Using a dark theme
        height=max(400, 250 + (num_rows * 150)), # Adjust height based on number of subplots
        legend_tracegroupgap=20 # Space between legend groups
    )

    # Ensure the bottom-most x-axis always has the title "Date" and its tick labels are visible.
    if num_rows > 0 : # Should always be true
        fig.update_xaxes(title_text="Date", showticklabels=True, row=num_rows, col=1)


    # General style adjustments for all axes grids (consistent look)
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)')
    for i in range(1, num_rows + 1):
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)', row=i, col=1)

    return fig

if __name__ == '__main__':
    import sys
    import os
    # Add the project root to the Python path for src imports
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    from src.indicators.saty_phase_oscillator import SatyPhaseOscillator
    import numpy as np

    # Ensure numpy.NaN is available for pandas_ta compatibility if it were used (not directly here)
    if not hasattr(np, 'NaN'):
        np.NaN = np.nan # type: ignore

    sample_df_main = None
    rsi_values_main = None
    custom_sma_series_main = None
    custom_ema_series_main = None
    dummy_ml_signals_main = None
    spo_results_df_main = None

    try:
        from src.core.data_manager import fetch_spy_data
        print("Fetching sample SPY data for chart generation test...")
        sample_df_main = fetch_spy_data(period="1y")

        if sample_df_main is not None and not sample_df_main.empty:
            from src.core.indicator_calculator import calculate_rsi
            if 'Close' in sample_df_main.columns:
                rsi_values_main = calculate_rsi(sample_df_main['Close'], period=14)
                print("RSI calculated.")

            custom_sma_series_main = sample_df_main['Close'].rolling(window=20).mean().rename("SMA_20_Close")
            custom_ema_series_main = sample_df_main['Close'].ewm(span=10, adjust=False).mean().rename("EMA_10_Close")
            print("Custom SMA and EMA series created.")

            saty_osc_main = SatyPhaseOscillator()
            spo_results_df_main = saty_osc_main.calculate(sample_df_main.copy())
            print("Saty Phase Oscillator calculated.")

            signal_values_main = np.random.choice([0, 1, np.nan], size=len(sample_df_main))
            dummy_ml_signals_main = pd.Series(signal_values_main, index=sample_df_main.index, name="ML_Signals_Test")
            if pd.isna(dummy_ml_signals_main.iloc[5:25]).all():
                 dummy_ml_signals_main.iloc[5:15] = 1
                 dummy_ml_signals_main.iloc[15:25] = 0
            print("Dummy ML signals created.")
        else:
            raise ValueError("Fetched sample_df_main is None or empty.")

    except Exception as e_main:
        print(f"Could not fetch/process real data due to: {e_main}. Using generated dummy data.")
        dates_rng_main = pd.date_range(start='2023-01-01', periods=200, freq='B')
        data_vals_main = [100 + i + (i%10) - (i%5) + np.sin(i/20)*5 for i in range(200)]
        sample_df_main = pd.DataFrame({
            'Open': [v - 0.5 + np.random.randn()*0.2 for v in data_vals_main],
            'High': [v + 1 + np.abs(np.random.randn()*0.5) for v in data_vals_main],
            'Low': [v - 1 - np.abs(np.random.randn()*0.5) for v in data_vals_main],
            'Close': [v + np.random.randn()*0.3 for v in data_vals_main]
        }, index=pd.Index(dates_rng_main, name="Date"))
        sample_df_main['High'] = np.maximum(sample_df_main['High'], sample_df_main[['Open', 'Close']].max(axis=1))
        sample_df_main['Low'] = np.minimum(sample_df_main['Low'], sample_df_main[['Open', 'Close']].min(axis=1))

        if 'Close' in sample_df_main.columns:
            # Simplified RSI for dummy data
            rsi_values_main = sample_df_main['Close'].rolling(window=14).apply(lambda x: np.sum(x) / 14 if len(x) == 14 else np.nan, raw=True).rename("RSI_14")
            custom_sma_series_main = sample_df_main['Close'].rolling(window=20).mean().rename("SMA_20_Close")
            custom_ema_series_main = sample_df_main['Close'].ewm(span=10, adjust=False).mean().rename("EMA_10_Close")

            saty_osc_dummy_main = SatyPhaseOscillator()
            spo_results_df_main = saty_osc_dummy_main.calculate(sample_df_main.copy())

            signal_values_main = np.random.choice([0, 1], size=len(sample_df_main))
            dummy_ml_signals_main = pd.Series(signal_values_main, index=sample_df_main.index, name="ML_Signals_Test")


    if sample_df_main is not None and not sample_df_main.empty:
        print("\nSample DataFrame Head for Charting:")
        print(sample_df_main.head())

        custom_indicators_list = []
        if custom_sma_series_main is not None:
            custom_indicators_list.append({'name': custom_sma_series_main.name,
                                             'series': custom_sma_series_main,
                                             'plot_type': 'simple_series'})
        if custom_ema_series_main is not None:
            custom_indicators_list.append({'name': custom_ema_series_main.name,
                                             'series': custom_ema_series_main,
                                             'plot_type': 'simple_series'})
        if spo_results_df_main is not None:
            custom_indicators_list.append({
                'name': 'Saty Phase Osc.',
                'plot_type': 'saty_phase_oscillator',
                'data_df': spo_results_df_main,
                'params': SatyPhaseOscillator().get_params()
            })

        print("\nCreating candlestick chart WITH ALL features (RSI, Custom Indicators including SPO, ML Signals)...")
        fig_all_features = create_candlestick_chart(sample_df_main.copy(), ticker_symbol="TEST (All Features)",
                                                    rsi_series=rsi_values_main, rsi_period=14,
                                                    custom_indicators_data=custom_indicators_list,
                                                    ml_signals=dummy_ml_signals_main)
        print("Figure with all features created. In an interactive environment, call fig_all_features.show()")
        # fig_all_features.show() # Uncomment to display in an interactive session

        # Optional: Save to HTML for inspection
        # output_html_path_all = "candlestick_chart_all_features.html"
        # fig_all_features.write_html(output_html_path_all)
        # print(f"Chart with all features saved to {output_html_path_all}")

        print("\nCreating candlestick chart WITH only ML Signals...")
        fig_ml_only_main = create_candlestick_chart(sample_df_main.copy(), ticker_symbol="TEST (ML Signals Only)",
                                                    ml_signals=dummy_ml_signals_main)
        print("Figure with ML signals only created.")
        # fig_ml_only_main.show()

    else:
        print("Failed to get or generate sample data for chart testing.")
    print("\n--- Plot Generator Test Complete ---")

[end of src/charts/plot_generator.py]
