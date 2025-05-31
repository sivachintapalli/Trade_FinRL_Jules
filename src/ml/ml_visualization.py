# src/ml/ml_visualization.py
# Contains functions for visualizing ML model predictions on price data.

import pandas as pd
import numpy as np
import plotly.graph_objects as go

def plot_price_with_signals(price_df: pd.DataFrame,
                            signals_series: pd.Series,
                            buy_signal_val=1,
                            sell_signal_val=0,
                            plot_title: str = 'Price with Buy/Sell Signals') -> go.Figure:
    """
    Plots price data (candlestick) with buy and sell signals using Plotly.

    Parameters:
    - price_df (pd.DataFrame): DataFrame with 'Date' (or index as date),
                               'Open', 'High', 'Low', 'Close' columns.
    - signals_series (pd.Series): Series with dates as index (matching price_df)
                                  and signal values (e.g., 1 for buy, 0 for sell).
    - buy_signal_val (any): The value in signals_series representing a buy signal.
    - sell_signal_val (any): The value in signals_series representing a sell signal.
    - plot_title (str): The title for the plot.

    Returns:
    - go.Figure: The Plotly figure object.
    """
    plot_df = price_df.copy()

    # Ensure index is datetime
    if not isinstance(plot_df.index, pd.DatetimeIndex):
        if 'Date' in plot_df.columns:
            plot_df['Date'] = pd.to_datetime(plot_df['Date'])
            plot_df.set_index('Date', inplace=True)
        else:
            try:
                plot_df.index = pd.to_datetime(plot_df.index)
            except Exception as e:
                raise ValueError("price_df index must be convertible to DatetimeIndex or have a 'Date' column.") from e

    fig = go.Figure()

    # Add Candlestick trace
    fig.add_trace(go.Candlestick(
        x=plot_df.index,
        open=plot_df['Open'],
        high=plot_df['High'],
        low=plot_df['Low'],
        close=plot_df['Close'],
        name='Price'
    ))

    # Filter signals
    # Ensure signals_series index is aligned with price_df index for proper selection
    aligned_signals = signals_series.reindex(plot_df.index)

    buy_signals = aligned_signals[aligned_signals == buy_signal_val]
    sell_signals = aligned_signals[aligned_signals == sell_signal_val]

    # Add Buy signals markers
    if not buy_signals.empty:
        buy_prices = plot_df.loc[buy_signals.index]['Low'] * 0.98  # Below the Low
        fig.add_trace(go.Scatter(
            x=buy_signals.index,
            y=buy_prices,
            mode='markers',
            name='Buy Signal',
            marker=dict(symbol='triangle-up', color='green', size=10, line=dict(width=1, color='DarkSlateGrey')),
            hoverinfo='x+text',
            text=[f'Buy Signal at {price:.2f}' for price in plot_df.loc[buy_signals.index]['Close']]
        ))

    # Add Sell signals markers
    if not sell_signals.empty:
        sell_prices = plot_df.loc[sell_signals.index]['High'] * 1.02  # Above the High
        fig.add_trace(go.Scatter(
            x=sell_signals.index,
            y=sell_prices,
            mode='markers',
            name='Sell Signal',
            marker=dict(symbol='triangle-down', color='red', size=10, line=dict(width=1, color='DarkSlateGrey')),
            hoverinfo='x+text',
            text=[f'Sell Signal at {price:.2f}' for price in plot_df.loc[sell_signals.index]['Close']]
        ))

    # Update layout
    fig.update_layout(
        title=plot_title,
        xaxis_title='Date',
        yaxis_title='Price',
        xaxis_rangeslider_visible=False,
        legend_title_text='Legend',
        template='plotly_white' # Using a clean template
    )

    return fig

if __name__ == '__main__':
    # Create dummy price data
    num_days = 100
    start_date = pd.to_datetime('2023-01-01')
    date_rng = pd.date_range(start=start_date, periods=num_days, freq='B') # Business days

    data = {
        'Open': np.random.uniform(100, 150, num_days),
        'High': np.random.uniform(150, 200, num_days),
        'Low': np.random.uniform(50, 100, num_days),
        'Close': np.random.uniform(100, 150, num_days) # Re-randomize close
    }
    # Ensure High is highest and Low is lowest
    data['High'] = np.maximum(data['Open'], data['Close']) + np.random.uniform(1, 10, num_days)
    data['Low'] = np.minimum(data['Open'], data['Close']) - np.random.uniform(1, 10, num_days)

    price_data_example = pd.DataFrame(data, index=date_rng)
    price_data_example.index.name = 'Date'


    # Create dummy signals series
    # Ensure signals are within the date range of price_data_example
    signal_dates = np.random.choice(price_data_example.index, size=20, replace=False)
    signal_values = np.random.choice([0, 1], size=20) # 0 for sell, 1 for buy
    signals_example = pd.Series(signal_values, index=signal_dates, name='ML_Signal')
    signals_example.sort_index(inplace=True)

    print("Example Price Data Head:")
    print(price_data_example.head())
    print("\nExample Signals Series Head (subset of dates):")
    print(signals_example.head())

    # Check alignment (signals_example might not have all dates from price_data_example)
    # The function handles reindexing.

    # Plot
    fig_example = plot_price_with_signals(price_data_example, signals_example,
                                          buy_signal_val=1, sell_signal_val=0,
                                          plot_title="Dummy Price Data with ML Signals")

    # Save to HTML (safer for automated environments)
    output_html_file = "price_with_signals_example.html"
    try:
        fig_example.write_html(output_html_file)
        print(f"\nExample plot saved to {output_html_file}")
        # To view, open this HTML file in a browser.
    except Exception as e:
        print(f"Error saving plot to HTML: {e}")

    # If running in an environment that supports it, uncomment to show:
    # fig_example.show()
