import plotly.graph_objects as go
import pandas as pd
from plotly.subplots import make_subplots

def generate_candlestick_chart(
    df: pd.DataFrame,
    ticker: str = "SPY",
    rsi_series: pd.Series = None,
    show_rsi: bool = False,
    rsi_oversold: int = 30,
    rsi_overbought: int = 70
) -> go.Figure:
    """
    Generates an interactive candlestick chart with optional volume and RSI subplots.

    Args:
        df: Pandas DataFrame with OHLCV data. Must contain columns:
            'Open', 'High', 'Low', 'Close', and 'Volume'.
            The DataFrame index should be a DatetimeIndex (dates).
        ticker: The ticker symbol for the chart title.
        rsi_series: Pandas Series containing RSI values, indexed by date.
        show_rsi: Boolean to control the visibility of the RSI subplot.
        rsi_oversold: RSI level for the oversold line.
        rsi_overbought: RSI level for the overbought line.

    Returns:
        A Plotly Figure object.
    """
    if show_rsi and rsi_series is not None:
        rows = 3
        row_heights = [0.6, 0.2, 0.2] # Main chart, Volume, RSI
        specs = [[{"secondary_y": False}], [{"secondary_y": False}], [{"secondary_y": False}]]
        subplot_titles = ["", "", "RSI"] # Titles for subplots if needed, RSI title added below
        fig = make_subplots(rows=rows, cols=1, shared_xaxes=True,
                            vertical_spacing=0.03, row_heights=row_heights,
                            specs=specs, subplot_titles=subplot_titles)
    else:
        rows = 2
        row_heights = [0.7, 0.3] # Main chart, Volume
        specs = [[{"secondary_y": False}], [{"secondary_y": False}]]
        fig = make_subplots(rows=rows, cols=1, shared_xaxes=True,
                            vertical_spacing=0.03, row_heights=row_heights,
                            specs=specs)

    # Candlestick trace
    fig.add_trace(go.Candlestick(x=df.index,
                               open=df['Open'],
                               high=df['High'],
                               low=df['Low'],
                               close=df['Close'],
                               name='Price',
                               increasing_line_color='green',
                               decreasing_line_color='red'), row=1, col=1)

    # Volume bar trace
    fig.add_trace(go.Bar(x=df.index,
                         y=df['Volume'],
                         name='Volume',
                         marker_color='rgba(100, 100, 150, 0.6)'), row=2, col=1)

    if show_rsi and rsi_series is not None:
        fig.add_trace(go.Scatter(x=rsi_series.index, y=rsi_series,
                                 name='RSI', line=dict(color='orange')), row=3, col=1)
        fig.add_hline(y=rsi_overbought, line_dash="dash", line_color="red",
                      annotation_text="Overbought", annotation_position="bottom right", row=3, col=1)
        fig.add_hline(y=rsi_oversold, line_dash="dash", line_color="green",
                      annotation_text="Oversold", annotation_position="top right", row=3, col=1)
        fig.update_yaxes(title_text="RSI", row=3, col=1, range=[0, 100])


    # Update layout
    fig.update_layout(
        title=f'{ticker} Candlestick Chart',
        xaxis_title=None, # Remove individual x-axis titles, shared x-axis title is fine
        # yaxis_title='Price (USD)', # Set per subplot
        # yaxis2_title='Volume',
        xaxis_rangeslider_visible=False,  # Hide the range slider by default
        hovermode='x unified',  # Show unified tooltip for all traces at a given x
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    # Update y-axis titles for each subplot
    fig.update_yaxes(title_text="Price (USD)", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)


    # Interactivity - applied to the main (last) x-axis
    fig.update_xaxes(
        rangeselector=dict(
            buttons=list([
                dict(count=1, label="1m", step="month", stepmode="backward"),
                dict(count=6, label="6m", step="month", stepmode="backward"),
                dict(count=1, label="YTD", step="year", stepmode="todate"),
                dict(count=1, label="1y", step="year", stepmode="backward"),
                dict(step="all")
            ])
        ),
        rangeslider=dict(
            visible=True # Show only on the bottom-most x-axis
        ),
        type="date",
        row=rows, col=1 # Apply to the last row's x-axis
    )
    if show_rsi and rsi_series is not None:
         fig.update_xaxes(rangeslider_visible=False, row=1, col=1) # Hide for price
         fig.update_xaxes(rangeslider_visible=False, row=2, col=1) # Hide for volume
    else:
         fig.update_xaxes(rangeslider_visible=False, row=1, col=1) # Hide for price

    # Adjust overall figure height based on whether RSI is shown
    fig.update_layout(height=800 if (show_rsi and rsi_series is not None) else 600)


    return fig

if __name__ == '__main__':
    # Create a dummy DataFrame for testing
    import numpy as np
    from datetime import datetime, timedelta
    # Assuming calculate_rsi is in a sibling directory core
    # Assuming calculate_rsi is in a sibling directory core
    from ..core.indicator_calculator import calculate_rsi


    date_today = datetime.now()
    days = pd.date_range(date_today - timedelta(days=299), date_today, freq='D')
    np.random.seed(seed=42)

    # Generate somewhat realistic price data
    open_prices = np.random.normal(loc=100, scale=2, size=len(days))
    open_prices = np.cumsum(open_prices)
    open_prices[open_prices <=0] = 1 # Ensure positive prices

    close_prices = open_prices + np.random.normal(loc=0, scale=3, size=len(days))
    close_prices[close_prices <=0] = 1

    high_prices = np.maximum(open_prices, close_prices) + np.random.uniform(0, 5, size=len(days))
    low_prices = np.minimum(open_prices, close_prices) - np.random.uniform(0, 5, size=len(days))
    low_prices[low_prices <=0] = 0.5 # Ensure positive prices

    volumes = np.random.randint(1000000, 50000000, size=len(days))

    dummy_df = pd.DataFrame({
        'Open': open_prices,
        'High': high_prices,
        'Low': low_prices,
        'Close': close_prices,
        'Volume': volumes
    }, index=days)
    dummy_df.index.name = 'Date'


    if not dummy_df.empty:
        print("Generating chart with dummy data (NO RSI)...")
        chart_fig_no_rsi = generate_candlestick_chart(dummy_df.copy(), ticker="TEST_NO_RSI")
        # chart_fig_no_rsi.show()
        print("Dummy chart (NO RSI) figure created.")

        print("\nCalculating RSI for dummy data...")
        dummy_rsi = calculate_rsi(dummy_df.copy(), window=14)
        dummy_df['RSI'] = dummy_rsi

        print("\nGenerating chart with dummy data (WITH RSI)...")
        chart_fig_with_rsi = generate_candlestick_chart(dummy_df.copy(), ticker="TEST_WITH_RSI", rsi_series=dummy_df['RSI'], show_rsi=True)
        # chart_fig_with_rsi.show()
        print("Dummy chart (WITH RSI) figure created. Run app.py to display.")
    else:
        print("Dummy DataFrame is empty.")
