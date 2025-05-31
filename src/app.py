import streamlit as st
from core.data_manager import get_spy_data
from charts.plot_generator import generate_candlestick_chart
from core.indicator_calculator import calculate_rsi # Import RSI function
from datetime import datetime, timedelta

# Set page configuration for wider layout
st.set_page_config(layout="wide")

st.title("SPY Candlestick Chart with RSI")

# Date inputs
today = datetime.today()
default_start_date = today - timedelta(days=365 * 2) # Default to 2 years ago
default_end_date = today

# Sidebar for controls
st.sidebar.header("Chart Controls")
start_date = st.sidebar.date_input(
    "Start Date",
    value=default_start_date,
    min_value=datetime(1990, 1, 1),
    max_value=today
)
end_date = st.sidebar.date_input(
    "End Date",
    value=default_end_date,
    min_value=start_date,
    max_value=today
)

# RSI Controls
st.sidebar.subheader("RSI Indicator")
show_rsi_checkbox = st.sidebar.checkbox("Show RSI", value=True)
rsi_window = st.sidebar.slider("RSI Window", min_value=2, max_value=30, value=14, step=1, disabled=not show_rsi_checkbox)
rsi_oversold = st.sidebar.number_input("RSI Oversold Level", min_value=0, max_value=100, value=30, step=1, disabled=not show_rsi_checkbox)
rsi_overbought = st.sidebar.number_input("RSI Overbought Level", min_value=0, max_value=100, value=70, step=1, disabled=not show_rsi_checkbox)


# Fetch data
# Convert date objects to string format required by get_spy_data
start_date_str = start_date.strftime("%Y-%m-%d")
end_date_str = end_date.strftime("%Y-%m-%d")

@st.cache_data # Cache the data fetching
def load_data(start, end):
    data = get_spy_data(start_date=start, end_date=end)
    return data

@st.cache_data # Cache RSI calculation
def compute_rsi(data, window):
    if data is not None and not data.empty:
        return calculate_rsi(data, window=window)
    return None

data_load_state = st.text(f"Loading SPY data for {start_date_str} to {end_date_str}...")
spy_data_df = load_data(start_date_str, end_date_str)
data_load_state.text(f"Loading SPY data for {start_date_str} to {end_date_str}... Done!")

if spy_data_df is None or spy_data_df.empty:
    st.error("No data fetched for the selected date range. Please check the dates or try again later.")
else:
    # Calculate RSI if checkbox is selected
    rsi_series = None
    if show_rsi_checkbox:
        rsi_load_state = st.text(f"Calculating RSI with window {rsi_window}...")
        rsi_series = compute_rsi(spy_data_df.copy(), window=rsi_window) # Use .copy()
        rsi_load_state.text(f"Calculating RSI with window {rsi_window}... Done!")

    # Generate chart
    chart_load_state = st.text("Generating chart...")
    candlestick_fig = generate_candlestick_chart(
        spy_data_df,
        ticker="SPY",
        rsi_series=rsi_series,
        show_rsi=show_rsi_checkbox,
        rsi_oversold=rsi_oversold,
        rsi_overbought=rsi_overbought
    )
    chart_load_state.text("Generating chart... Done!")

    st.plotly_chart(candlestick_fig, use_container_width=True)

    st.subheader("Raw Data (Last 5 rows)")
    st.dataframe(spy_data_df.tail())


# App information and instructions moved to sidebar and expanded
st.sidebar.divider()
st.sidebar.header("About")
st.sidebar.info(
    "This application displays an interactive candlestick chart for SPY (S&P 500 ETF) data, "
    "with an optional Relative Strength Index (RSI) indicator. "
    "Data is fetched from Yahoo Finance via the `yfinance` library. "
    "The chart is generated using `plotly`."
)
st.sidebar.header("Instructions")
st.sidebar.markdown("""
- **Date Range:** Select the start and end dates in the sidebar.
- **RSI Display:**
    - Check 'Show RSI' to display the RSI indicator.
    - Adjust the RSI window period.
    - Set custom oversold and overbought levels.
- **Chart Interactivity:**
    - **Zoom:** Use the mouse wheel or the range slider at the bottom of the chart.
    - **Pan:** Click and drag on the chart.
    - **Tooltips:** Hover over data points to see detailed values (OHLCV, RSI).
    - **Date Range Shortcuts:** Use the buttons (1m, 6m, YTD, 1y, All) above the chart.
""")

# To run this app:
# 1. Ensure you have yfinance, plotly, streamlit, pandas, nbformat installed:
#    pip install -r requirements.txt
# 2. Ensure the application is not already running from a previous step.
# 3. Navigate to the root directory of the project.
# 4. Run `streamlit run src/app.py` in your terminal.
