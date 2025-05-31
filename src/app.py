import streamlit as st
import pandas as pd

# Add project root to sys.path to allow direct imports from src
import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.core.data_manager import fetch_spy_data
from src.core.indicator_calculator import calculate_rsi
from src.charts.plot_generator import create_candlestick_chart

# --- Page Configuration ---
st.set_page_config(
    page_title="Trading Analysis Platform",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Application State ---
# Using st.session_state to hold data and prevent re-fetching on every interaction
if 'spy_data' not in st.session_state:
    st.session_state.spy_data = None
if 'rsi_period' not in st.session_state:
    st.session_state.rsi_period = 14 # Default RSI period

# --- Helper Functions ---
def load_data(ticker="SPY", period="1y", force_refresh=False):
    """Loads data, using cache by default, unless force_refresh is True."""
    use_cache_cond = not force_refresh
    st.write(f"Loading data for {ticker} (Period: {period}, Cache Used: {use_cache_cond})...")
    data = fetch_spy_data(ticker_symbol=ticker, period=period, use_cache=use_cache_cond)
    if data is not None and not data.empty:
        st.session_state.spy_data = data
        st.success(f"Successfully loaded data for {ticker} ({len(data)} rows).")
        # Automatically calculate RSI for the loaded data
        if 'Close' in st.session_state.spy_data.columns:
            st.session_state.rsi_values = calculate_rsi(st.session_state.spy_data['Close'],
                                                        period=st.session_state.rsi_period)
    else:
        st.error(f"Failed to load data for {ticker}.")
        st.session_state.spy_data = None # Clear data on failure
        st.session_state.rsi_values = None


# --- Sidebar ---
st.sidebar.title("Controls")

# Data loading section
st.sidebar.header("Data Loading")
selected_ticker = st.sidebar.text_input("Ticker Symbol", value="SPY").upper()
selected_period = st.sidebar.selectbox("Data Period", ["1mo", "3mo", "6mo", "1y", "2y", "5y", "max"], index=3) # Default 1y

if st.sidebar.button("Load/Refresh Data", key="load_data_button"):
    with st.spinner(f"Fetching data for {selected_ticker}..."):
        load_data(ticker=selected_ticker, period=selected_period, force_refresh=True)

# RSI Configuration
st.sidebar.header("Indicators")
st.session_state.rsi_period = st.sidebar.slider("RSI Period", min_value=7, max_value=30,
                                                value=st.session_state.rsi_period, key="rsi_period_slider")

# Re-calculate RSI if period changes and data exists
if st.session_state.spy_data is not None and 'Close' in st.session_state.spy_data.columns:
    # Check if RSI needs recalculation (e.g. period changed)
    # This simple check might recalculate more than needed, but is fine for now.
    # A more robust way would be to store the period used for the current rsi_values.
    st.session_state.rsi_values = calculate_rsi(st.session_state.spy_data['Close'],
                                                period=st.session_state.rsi_period)


# --- Main Page ---
st.title(f"📊 {selected_ticker} Financial Analysis")

# Initial data load if not already loaded
if st.session_state.spy_data is None:
    with st.spinner("Performing initial data load for SPY (1 year)..."):
        load_data(ticker="SPY", period="1y") # Initial load for SPY

# Display Chart
if st.session_state.spy_data is not None:
    st.header("Candlestick Chart with RSI")

    # Retrieve RSI values from session state
    rsi_to_plot = st.session_state.get('rsi_values', None)

    if rsi_to_plot is None and 'Close' in st.session_state.spy_data.columns : # Should have been calculated on load
         rsi_to_plot = calculate_rsi(st.session_state.spy_data['Close'], period=st.session_state.rsi_period)
         st.session_state.rsi_values = rsi_to_plot


    chart_fig = create_candlestick_chart(
        df=st.session_state.spy_data,
        ticker_symbol=selected_ticker,
        rsi_series=rsi_to_plot,
        rsi_period=st.session_state.rsi_period
    )
    st.plotly_chart(chart_fig, use_container_width=True)

    # Display raw data in an expander
    with st.expander(f"View Raw Data for {selected_ticker}"):
        st.dataframe(st.session_state.spy_data)

    if rsi_to_plot is not None:
        with st.expander(f"View RSI Data (Period: {st.session_state.rsi_period})"):
            st.dataframe(rsi_to_plot)
else:
    st.warning(f"No data loaded for {selected_ticker}. Please load data using the sidebar controls.")

st.sidebar.markdown("---")
st.sidebar.info("Built with FinRL & Streamlit")

# To run this app: streamlit run src/app.py
