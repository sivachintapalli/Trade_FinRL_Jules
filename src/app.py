import streamlit as st
import pandas as pd

# Add project root to sys.path to allow direct imports from src
import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.core.data_manager import fetch_spy_data
from src.core.indicator_calculator import calculate_rsi # Will be replaced or augmented by custom indicators
from src.charts.plot_generator import create_candlestick_chart
from src.core.indicator_manager import discover_indicators
from src.core.custom_indicator_interface import CustomIndicator
import inspect # For inspecting indicator parameters
import subprocess # For running train.py
from src.ml.predict import get_predictions # Import the new function

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
if 'rsi_period' not in st.session_state: # Retain for existing RSI logic if needed, or phase out
    st.session_state.rsi_period = 14
if 'rsi_values' not in st.session_state: # For existing RSI plot
    st.session_state.rsi_values = None

# Custom Indicator State
if 'available_custom_indicators' not in st.session_state:
    st.session_state.available_custom_indicators = discover_indicators()
if 'active_custom_indicators' not in st.session_state: # Stores list of dicts: {"name": str, "instance": CustomIndicator, "data": pd.Series/df, "params": dict}
    st.session_state.active_custom_indicators = []
if 'custom_indicator_param_values' not in st.session_state: # Temporarily stores UI inputs for selected custom indicator
    st.session_state.custom_indicator_param_values = {}
# ML State
if 'ml_model_path' not in st.session_state:
    st.session_state.ml_model_path = 'models/lstm_model_v1.pth' # Default path
if 'ml_signals' not in st.session_state:
    st.session_state.ml_signals = None
# Default hyperparameters for LSTM model (should match training defaults or be configurable)
if 'ml_sequence_length' not in st.session_state:
    st.session_state.ml_sequence_length = 20
if 'ml_hidden_size' not in st.session_state:
    st.session_state.ml_hidden_size = 50
if 'ml_num_layers' not in st.session_state:
    st.session_state.ml_num_layers = 2
if 'ml_dropout' not in st.session_state:
    st.session_state.ml_dropout = 0.2


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

# RSI Configuration (keeping existing for now, can be integrated/removed later)
st.sidebar.header("Built-in Indicators")
st.session_state.rsi_period = st.sidebar.slider("RSI Period", min_value=7, max_value=30,
                                                value=st.session_state.rsi_period, key="rsi_period_slider_main")

# Re-calculate RSI if period changes and data exists
if st.session_state.spy_data is not None and 'Close' in st.session_state.spy_data.columns:
    st.session_state.rsi_values = calculate_rsi(st.session_state.spy_data['Close'],
                                                period=st.session_state.rsi_period)

# --- Custom Indicators Section ---
st.sidebar.header("Custom Indicators")
custom_indicator_options = ["None"] + sorted(list(st.session_state.available_custom_indicators.keys()))
selected_indicator_name = st.sidebar.selectbox(
    "Add Indicator",
    options=custom_indicator_options,
    index=0, # Default to "None"
    key="custom_indicator_selector"
)

if selected_indicator_name != "None" and selected_indicator_name in st.session_state.available_custom_indicators:
    IndicatorClass = st.session_state.available_custom_indicators[selected_indicator_name]
    sig = inspect.signature(IndicatorClass.__init__)

    param_inputs = {}
    st.sidebar.markdown(f"**Parameters for {selected_indicator_name}:**")
    for name, param in sig.parameters.items():
        if name == 'self' or name == 'kwargs' or name == 'args': # Skip self and generic collectors
            continue

        default_value = param.default if param.default is not inspect.Parameter.empty else None
        # Key for widget should be unique per indicator and parameter
        widget_key = f"param_{selected_indicator_name}_{name}"

        if param.annotation == int:
            param_inputs[name] = st.sidebar.number_input(
                f"{name} (int)",
                value=int(default_value) if default_value is not None else 0,
                key=widget_key
            )
        elif param.annotation == float:
            param_inputs[name] = st.sidebar.number_input(
                f"{name} (float)",
                value=float(default_value) if default_value is not None else 0.0,
                format="%.5f", # Increased precision for typical financial indicators
                key=widget_key
            )
        elif param.annotation == bool:
            param_inputs[name] = st.sidebar.selectbox(
                f"{name} (bool)",
                options=[True, False],
                index=0 if default_value is True else 1,
                key=widget_key
            )
        elif param.annotation == str:
            df_cols_with_empty = [""] # Add empty option for string params that aren't columns
            if st.session_state.spy_data is not None:
                df_cols_with_empty = [""] + list(st.session_state.spy_data.columns)

            val = str(default_value) if default_value is not None else ""

            if name in ['column', 'source'] and st.session_state.spy_data is not None: # Special handling for column names
                 col_idx = df_cols_with_empty.index(val) if val in df_cols_with_empty else 0
                 param_inputs[name] = st.sidebar.selectbox(
                    f"{name} (data column)",
                    options=df_cols_with_empty,
                    index=col_idx,
                    key=widget_key
                )
            else:
                param_inputs[name] = st.sidebar.text_input(
                    f"{name} (str)",
                    value=val,
                    key=widget_key
                )
        else: # Fallback for unsupported types
            param_inputs[name] = st.sidebar.text_input(
                f"{name} (unknown type - as str)",
                value=str(default_value) if default_value is not None else "",
                key=widget_key,
                help=f"Type {param.annotation} not directly supported. Enter as string."
            )

    st.session_state.custom_indicator_param_values = param_inputs # Store current params for "Add" button

    if st.sidebar.button(f"Add {selected_indicator_name} Instance", key=f"add_custom_indicator_btn"):
        if st.session_state.spy_data is not None and not st.session_state.spy_data.empty:
            try:
                current_params = st.session_state.custom_indicator_param_values
                indicator_instance = IndicatorClass(**current_params)
                calculated_data = indicator_instance.calculate(st.session_state.spy_data)

                # Ensure series/df has a good name
                if isinstance(calculated_data, pd.Series) and calculated_data.name is None:
                    calculated_data.name = f"{selected_indicator_name}_{'_'.join(map(str, current_params.values()))}"
                elif isinstance(calculated_data, pd.DataFrame):
                    # Name columns of DataFrame if they are not named well
                    base_name = f"{selected_indicator_name}_{'_'.join(map(str, current_params.values()))}"
                    calculated_data.columns = [f"{base_name}_{i}" if col_name is None or isinstance(col_name, int) else f"{base_name}_{col_name}" for i, col_name in enumerate(calculated_data.columns)]


                st.session_state.active_custom_indicators.append({
                    "name": selected_indicator_name,
                    "instance": indicator_instance, # Store instance for repr or future use
                    "data": calculated_data,
                    "params": current_params.copy(),
                    "id": f"{selected_indicator_name}_{len(st.session_state.active_custom_indicators)}" # Unique ID
                })
                st.sidebar.success(f"Added {selected_indicator_name} instance.")
                # Reset selector to "None" to allow adding another or same indicator with different params
                # st.session_state.custom_indicator_selector = "None" # This causes issues with Streamlit's widget state.
                # Instead, rely on user to change params or re-select.
            except Exception as e:
                st.sidebar.error(f"Error adding {selected_indicator_name}: {e}")
        else:
            st.sidebar.warning("Load data before adding indicators.")

# Display and allow removal of active custom indicators
if st.session_state.active_custom_indicators:
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Active Custom Indicators:**")

    # Create a copy for iteration as we might modify the list
    for i, indicator_info in reversed(list(enumerate(st.session_state.active_custom_indicators))):
        params_str = ", ".join(f"{k}={v}" for k, v in indicator_info['params'].items())
        label = f"{indicator_info['name']}({params_str})"

        col1, col2 = st.sidebar.columns([0.85, 0.15])
        col1.text(label[:40] + "..." if len(label) > 40 else label) # Truncate long labels
        if col2.button("X", key=f"remove_custom_ind_{indicator_info['id']}", help="Remove this indicator"):
            st.session_state.active_custom_indicators.pop(i)
            st.experimental_rerun() # Rerun to update the UI and chart

    if st.sidebar.button("Clear All Custom Indicators", key="clear_all_custom_btn"):
        st.session_state.active_custom_indicators = []
        st.experimental_rerun()

# --- ML Section ---
st.sidebar.header("Machine Learning")
st.session_state.ml_model_path = st.sidebar.text_input(
    "ML Model Path",
    value=st.session_state.ml_model_path,
    key="ml_model_path_input"
)

if st.sidebar.button("Generate & Display ML Signals", key="generate_ml_signals"):
    if st.session_state.spy_data is not None and not st.session_state.spy_data.empty:
        if os.path.exists(st.session_state.ml_model_path):
            with st.spinner("Generating ML signals..."):
                try:
                    st.session_state.ml_signals = get_predictions(
                        raw_data_df=st.session_state.spy_data,
                        model_path=st.session_state.ml_model_path,
                        sequence_length=st.session_state.ml_sequence_length,
                        hidden_size=st.session_state.ml_hidden_size,
                        num_layers=st.session_state.ml_num_layers,
                        dropout=st.session_state.ml_dropout
                    )
                    if st.session_state.ml_signals is not None and not st.session_state.ml_signals.empty:
                        st.sidebar.success("ML signals generated.")
                    else:
                        st.sidebar.warning("ML signals generated but were empty.")
                except FileNotFoundError as fnf_error:
                    st.sidebar.error(f"ML Error: {fnf_error}")
                except Exception as e:
                    st.sidebar.error(f"ML signal generation failed: {e}")
        else:
            st.sidebar.error(f"ML Model not found at: {st.session_state.ml_model_path}")
    else:
        st.sidebar.warning("Load stock data first.")

# Optional: Button to trigger training
if st.sidebar.button("Train New ML Model (SPY - Default Params)", key="train_ml_model"):
    spy_data_path = f"data/cache/{selected_ticker}_daily.csv" # Assuming SPY data is loaded and cached
    if selected_ticker != "SPY":
        st.sidebar.warning("Training is currently hardcoded for SPY data if found in cache. Ensure SPY data is loaded/cached.")
        # Potentially fetch SPY data if not available, or make data_path configurable.
        # For now, we rely on it being cached if user loaded SPY data.
        spy_data_path = "data/cache/SPY_daily.csv"


    if os.path.exists(spy_data_path):
        st.sidebar.info("Starting ML model training... This may take a while.")
        try:
            # Construct the command
            # Ensure paths in train.py defaults are correct or make them configurable too.
            # Default model save path in train.py is 'models/lstm_model_v1.pth'
            command = [
                sys.executable, "src/ml/train.py",
                "--data_path", spy_data_path,
                # Using default epochs, batch_size etc. from train.py argparse defaults
                # Add more args here if you want to override them from Streamlit UI in future
            ]

            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

            # Live output display (optional, can be long)
            # with st.expander("Training Log"):
            #     stdout_container = st.empty()
            #     stderr_container = st.empty()
            #     stdout_log = ""
            #     stderr_log = ""
            #     while True:
            #         out = process.stdout.readline()
            #         err = process.stderr.readline()
            #         if not out and not err and process.poll() is not None:
            #             break
            #         if out:
            #             stdout_log += out
            #             stdout_container.text(stdout_log)
            #         if err:
            #             stderr_log += err
            #             stderr_container.text(stderr_log)

            stdout, stderr = process.communicate() # Wait for completion if not live streaming output

            if process.returncode == 0:
                st.sidebar.success("ML model training completed successfully.")
                st.session_state.ml_model_path = 'models/lstm_model_v1.pth' # Update to default path after training
                st.sidebar.info(f"Model potentially saved to default path used in train.py (e.g., {st.session_state.ml_model_path}).")
            else:
                st.sidebar.error(f"ML model training failed. Error:\n{stderr}")
        except Exception as e:
            st.sidebar.error(f"Error launching training process: {e}")
    else:
        st.sidebar.error(f"SPY data not found at {spy_data_path}. Please load SPY data first to enable caching for training.")


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
    rsi_to_plot = st.session_state.get('rsi_values', None) # Built-in RSI

    # Prepare custom indicators data for plotting
    custom_indicators_to_plot = []
    for ind_info in st.session_state.active_custom_indicators:
        # The structure expected by plot_generator will be: {'name': str, 'series': pd.Series, ...}
        # If ind_info['data'] is a DataFrame, we need to handle each column as a separate series or adapt plot_generator
        if isinstance(ind_info['data'], pd.Series):
            custom_indicators_to_plot.append({
                "name": ind_info['data'].name if ind_info['data'].name else ind_info['name'], # Ensure name attribute
                "series": ind_info['data']
                # 'subplot_row' will be handled by plot_generator
            })
        elif isinstance(ind_info['data'], pd.DataFrame):
            for col in ind_info['data'].columns:
                series_data = ind_info['data'][col]
                custom_indicators_to_plot.append({
                    "name": series_data.name if series_data.name else f"{ind_info['name']}_{col}",
                    "series": series_data
                })

    chart_fig = create_candlestick_chart(
        df=st.session_state.spy_data,
        ticker_symbol=selected_ticker,
        rsi_series=rsi_to_plot, # Pass existing RSI data
        rsi_period=st.session_state.rsi_period,
        custom_indicators_data=custom_indicators_to_plot, # Pass list of custom indicator data
        ml_signals=st.session_state.get('ml_signals') # Pass ML signals
    )
    st.plotly_chart(chart_fig, use_container_width=True)

    # Display raw data in an expander
    with st.expander(f"View Raw Data for {selected_ticker}"):
        st.dataframe(st.session_state.spy_data)

    if rsi_to_plot is not None:
        with st.expander(f"View RSI Data (Period: {st.session_state.rsi_period})"): # Built-in RSI
            st.dataframe(rsi_to_plot)

    if custom_indicators_to_plot:
        with st.expander("View Custom Indicator Data"):
            for i, ind_data in enumerate(custom_indicators_to_plot):
                st.markdown(f"**{ind_data['name']}**")
                if isinstance(ind_data['series'], pd.Series):
                    st.dataframe(ind_data['series'])
                elif isinstance(ind_data['series'], pd.DataFrame): # Should be series by now based on above logic
                    st.dataframe(ind_data['series'])


else:
    st.warning(f"No data loaded for {selected_ticker}. Please load data using the sidebar controls.")

st.sidebar.markdown("---")
st.sidebar.info("Built with Streamlit") # Updated credit

# To run this app: streamlit run src/app.py
