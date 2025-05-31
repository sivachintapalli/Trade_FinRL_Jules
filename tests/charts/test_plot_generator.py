import pandas as pd
import numpy as np
import pytest
import plotly.graph_objects as go
from src.charts.plot_generator import generate_candlestick_chart # Assuming src is in PYTHONPATH
from src.core.indicator_calculator import calculate_rsi # For generating test RSI data

@pytest.fixture
def sample_ohlcv_data():
    """Provides sample OHLCV data for chart generation."""
    num_days = 30
    dates = pd.date_range(start='2023-01-01', periods=num_days, freq='D')
    data = {
        'Open': np.random.uniform(100, 105, num_days),
        'High': np.random.uniform(105, 110, num_days),
        'Low': np.random.uniform(95, 100, num_days),
        'Close': np.random.uniform(100, 105, num_days),
        'Volume': np.random.randint(100000, 500000, num_days)
    }
    df = pd.DataFrame(data, index=dates)
    # Ensure High is max and Low is min
    df['High'] = df[['Open', 'Close']].max(axis=1) + np.random.uniform(0, 2, num_days)
    df['Low'] = df[['Open', 'Close']].min(axis=1) - np.random.uniform(0, 2, num_days)
    return df

@pytest.fixture
def sample_rsi_data(sample_ohlcv_data):
    """Provides sample RSI data derived from ohlcv_data."""
    return calculate_rsi(sample_ohlcv_data.copy(), window=14)


def test_generate_candlestick_chart_returns_figure(sample_ohlcv_data):
    """Test that the function returns a Plotly Figure object."""
    fig = generate_candlestick_chart(sample_ohlcv_data)
    assert isinstance(fig, go.Figure)

def test_generate_candlestick_chart_no_rsi(sample_ohlcv_data):
    """Test chart generation without RSI data."""
    fig = generate_candlestick_chart(sample_ohlcv_data, show_rsi=False)
    assert isinstance(fig, go.Figure)
    # Expect 2 traces: Candlestick and Volume, if subplots are counted this way
    # Or, check number of subplots. Default layout has 2 rows (price, volume)
    assert len(fig.data) == 2
    assert fig.layout.yaxis.title.text == 'Price (USD)'
    assert fig.layout.yaxis2.title.text == 'Volume'
    assert len(fig.layout.annotations) == 0 # No RSI annotations for overbought/sold

    # Check number of rows in subplot structure
    # This depends on how make_subplots constructs the layout object.
    # For rows=2, fig.layout has yaxis, yaxis2.
    assert hasattr(fig.layout, 'yaxis') and fig.layout.yaxis.title.text == 'Price (USD)'
    assert hasattr(fig.layout, 'yaxis2') and fig.layout.yaxis2.title.text == 'Volume'
    assert not hasattr(fig.layout, 'yaxis3') # No third y-axis for RSI

def test_generate_candlestick_chart_with_rsi(sample_ohlcv_data, sample_rsi_data):
    """Test chart generation with RSI data."""
    fig = generate_candlestick_chart(sample_ohlcv_data, rsi_series=sample_rsi_data, show_rsi=True)
    assert isinstance(fig, go.Figure)
    # Expect 3 traces: Candlestick, Volume, and RSI line
    assert len(fig.data) == 3

    # Check y-axis titles for all three subplots
    assert fig.layout.yaxis.title.text == 'Price (USD)'  # Row 1
    assert fig.layout.yaxis2.title.text == 'Volume' # Row 2
    assert fig.layout.yaxis3.title.text == 'RSI'    # Row 3

    # Check for RSI overbought/oversold annotations (hline shapes)
    # Plotly adds hlines as shapes in the layout
    rsi_related_shapes = [shape for shape in fig.layout.shapes if shape.type == 'line' and shape.yref == 'y3']
    assert len(rsi_related_shapes) == 2 # One for overbought, one for oversold

    # Check annotations text for overbought/oversold
    # Annotations for hlines are also in layout.annotations
    rsi_annotations = [ann for ann in fig.layout.annotations if ann.text in ["Overbought", "Oversold"]]
    assert len(rsi_annotations) == 2


def test_generate_candlestick_chart_with_rsi_hidden(sample_ohlcv_data, sample_rsi_data):
    """Test chart generation with RSI data provided but show_rsi is False."""
    fig = generate_candlestick_chart(sample_ohlcv_data, rsi_series=sample_rsi_data, show_rsi=False)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 2 # Candlestick and Volume only
    assert hasattr(fig.layout, 'yaxis') and fig.layout.yaxis.title.text == 'Price (USD)'
    assert hasattr(fig.layout, 'yaxis2') and fig.layout.yaxis2.title.text == 'Volume'
    assert not hasattr(fig.layout, 'yaxis3') # No third y-axis for RSI

    rsi_related_shapes = [shape for shape in fig.layout.shapes if shape.yref == 'y3']
    assert len(rsi_related_shapes) == 0
    rsi_annotations = [ann for ann in fig.layout.annotations if ann.text in ["Overbought", "Oversold"]]
    assert len(rsi_annotations) == 0


def test_generate_candlestick_chart_custom_rsi_levels(sample_ohlcv_data, sample_rsi_data):
    """Test chart with custom RSI overbought/oversold levels."""
    custom_ob = 80
    custom_os = 20
    fig = generate_candlestick_chart(
        sample_ohlcv_data,
        rsi_series=sample_rsi_data,
        show_rsi=True,
        rsi_overbought=custom_ob,
        rsi_oversold=custom_os
    )
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 3 # Price, Volume, RSI

    # Check if hlines are created at custom levels
    # Shapes are used for hlines. We need to find the ones associated with y3 (RSI axis)
    rsi_hlines = [shape for shape in fig.layout.shapes if shape.yref == 'y3' and shape.type == 'line']
    y_values_of_hlines = sorted([shape.y0 for shape in rsi_hlines]) # y0 is where hline is drawn

    assert len(y_values_of_hlines) == 2
    assert y_values_of_hlines[0] == custom_os
    assert y_values_of_hlines[1] == custom_ob

if __name__ == '__main__':
    pytest.main([__file__])
