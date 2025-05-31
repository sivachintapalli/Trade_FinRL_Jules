import pandas as pd
import numpy as np
import pytest
import plotly.graph_objects as go
from src.charts.plot_generator import create_candlestick_chart # Assuming src is in PYTHONPATH
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
    return calculate_rsi(sample_ohlcv_data['Close'].copy(), period=14) # Use 'Close' series and period


def test_create_candlestick_chart_returns_figure(sample_ohlcv_data):
    """Test that the function returns a Plotly Figure object."""
    fig = create_candlestick_chart(sample_ohlcv_data)
    assert isinstance(fig, go.Figure)

def test_create_candlestick_chart_no_rsi(sample_ohlcv_data):
    """Test chart generation without RSI data."""
    fig = create_candlestick_chart(sample_ohlcv_data, rsi_series=None)
    assert isinstance(fig, go.Figure)
    # Expect 1 trace: Candlestick only.
    assert len(fig.data) == 1
    assert fig.layout.yaxis.title.text == 'Price'
    assert not hasattr(fig.layout, 'yaxis2') # No second y-axis

def test_create_candlestick_chart_with_rsi(sample_ohlcv_data, sample_rsi_data):
    """Test chart generation with RSI data."""
    fig = create_candlestick_chart(sample_ohlcv_data, rsi_series=sample_rsi_data)
    assert isinstance(fig, go.Figure)
    # Expect 2 traces: Candlestick and RSI line.
    assert len(fig.data) == 2

    # Check y-axis titles for the two subplots
    assert fig.layout.yaxis.title.text == 'Price'  # Row 1
    assert fig.layout.yaxis2.title.text == 'RSI'   # Row 2 (RSI plot)

    # Check for RSI overbought/oversold lines (shapes)
    # These are horizontal lines on the second subplot (yref='y2')
    rsi_related_shapes = [shape for shape in fig.layout.shapes if shape.type == 'line' and shape.yref == 'y2']
    assert len(rsi_related_shapes) == 2 # One for overbought, one for oversold

    # Annotations are not added for hlines in the new function version
    # rsi_annotations = [ann for ann in fig.layout.annotations if ann.text in ["Overbought", "Oversold"]]
    # assert len(rsi_annotations) == 0 # No text annotations for these lines


def test_create_candlestick_chart_with_rsi_hidden_adapted(sample_ohlcv_data, sample_rsi_data):
    """Test that if rsi_series is None, RSI plot is not generated."""
    fig = create_candlestick_chart(sample_ohlcv_data, rsi_series=None) # Explicitly pass None
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 1 # Candlestick only
    assert hasattr(fig.layout, 'yaxis') and fig.layout.yaxis.title.text == 'Price'
    assert not hasattr(fig.layout, 'yaxis2')

    rsi_related_shapes = [shape for shape in fig.layout.shapes if hasattr(shape, 'yref') and shape.yref == 'y2']
    assert len(rsi_related_shapes) == 0


def test_create_candlestick_chart_custom_rsi_levels(sample_ohlcv_data, sample_rsi_data):
    """Test chart with custom RSI overbought/oversold levels."""
    custom_ob = 80
    custom_os = 20
    fig = create_candlestick_chart(
        sample_ohlcv_data,
        rsi_series=sample_rsi_data,
        overbought_level=custom_ob, # Parameter name changed
        oversold_level=custom_os    # Parameter name changed
    )
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 2 # Candlestick, RSI

    # Check if hlines are created at custom levels on the RSI subplot (y2)
    rsi_hlines = [shape for shape in fig.layout.shapes if shape.yref == 'y2' and shape.type == 'line']
    y_values_of_hlines = sorted([shape.y0 for shape in rsi_hlines])

    assert len(y_values_of_hlines) == 2
    assert y_values_of_hlines[0] == custom_os
    assert y_values_of_hlines[1] == custom_ob

if __name__ == '__main__':
    pytest.main([__file__])
