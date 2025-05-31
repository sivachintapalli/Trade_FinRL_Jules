# src/indicators/__init__.py

from .saty_phase_oscillator import SatyPhaseOscillator
from .custom_sma import SimpleMovingAverage # Assuming custom_sma.py contains SimpleMovingAverage
# Add other indicators here if you want to make them directly importable
# from .another_indicator import AnotherIndicator

__all__ = [
    "SatyPhaseOscillator",
    "SimpleMovingAverage"
    # "AnotherIndicator"
]
