"""
VolSurface
==========
Professional volatility surface construction and options analytics engine
for Deribit BTC/ETH options.

Quick start
-----------
>>> from src import DeribitClient, VolSurface, compute_iv, GreeksCalculator
>>> client = DeribitClient()
>>> snap   = client.get_vol_surface_snapshot("BTC")
>>> from src.iv_calculator import compute_iv_surface
>>> snap   = compute_iv_surface(snap)
>>> surface = VolSurface()
>>> surface.fit(snap)
>>> iv = surface.get_iv(K=70_000, T=0.25, F=68_000)
"""

from .deribit_client import DeribitClient, OptionTicker, OrderBook
from .greeks import GreeksCalculator, BSGreeks
from .iv_calculator import compute_iv, compute_iv_surface, bs_price, bs_vega
from .skew_analyzer import SkewAnalyzer
from .surface_fit import VolSurface, SVIParams, SVISliceFitter, SABRFitter
from .term_structure import TermStructureAnalyzer
from .backtester import DeltaHedgeBacktester
from .utils import get_logger, load_config, parse_instrument, time_to_expiry

__version__ = "1.1.0"
__author__ = "VolSurface"

__all__ = [
    # Data layer
    "DeribitClient",
    "OptionTicker",
    "OrderBook",
    # IV computation
    "compute_iv",
    "compute_iv_surface",
    "bs_price",
    "bs_vega",
    # Surface fitting
    "VolSurface",
    "SVIParams",
    "SVISliceFitter",
    "SABRFitter",
    # Analytics
    "GreeksCalculator",
    "BSGreeks",
    "SkewAnalyzer",
    "TermStructureAnalyzer",
    # Backtesting
    "DeltaHedgeBacktester",
    # Utils
    "get_logger",
    "load_config",
    "parse_instrument",
    "time_to_expiry",
]
