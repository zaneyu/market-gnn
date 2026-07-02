"""Data access layer: cached OHLCV + exogenous benchmark, and the point-in-time
universe / delisting handling that defends against survivorship bias."""

from .download import load_market

__all__ = ["load_market"]
