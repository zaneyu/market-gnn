"""Data access: cached daily OHLCV + an exogenous SPY benchmark.

Real data comes from yfinance (parquet-cached so a reviewer pays the download
once). Everything degrades gracefully: with ``synthetic=True`` or when yfinance /
network is unavailable, it returns the synthetic factor market so the pipeline
still runs offline.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..dataset import make_synthetic

CACHE = Path(__file__).resolve().parent / "cache"


def _cache_path(name: str) -> Path:
    CACHE.mkdir(exist_ok=True)
    return CACHE / f"{name}.parquet"


def download_prices(tickers, start: str, end: str, *, benchmark: str = "SPY", cache_key: str = "prices"):
    """Return (prices, volume, market_return). Cached to parquet by ``cache_key``."""
    px_p, vol_p, mkt_p = _cache_path(f"{cache_key}_px"), _cache_path(f"{cache_key}_vol"), _cache_path(f"{cache_key}_mkt")
    if px_p.exists() and vol_p.exists() and mkt_p.exists():
        return pd.read_parquet(px_p), pd.read_parquet(vol_p), pd.read_parquet(mkt_p).iloc[:, 0]

    import yfinance as yf  # lazy: only needed for a real pull

    syms = list(dict.fromkeys(list(tickers) + [benchmark]))
    raw = yf.download(syms, start=start, end=end, auto_adjust=True, progress=False)
    prices = raw["Close"].reindex(columns=syms).dropna(how="all")
    volume = raw["Volume"].reindex(columns=syms).reindex(prices.index)
    market = prices[benchmark].pct_change()

    prices = prices.drop(columns=[benchmark])
    volume = volume.drop(columns=[benchmark])
    prices.to_parquet(px_p)
    volume.to_parquet(vol_p)
    market.to_frame("market").to_parquet(mkt_p)
    return prices, volume, market


def load_market(*, synthetic: bool, tickers=None, start="2015-01-01", end="2024-12-31", **kw):
    """Single entry point used by the trainer. Falls back to synthetic on any error."""
    if synthetic:
        return make_synthetic(**kw)
    try:
        from .universe import default_sectors

        prices, volume, market = download_prices(tickers, start, end)
        sectors = default_sectors(list(prices.columns))
        return prices, volume, sectors, market
    except Exception as exc:  # noqa: BLE001 -- offline / missing dep -> synthetic
        print(f"[download] real data unavailable ({exc}); falling back to synthetic")
        return make_synthetic(**kw)
