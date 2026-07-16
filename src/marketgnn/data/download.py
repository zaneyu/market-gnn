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


def load_market(*, synthetic: bool, tickers=None, start="2015-01-01", end="2024-12-31",
                allow_synthetic_fallback: bool = False, **kw):
    """Single entry point used by the trainer.

    With ``synthetic=True`` returns the offline factor market. With ``synthetic=False``
    it returns REAL data and, by default, **raises** if that data is unavailable rather
    than silently substituting synthetic — so a results table headed "real prices" can
    never be quietly computed on a 60-name synthetic panel (which is a different
    universe entirely). Pass ``allow_synthetic_fallback=True`` to opt into the old
    degrade-gracefully behaviour for interactive/offline demos; it emits a LOUD banner
    and returns synthetic data for the *requested* tickers so at least the universe
    matches."""
    if synthetic:
        return make_synthetic(**kw)
    try:
        from .universe import default_sectors

        prices, volume, market = download_prices(tickers, start, end)
        sectors = default_sectors(list(prices.columns))
        return prices, volume, sectors, market
    except Exception as exc:  # noqa: BLE001 -- offline / missing dep
        if not allow_synthetic_fallback:
            raise RuntimeError(
                f"real market data unavailable ({exc}); refusing to silently return "
                f"synthetic data for a synthetic=False call. Pre-populate the parquet "
                f"cache, install yfinance, or pass allow_synthetic_fallback=True."
            ) from exc
        n = len(list(tickers)) if tickers is not None else 60
        print("\n" + "!" * 72 + f"\n[download] REAL DATA UNAVAILABLE ({exc}).\n"
              f"[download] FALLING BACK TO SYNTHETIC ({n} names) — results are NOT real.\n"
              + "!" * 72 + "\n")
        return make_synthetic(n_assets=n, **kw)
