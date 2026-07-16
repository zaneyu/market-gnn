"""Reproducibility manifest. We do NOT redistribute vendor price data (Yahoo ToS),
so real-data results can't be reproduced by shipping a parquet. Instead we pin the
exact universe + date range and record a content hash of the fetched arrays.

Honest scope: this hash is a *drift detector*, not a promise of bit-reproducibility.
Yahoo retroactively re-adjusts historical closes for every later split/dividend, so a
clone re-fetching months later will generally get a DIFFERENT (still-correct) matrix and
a different hash. A mismatch therefore means "your pull differs from mine" (expected over
time), and real-data figures reproduce to ~2 significant figures, not byte-identically.
The seeded synthetic planted-recovery results are the exactly-reproducible ones, which is
why they carry the load-bearing power claims. The hash still pins column/date/rounding
conventions and catches gross universe/date drift.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

MANIFEST = Path(__file__).resolve().parents[3] / "data_manifest.json"


def content_hash(prices: pd.DataFrame) -> str:
    """Order-independent SHA-256 of the (rounded) price matrix + axes, so the same
    data hashes the same regardless of column order or float noise below 1e-6."""
    cols = sorted(prices.columns)
    p = prices[cols].round(6)
    h = hashlib.sha256()
    h.update(",".join(cols).encode())
    h.update(",".join(p.index.strftime("%Y-%m-%d")).encode())
    h.update(np.ascontiguousarray(p.to_numpy(np.float64)).tobytes())
    return h.hexdigest()


def write_manifest(prices: pd.DataFrame, *, tickers, start: str, end: str, path: Path = MANIFEST) -> dict:
    m = {
        "universe": sorted(tickers), "start": start, "end": end,
        "n_dates": int(prices.shape[0]), "n_assets": int(prices.shape[1]),
        "first_date": str(prices.index.min().date()), "last_date": str(prices.index.max().date()),
        "price_sha256": content_hash(prices),
    }
    path.write_text(json.dumps(m, indent=2))
    return m


def verify(prices: pd.DataFrame, path: Path = MANIFEST) -> bool:
    """True iff the current data matches the committed manifest hash."""
    if not path.exists():
        return False
    return content_hash(prices) == json.loads(path.read_text()).get("price_sha256")
