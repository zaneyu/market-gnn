"""Shared synthetic fixtures. All deterministic (seeded) so failures are real."""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def panel():
    """Synthetic prices/volume for 8 assets over 400 business days."""
    rng = np.random.default_rng(42)
    dates = pd.bdate_range("2020-01-01", periods=400)
    assets = [f"A{i}" for i in range(8)]
    rets = rng.normal(0.0003, 0.02, size=(len(dates), len(assets)))
    prices = pd.DataFrame(100 * np.exp(np.cumsum(rets, axis=0)), index=dates, columns=assets)
    volume = pd.DataFrame(rng.lognormal(12, 0.5, size=prices.shape), index=dates, columns=assets)
    return prices, volume, assets
