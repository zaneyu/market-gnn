"""The harness must find a REAL non-null when one exists. Plant a mean-reverting
market (1-day cross-sectional reversal) and assert run_positive_controls recovers a
significant reversal_1d IC -- the real-data analogue of the planted lead-lag test."""

import numpy as np
import pandas as pd

from marketgnn.signals import run_positive_controls


def _mean_reverting_market(n_days=1500, n_assets=40, phi=-0.18, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2015-01-01", periods=n_days)
    assets = [f"R{i:02d}" for i in range(n_assets)]
    r = np.zeros((n_days, n_assets))
    eps = rng.normal(0, 0.012, size=(n_days, n_assets))
    for t in range(1, n_days):
        r[t] = phi * r[t - 1] + eps[t]  # negative AR(1) -> 1-day reversal
    prices = pd.DataFrame(100 * np.exp(np.cumsum(r, axis=0)), index=dates, columns=assets)
    return prices


def test_recovers_planted_reversal():
    prices = _mean_reverting_market()
    table = run_positive_controls(
        prices, warmup=260, controls=[{"name": "reversal_1d", "lookback": 1, "horizon": 1, "sign": -1}]
    )
    row = table.iloc[0]
    assert row.mean_ic > 0.05          # planted reversal is recovered...
    assert row.hac_t > 3               # ...and clearly significant under HAC
    assert row.fdr_sig


def test_no_spurious_signal_in_random_walk():
    rng = np.random.default_rng(1)
    dates = pd.bdate_range("2015-01-01", periods=1500)
    prices = pd.DataFrame(
        100 * np.exp(np.cumsum(rng.normal(0, 0.012, size=(1500, 40)), axis=0)),
        index=dates, columns=[f"W{i:02d}" for i in range(40)],
    )
    table = run_positive_controls(
        prices, warmup=260, controls=[{"name": "reversal_1d", "lookback": 1, "horizon": 1, "sign": -1}]
    )
    assert not table.iloc[0].fdr_sig   # no reversal planted -> no spurious signal
