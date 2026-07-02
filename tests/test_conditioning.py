"""Liquidity conditioning must recover a reversal effect that is PLANTED only in
the illiquid names -- strong in the low-liquidity group, absent in the high, and a
significant low-minus-high spread."""

import numpy as np
import pandas as pd

from marketgnn.conditioning import run_liquidity_conditioning

REVERSAL = {"name": "reversal_1d", "lookback": 1, "horizon": 1, "sign": -1}


def _liquidity_split_market(n_days=1500, per_group=15, seed=0):
    """3 liquidity groups by construction: the LOW-liquidity group mean-reverts
    (planted reversal), the HIGH-liquidity group is a random walk."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2015-01-01", periods=n_days)
    phis = {0: -0.25, 1: -0.10, 2: 0.0}          # reversal strength by group
    vol_levels = {0: 1e5, 1: 1e6, 2: 1e7}         # dollar-volume level by group
    cols, rets, vols = [], [], []
    for g in range(3):
        for j in range(per_group):
            cols.append(f"G{g}_{j:02d}")
            e = rng.normal(0, 0.012, size=n_days)
            r = np.zeros(n_days)
            for t in range(1, n_days):
                r[t] = phis[g] * r[t - 1] + e[t]
            rets.append(r)
            vols.append(np.full(n_days, vol_levels[g]))
    prices = pd.DataFrame(100 * np.exp(np.cumsum(np.array(rets).T, axis=0)), index=dates, columns=cols)
    volume = pd.DataFrame(np.array(vols).T, index=dates, columns=cols)
    return prices, volume


def test_reversal_concentrates_in_illiquid():
    prices, volume = _liquidity_split_market()
    table, spread = run_liquidity_conditioning(prices, volume, spec=REVERSAL, warmup=260)
    low = table[table.group == 0].iloc[0]
    high = table[table.group == 2].iloc[0]

    assert low.mean_ic > 0.05 and low.hac_t > 3        # planted reversal recovered in illiquid
    assert abs(high.mean_ic) < low.mean_ic / 2         # ~absent in the most liquid
    assert spread["mean_spread"] > 0 and spread["p"] < 0.05   # gradient is significant
