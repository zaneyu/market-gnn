"""Transaction-cost model: a gross positive must have a positive breakeven, net
Sharpe must fall as costs rise, and a pure random walk must not look tradable."""

import numpy as np
import pandas as pd

from marketgnn.costs import cost_summary
from marketgnn.signals import signal_panel

REVERSAL = {"name": "reversal_1d", "lookback": 1, "horizon": 1, "sign": -1}


def _market(phi, seed):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2015-01-01", periods=1500)
    r = np.zeros((1500, 40))
    eps = rng.normal(0, 0.012, size=(1500, 40))
    for t in range(1, 1500):
        r[t] = phi * r[t - 1] + eps[t]
    return pd.DataFrame(100 * np.exp(np.cumsum(r, axis=0)), index=dates, columns=[f"A{i:02d}" for i in range(40)])


def test_planted_reversal_has_positive_breakeven_and_cost_decay():
    s = cost_summary(signal_panel(_market(-0.18, 0), REVERSAL))
    assert s["gross_mean_bps"] > 0
    assert s["breakeven_bps"] > 0
    assert s["sharpe_gross"] > 0
    # net Sharpe must monotonically worsen as costs rise
    assert s["sharpe_net_5bps"] > s["sharpe_net_10bps"] > s["sharpe_net_20bps"]


def test_random_walk_is_not_tradable():
    s = cost_summary(signal_panel(_market(0.0, 1), REVERSAL))
    # no edge -> breakeven near/below zero, and it can't beat 5bps
    assert s["breakeven_bps"] < 1.0
    assert s["sharpe_net_5bps"] < 0.5
