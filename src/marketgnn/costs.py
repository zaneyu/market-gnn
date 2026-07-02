"""Transaction-cost model: turn a gross rank-IC into the number a trader cares
about — the **breakeven cost** and the net-of-cost Sharpe of the decile long-short
portfolio a signal implies.

A gross positive with no cost accounting is how backtests lie. Short-horizon
reversal is the canonical example: strongly significant gross, but ~daily turnover
means realistic spreads eat it. This module quantifies exactly where it dies.

Costs are charged as `cost_bps * turnover`, where turnover is the one-way fraction
of the (dollar-neutral, unit-gross-per-leg) portfolio traded each rebalance.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def longshort_portfolio(panel: pd.DataFrame, *, q: float = 0.2):
    """Decile long-short weights and per-rebalance gross return + turnover.

    `panel` has columns (date, asset, sig, fwd). Long the top q by signal, short
    the bottom q, equal-weight within each leg (weights sum to +1 long / −1 short).
    """
    sig = panel.pivot(index="date", columns="asset", values="sig")
    fwd = panel.pivot(index="date", columns="asset", values="fwd")
    ranks = sig.rank(axis=1)
    n = sig.notna().sum(axis=1)
    k = (n * q).clip(lower=1).round()

    long_mask = ranks.gt(n - k, axis=0)
    short_mask = ranks.le(k, axis=0)
    W = long_mask.div(long_mask.sum(axis=1), axis=0) - short_mask.div(short_mask.sum(axis=1), axis=0)
    W = W.fillna(0.0)

    gross = (W * fwd.fillna(0.0)).sum(axis=1)
    turnover = W.diff().abs().sum(axis=1)  # first row is all-NaN -> dropped below
    return gross, turnover


def cost_summary(panel: pd.DataFrame, *, q: float = 0.2, periods_per_year: float = 252,
                 cost_grid=(1, 5, 10, 20)) -> dict:
    """Gross/net Sharpe and the breakeven cost (bps) for a signal's long-short book."""
    gross, turnover = longshort_portfolio(panel, q=q)
    gross, turnover = gross.iloc[1:], turnover.iloc[1:]  # drop the undefined first turnover
    g_mean, t_mean = gross.mean(), turnover.mean()

    def sharpe(r):
        sd = r.std(ddof=1)
        return float(r.mean() / sd * np.sqrt(periods_per_year)) if sd else np.nan

    breakeven_bps = float(g_mean / t_mean * 1e4) if t_mean else np.nan
    out = {
        "gross_mean_bps": float(g_mean * 1e4), "turnover": float(t_mean),
        "sharpe_gross": sharpe(gross), "breakeven_bps": breakeven_bps,
    }
    for c in cost_grid:
        net = gross - c * 1e-4 * turnover
        out[f"sharpe_net_{c}bps"] = sharpe(net)
        out[f"net_mean_bps_{c}bps"] = float(net.mean() * 1e4)
    return out


def main():
    from .data.download import load_market
    from .data.universe import default_universe
    from .signals import DEFAULT_CONTROLS, signal_panel

    prices, *_ = load_market(synthetic=False, tickers=default_universe(),
                             start="2014-01-01", end="2024-12-31")
    print("=== net-of-cost decile long-short (gross rank-IC is not enough) ===")
    for spec in DEFAULT_CONTROLS:
        panel = signal_panel(prices, spec)
        s = cost_summary(panel)
        print(f"\n{spec['name']}:  gross {s['gross_mean_bps']:+.1f} bps/rebal  "
              f"turnover {s['turnover']:.2f}  Sharpe(gross) {s['sharpe_gross']:+.2f}")
        print(f"  breakeven cost = {s['breakeven_bps']:.1f} bps  |  "
              f"Sharpe@5bps {s['sharpe_net_5bps']:+.2f}  @10bps {s['sharpe_net_10bps']:+.2f}  "
              f"@20bps {s['sharpe_net_20bps']:+.2f}")
    print("\nLarge-cap round-trip cost is ~2-5 bps; a signal whose breakeven is below "
          "that is arbitraged. This is why a gross IC needs a cost model before any claim.")


if __name__ == "__main__":
    main()
