"""The pipeline must RECOVER a planted lead-lag effect over the true graph and
find nothing over the degree-preserving rewire null. This is the power validation
that makes a real-data null meaningful rather than just underpowered."""

import numpy as np

from marketgnn.leadlag import leadlag_signal, make_synthetic_leadlag, run_leadlag


def test_recovers_planted_leadlag_and_rewire_is_null():
    prices, volume, sectors, market, _ = make_synthetic_leadlag(
        n_days=1200, n_assets=48, n_blocks=6, beta_ll=0.4, h=5, seed=0
    )
    table = run_leadlag(prices, sectors, market, edge_kinds=("sector", "rewire"),
                        label_horizon=5, lookback=5, warmup=260, rewire_seeds=(0, 1))
    ll_true = table[(table.edges == "sector") & (table.signal == "leadlag")].iloc[0]
    ll_null = table[(table.edges == "rewire") & (table.signal == "leadlag")].iloc[0]

    # planted effect over the TRUE (block) graph is recovered and significant
    assert ll_true.mean_ic > 0.03
    assert ll_true.fdr_sig
    # the degree-preserving rewire null does NOT spuriously find it
    assert abs(ll_null.mean_ic) < ll_true.mean_ic / 2


def test_leadlag_signal_is_pit_pure():
    import pandas as pd

    from marketgnn.graph import sector_graph

    prices, volume, sectors, market, _ = make_synthetic_leadlag(n_days=600, n_assets=24, seed=1)
    rets = prices.pct_change()
    g = sector_graph(sectors, list(prices.columns))
    asof = rets.index[400]
    base = leadlag_signal(rets, asof, g, lookback=5)
    corrupt = rets.copy()
    corrupt.loc[corrupt.index > asof] = 9.0
    after = leadlag_signal(corrupt, asof, g, lookback=5)
    pd.testing.assert_series_equal(base, after)
