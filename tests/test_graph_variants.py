"""Shrinkage and frozen graph variants (added to test the 'shouldn't the graph be
static?' hypothesis): shrinkage stays PIT-pure and denoises; frozen is one graph
reused across all dates while features keep updating."""

import numpy as np
import pandas as pd

from marketgnn.dataset import build_dataset, make_synthetic, rebalance_dates
from marketgnn.graph import correlation_knn


def _returns(seed=1, n=400, k=20):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-01", periods=n)
    cols = [f"A{i}" for i in range(k)]
    return pd.DataFrame(rng.normal(size=(n, k)), index=dates, columns=cols)


def test_shrinkage_graph_is_pit_pure():
    r = _returns()
    nodes = list(r.columns)
    asof = r.index[250]
    base = correlation_knn(r, asof, nodes, window=120, k=5, shrinkage="lw")
    corrupt = r.copy()
    corrupt.loc[corrupt.index > asof] = 999.0
    after = correlation_knn(corrupt, asof, nodes, window=120, k=5, shrinkage="lw")
    assert np.array_equal(base.edge_index, after.edge_index)
    assert np.allclose(base.edge_weight, after.edge_weight)


def test_shrinkage_pulls_correlations_toward_zero():
    r = _returns()
    nodes = list(r.columns)
    asof = r.index[250]
    raw = correlation_knn(r, asof, nodes, window=60, k=5)
    shrunk = correlation_knn(r, asof, nodes, window=60, k=5, shrinkage=0.5)
    # a 0.5 identity shrink halves off-diagonal magnitudes
    assert np.mean(np.abs(shrunk.edge_weight)) < np.mean(np.abs(raw.edge_weight))


def test_frozen_graph_is_identical_across_dates():
    prices, volume, sectors, market = make_synthetic(n_days=800, n_assets=30, n_sectors=4, seed=0)
    rebal = rebalance_dates(prices.index, "W")
    ds = build_dataset(prices, volume, sectors, market, rebal, graph_kind="frozen",
                       label_horizon=5, corr_window=60, k=6, warmup=520)
    graphs = list(ds.graphs.values())
    assert len(graphs) > 5
    first = graphs[0]
    assert all(np.array_equal(g.edge_index, first.edge_index) for g in graphs)
    # ...but the neighbour feature still varies date to date (features aren't frozen)
    nbr = ds.X["nbr_ret"].groupby(level="date").mean()
    assert nbr.nunique() > 1
