"""Hygiene test 6 -- the neighbor-aggregated return feature is point-in-time pure
and computes the intended graph aggregation."""

import numpy as np
import pandas as pd

from marketgnn.features import neighbor_return_feature
from marketgnn.graph import Graph


def _returns(seed=3, n=300, k=6):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-01", periods=n)
    cols = [f"A{i}" for i in range(k)]
    return pd.DataFrame(rng.normal(0, 0.01, size=(n, k)), index=dates, columns=cols)


def _line_graph(nodes):
    # 0->1->2->...  (each node's only out-neighbour is the next)
    src = np.arange(len(nodes) - 1, dtype=np.int64)
    dst = src + 1
    return Graph(np.vstack([src, dst]), np.ones(len(src), np.float32), np.asarray(nodes))


def test_neighbor_feature_is_pit_pure():
    returns = _returns()
    nodes = list(returns.columns)
    g = _line_graph(nodes)
    asof = returns.index[200]
    base = neighbor_return_feature(returns, asof, g, lookback=60)

    corrupted = returns.copy()
    corrupted.loc[corrupted.index > asof] = 5.0
    after = neighbor_return_feature(corrupted, asof, g, lookback=60)
    pd.testing.assert_series_equal(base, after)


def test_neighbor_feature_matches_manual_aggregation():
    returns = _returns()
    nodes = list(returns.columns)
    g = _line_graph(nodes)
    asof = returns.index[200]
    feat = neighbor_return_feature(returns, asof, g, lookback=60)

    trailing = (1 + returns.loc[:asof].iloc[-60:]).prod() - 1
    # node i's only neighbour is i+1, so feat[i] == trailing_return[i+1]
    assert np.isclose(feat.iloc[0], trailing.iloc[1])
    assert np.isnan(feat.iloc[-1])  # last node has no out-neighbour
