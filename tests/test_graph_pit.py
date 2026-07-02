"""Hygiene test 3 -- graph point-in-time purity: the graph at ``asof`` is
identical whether or not future rows exist (or are corrupted) in the input."""

import numpy as np
import pandas as pd

from marketgnn.graph import (
    add_self_loops,
    correlation_knn,
    degree_preserving_rewire,
    in_out_degrees,
    match_random,
    random_graph,
    sector_graph,
    symmetrize,
)


def _returns(seed=1, n=300, k=8):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-01", periods=n)
    cols = [f"A{i}" for i in range(k)]
    return pd.DataFrame(rng.normal(size=(n, k)), index=dates, columns=cols)


def test_correlation_graph_is_pit_pure():
    returns = _returns()
    nodes = list(returns.columns)
    asof = returns.index[200]
    base = correlation_knn(returns, asof, nodes, window=120, k=3)

    # (a) appending future rows changes nothing
    extra = _returns(seed=9, n=50).set_axis(
        pd.bdate_range(returns.index[-1] + pd.offsets.BDay(1), periods=50)
    )
    extra.columns = returns.columns
    appended = correlation_knn(pd.concat([returns, extra]), asof, nodes, window=120, k=3)

    # (b) corrupting everything strictly after asof changes nothing
    corrupted = returns.copy()
    corrupted.loc[corrupted.index > asof] = 999.0
    corrupt = correlation_knn(corrupted, asof, nodes, window=120, k=3)

    for other in (appended, corrupt):
        assert np.array_equal(base.edge_index, other.edge_index)
        assert np.allclose(base.edge_weight, other.edge_weight)


def test_knn_degree_is_capped():
    returns = _returns()
    nodes = list(returns.columns)
    g = correlation_knn(returns, returns.index[200], nodes, window=120, k=3)
    # each node emits at most k out-edges
    counts = np.bincount(g.edge_index[0], minlength=len(nodes))
    assert counts.max() <= 3


def test_sector_and_random_are_deterministic():
    nodes = [f"A{i}" for i in range(8)]
    sectors = pd.Series(["tech", "tech", "fin", "fin", "energy", "energy", "tech", "fin"], index=nodes)
    assert np.array_equal(
        sector_graph(sectors, nodes).edge_index, sector_graph(sectors, nodes).edge_index
    )
    assert np.array_equal(
        random_graph(nodes, degree=3, seed=0).edge_index,
        random_graph(nodes, degree=3, seed=0).edge_index,
    )


def test_random_control_matches_degree():
    returns = _returns()
    nodes = list(returns.columns)
    real = correlation_knn(returns, returns.index[200], nodes, window=120, k=3)
    null = match_random(real, seed=0)
    assert abs(null.avg_degree - real.avg_degree) <= 1.0


def test_degree_preserving_null_matches_full_degree_sequence():
    returns = _returns(k=12)
    nodes = list(returns.columns)
    real = correlation_knn(returns, returns.index[200], nodes, window=120, k=4)
    null = degree_preserving_rewire(real, seed=0)
    out_r, in_r = in_out_degrees(real)
    out_n, in_n = in_out_degrees(null)
    # the fair H2 null preserves BOTH in- and out-degree of every node exactly
    assert np.array_equal(out_r, out_n)
    assert np.array_equal(in_r, in_n)
    # ...while actually randomizing which neighbours (not a no-op)
    assert not np.array_equal(real.edge_index, null.edge_index)
    # no self-loops or duplicate edges introduced
    assert not (null.edge_index[0] == null.edge_index[1]).any()
    assert len({(int(s), int(d)) for s, d in null.edge_index.T}) == null.num_edges


def test_symmetrize_and_self_loops():
    returns = _returns()
    nodes = list(returns.columns)
    g = correlation_knn(returns, returns.index[200], nodes, window=120, k=3)
    sym = symmetrize(g)
    edges = {(int(s), int(d)) for s, d in sym.edge_index.T}
    assert all((j, i) in edges for i, j in edges)  # every edge has its reverse
    loops = add_self_loops(sym)
    assert all((i, i) in {(int(s), int(d)) for s, d in loops.edge_index.T} for i in range(len(nodes)))
