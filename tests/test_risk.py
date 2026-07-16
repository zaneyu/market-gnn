"""Run 10 — graph-structured covariance & the minimum-variance portfolio.

Tests the covariance estimators, the GMVP, and the graph-vs-null evaluation. The
block-structure recovery test is the positive control (mirrors the planted-signal
discipline used across the repo): given the TRUE block graph, a graph estimator must
beat both the sample covariance and a degree-preserving rewire. Torch-free."""

import numpy as np
import pandas as pd
import pytest

from marketgnn import graph as G
from marketgnn import risk


# --------------------------------------------------------------- primitives
def test_gmvp_weights_analytic():
    # 2-asset diagonal cov -> min-var weights inversely proportional to variance
    cov = np.array([[0.04, 0.0], [0.0, 0.01]])
    w = risk.gmvp_weights(cov)
    # closed form Σ⁻¹1 / 1ᵀΣ⁻¹1 = [25, 100]/125 = [0.2, 0.8]
    assert w == pytest.approx([0.2, 0.8], abs=1e-9)
    assert w.sum() == pytest.approx(1.0)


def test_gmvp_weights_singular_falls_back_to_finite():
    # a singular covariance must not blow up: ridge fallback -> finite weights summing to 1
    cov = np.array([[1.0, 1.0], [1.0, 1.0]])  # rank 1, singular
    w = risk.gmvp_weights(cov)
    assert np.all(np.isfinite(w))
    assert w.sum() == pytest.approx(1.0)


def test_dense_adjacency_aligned_and_symmetric():
    nodes = ["A", "B", "C"]
    # directed: A->B and A->C only (like correlation_knn top-k)
    g = G.edges_from_pairs(pd.DataFrame({"src": ["A", "A"], "dst": ["B", "C"]}), nodes)
    adj = risk.dense_adjacency(g, nodes)
    assert adj.dtype == bool
    assert (adj == adj.T).all()             # symmetrized
    assert adj[0, 1] and adj[1, 0]          # A-B both ways after symmetrize
    assert not adj[1, 2]                     # B-C never linked
    assert not adj.diagonal().any()          # no self-edges


def test_dense_adjacency_raises_on_node_mismatch():
    g = G.edges_from_pairs(pd.DataFrame({"src": ["A"], "dst": ["B"]}), ["A", "B", "C"])
    with pytest.raises(AssertionError):
        risk.dense_adjacency(g, ["A", "B", "X"])  # order/name mismatch


def test_ledoit_wolf_shrinks_and_conditions():
    rng = np.random.default_rng(0)
    n, T = 20, 40  # T only 2x n -> sample cov is poorly conditioned
    R = rng.normal(0, 0.01, size=(T, n))
    S = risk.sample_cov(R)
    cov, shrink = risk.ledoit_wolf_cc(R)
    assert 0.0 <= shrink <= 1.0
    assert np.linalg.cond(cov) < np.linalg.cond(S)  # better conditioned than sample


def test_sample_cov_complete_case_drops_nan_rows():
    R = np.array([[0.01, 0.02], [np.nan, 0.03], [0.00, -0.01], [0.02, 0.01]])
    S = risk.sample_cov(R)
    assert S.shape == (2, 2)
    assert np.all(np.isfinite(S))  # the NaN row was dropped, not propagated


def test_estimator_is_pit():
    # covariance at the end of a window must not change when FUTURE rows are perturbed
    rng = np.random.default_rng(1)
    R = rng.normal(0, 0.01, size=(60, 5))
    window = R[:40]
    cov_a = risk.sample_cov(window)
    R2 = R.copy()
    R2[40:] += 5.0  # blow up the future
    cov_b = risk.sample_cov(R2[:40])  # same window
    assert np.allclose(cov_a, cov_b)


# --------------------------------------------------------------- graphical lasso
def test_glasso_spd_on_illconditioned_input():
    rng = np.random.default_rng(2)
    n, T = 10, 6  # T < n -> rank-deficient sample cov
    R = rng.normal(0, 0.01, size=(T, n))
    S = risk.sample_cov(R)
    penalty = np.full((n, n), 0.001)
    np.fill_diagonal(penalty, 0.0)
    Theta, Z, converged = risk._glasso_admm(S, penalty, max_iter=200)
    # Theta must be SPD despite the rank-deficient S
    assert np.allclose(Theta, Theta.T)
    assert np.all(np.linalg.eigvalsh(Theta) > 0)


def test_glasso_recovers_offgraph_zeros():
    # planted: a chain precision (0-1-2 linked); off-chain entries should be forced ~0
    rng = np.random.default_rng(3)
    n, T = 6, 400
    # build data with a block structure so on-edge covariance is real
    z = rng.normal(size=(T, 2))
    R = np.column_stack([z[:, 0], z[:, 0], z[:, 0], z[:, 1], z[:, 1], z[:, 1]]) * 0.01
    R += rng.normal(0, 0.002, size=(T, n))
    S = risk.sample_cov(R)
    adj = np.zeros((n, n), bool)
    for i, j in [(0, 1), (1, 2), (3, 4), (4, 5)]:
        adj[i, j] = adj[j, i] = True
    penalty = np.where(adj, 0.0001, 5.0)  # huge off-edge penalty
    np.fill_diagonal(penalty, 0.0)
    _Theta, Z, _c = risk._glasso_admm(S, penalty, max_iter=500)
    # cross-block off-graph entry (0,3) must be driven to ~0 in the sparse iterate
    assert abs(Z[0, 3]) < 1e-4
    assert abs(Z[0, 1]) > abs(Z[0, 3])  # an on-graph entry is retained relative to off-graph


def test_graph_glasso_falls_back_on_nonconvergence():
    rng = np.random.default_rng(4)
    R = rng.normal(0, 0.01, size=(60, 8))
    adj = np.zeros((8, 8), bool)
    adj[0, 1] = adj[1, 0] = True
    cov = risk.graph_glasso(R, adj, edge_penalty=0.001, offedge_penalty=1.0, max_iter=1)
    # max_iter=1 forces non-convergence -> fall back to A, still a finite PD covariance
    assert np.all(np.isfinite(cov))
    assert np.all(np.linalg.eigvalsh(cov) > 0)


# --------------------------------------------------------------- evaluation core
def _block_market(n_blocks=3, per=10, T=500, rho_in=0.7, seed=0):
    """Synthetic block-factor market: within-block correlation rho_in, else independent.
    The true economic-link graph is block co-membership."""
    rng = np.random.default_rng(seed)
    n = n_blocks * per
    block = np.repeat(np.arange(n_blocks), per)
    f = rng.normal(0, 0.01, size=(T, n_blocks))          # block factors
    idio = rng.normal(0, 0.01, size=(T, n))
    R = np.sqrt(rho_in) * f[:, block] + np.sqrt(1 - rho_in) * idio
    adj = (block[:, None] == block[None, :]) & ~np.eye(n, dtype=bool)
    return R, adj, block


def test_graph_masked_recovers_block_structure():
    """Positive control: given the TRUE block graph, graph-masked shrinkage yields a lower
    OOS GMVP realized vol than the sample covariance AND than a degree-preserving rewire —
    proving the method exploits real structure (not just any sparsity)."""
    R, adj, block = _block_market(seed=0)
    # degree-preserving rewire of the block adjacency (same degree, shuffled targets)
    rng = np.random.default_rng(1)
    adj_rewire = _rewire_adjacency(adj, rng)

    v_sample = risk.realized_vol(risk.rolling_gmvp_returns(R, risk.sample_cov, window=60, hold=5))
    v_true = risk.realized_vol(risk.rolling_gmvp_returns(
        R, lambda w: risk.graph_masked_cov(w, adj), window=60, hold=5))
    v_rewire = risk.realized_vol(risk.rolling_gmvp_returns(
        R, lambda w: risk.graph_masked_cov(w, adj_rewire), window=60, hold=5))

    assert v_true < v_sample, f"graph {v_true:.4f} should beat sample {v_sample:.4f}"
    assert v_true < v_rewire, f"true graph {v_true:.4f} should beat rewire {v_rewire:.4f}"


def _rewire_adjacency(adj, rng, n_swaps=200):
    """Degree-preserving double-edge swap on a boolean symmetric adjacency."""
    A = adj.copy()
    n = A.shape[0]
    edges = [(i, j) for i in range(n) for j in range(i + 1, n) if A[i, j]]
    for _ in range(n_swaps):
        if len(edges) < 2:
            break
        (a, b), (c, d) = [edges[k] for k in rng.choice(len(edges), 2, replace=False)]
        if len({a, b, c, d}) < 4:
            continue
        if A[a, d] or A[c, b]:
            continue
        A[a, b] = A[b, a] = A[c, d] = A[d, c] = False
        A[a, d] = A[d, a] = A[c, b] = A[b, c] = True
        edges = [(i, j) for i in range(n) for j in range(i + 1, n) if A[i, j]]
    return A


def test_portfolio_qlike_penalizes_worse_forecast():
    # a variance forecast far from the realized proxy scores a higher (worse) QLIKE
    realized = np.full(50, 1e-4)
    good = risk.portfolio_qlike(realized, np.full(50, 1e-4))
    bad = risk.portfolio_qlike(realized, np.full(50, 4e-4))
    assert good < bad
    assert good == pytest.approx(0.0, abs=1e-9)  # perfect forecast -> 0


def test_paired_squared_diff_has_expected_sign():
    # estimator A has genuinely lower-variance OOS returns than B -> mean paired diff < 0
    rng = np.random.default_rng(7)
    a = rng.normal(0, 0.01, size=300)
    b = rng.normal(0, 0.02, size=300)
    d = risk.paired_squared_diff(a, b)
    assert d.mean() < 0  # a is lower variance


def test_regime_spread_detects_concentration():
    """Structure helps only in the high-correlation regime -> the graph-vs-benchmark
    log-variance benefit is larger there, and the spread test detects it."""
    rng = np.random.default_rng(9)
    N = 200
    regime = np.array([True] * (N // 2) + [False] * (N // 2))  # first half = high-corr
    bench = rng.normal(0, 0.02, size=N)
    graph = bench.copy()
    graph[regime] = rng.normal(0, 0.01, size=regime.sum())  # graph much lower vol in high regime
    out = risk.regime_spread(bench, graph, regime)
    assert out["logvar_ratio_high"] < out["logvar_ratio_low"]  # more benefit (lower ratio) in high
    assert out["spread"] < 0  # high-minus-low benefit is negative (graph helps more in high)
