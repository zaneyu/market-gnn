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
    # pin the shrinkage intensity (any wrong-but-in-[0,1] value would still improve
    # conditioning, so a conditioning-only assert can't validate the pi/rho/gamma math).
    # NOTE: pinned to the numpy default_rng(0) stream; numpy keeps Generator.normal stable
    # in practice but doesn't contractually guarantee it — if this ever fails after a numpy
    # upgrade with a value still in ~[0.75, 0.95], re-pin rather than suspect the formula.
    assert shrink == pytest.approx(0.818, abs=0.02)
    assert np.linalg.cond(cov) < np.linalg.cond(S)  # better conditioned than sample
    # regime check: with abundant data AND heterogeneous (2-block) correlations — where the
    # constant-correlation target is a POOR fit — LW barely shrinks toward it (shrink -> ~0).
    # (Independent data would shrink fully, since there the const-corr target IS the truth.)
    z = rng.normal(size=(4000, 2))
    R_big = np.column_stack([z[:, 0]] * 10 + [z[:, 1]] * 10) * 0.01 + rng.normal(0, 0.005, size=(4000, n))
    _, shrink_big = risk.ledoit_wolf_cc(R_big)
    assert shrink_big < 0.1


def test_sample_cov_complete_case_drops_nan_rows():
    R = np.array([[0.01, 0.02], [np.nan, 0.03], [0.00, -0.01], [0.02, 0.01]])
    S = risk.sample_cov(R)
    clean = risk.sample_cov(R[[0, 2, 3]])  # the 3 rows without a NaN
    assert np.allclose(S, clean)  # dropped the RIGHT row, not just "some" rows


def test_rolling_gmvp_returns_is_pit():
    # the WINDOWING (not just np.cov) must be look-ahead free: perturbing far-future rows
    # cannot change the realized returns of early rebalances whose holding periods precede them
    rng = np.random.default_rng(1)
    R = rng.normal(0, 0.01, size=(120, 6))
    r1 = risk.rolling_gmvp_returns(R, risk.sample_cov, window=40, hold=5)
    R2 = R.copy()
    R2[80:] += 3.0  # blow up the far future (rows >= 80)
    r2 = risk.rolling_gmvp_returns(R2, risk.sample_cov, window=40, hold=5)
    # periods with start+hold <= 80 (obs [0:40]) are entirely before the perturbation
    assert np.allclose(r1[:40], r2[:40])
    assert not np.allclose(r1[40:], r2[40:])  # later periods DO change -> data is actually used
    # exact-window check (catches the classic off-by-one where the window includes the
    # rebalance-day row, which the far-future perturbation above cannot detect): the first
    # period must be EXACTLY hold-returns of the GMVP built from rows [0:window)
    w0 = risk.gmvp_weights(risk.sample_cov(R[0:40]))
    assert np.allclose(r1[:5], R[40:45] @ w0)


def test_graph_masked_cov_is_psd():
    """Regression guard for the indefinite-target BLOCKER: a sparse graph that CONTRADICTS a
    dense one-factor correlation structure makes the naive zeroed target indefinite (a block
    graph on a block market would not — the zeroed entries are ~0 anyway, which made the first
    version of this test vacuous under mutation). graph_masked_cov must PSD-project it, else
    the GMVP optimizes over a non-covariance and returns leverage artifacts."""
    rng = np.random.default_rng(5)
    n, T = 30, 120
    f = rng.normal(0, 0.01, size=T)                     # one dense market factor
    R = np.sqrt(0.6) * f[:, None] + np.sqrt(0.4) * rng.normal(0, 0.01, size=(T, n))
    adj = np.zeros((n, n), bool)
    for i in range(n):                                   # sparse ring — contradicts the factor
        adj[i, (i + 1) % n] = adj[(i + 1) % n, i] = True
    # precondition: the RAW (unprojected) target really is indefinite here, so the test
    # exercises the hard case and fails if the projection is ever removed
    S = risk.sample_cov(R)
    F = risk._constant_correlation_target(S)
    T_raw = np.where(adj, F, 0.0)
    np.fill_diagonal(T_raw, np.diag(S))
    assert np.linalg.eigvalsh(T_raw).min() < 0, "precondition: raw target must be indefinite"
    cov = risk.graph_masked_cov(R, adj)
    assert np.linalg.eigvalsh(cov).min() > -1e-10  # positive semidefinite after projection


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
    # two blocks {0,1,2},{3,4,5}; glasso on the CORRELATION matrix (unit scale) with a
    # MODERATE per-edge penalty should zero cross-block partial correlations while keeping
    # within-block ones — and it must be the log-det data term (not a giant penalty) that
    # keeps on-edge entries alive, so the test actually exercises the solver, not soft-threshold.
    rng = np.random.default_rng(3)
    n, T = 6, 800
    z = rng.normal(size=(T, 2))
    R = np.column_stack([z[:, 0]] * 3 + [z[:, 1]] * 3) * 0.01 + rng.normal(0, 0.004, size=(T, n))
    S = risk.sample_cov(R)
    std = np.sqrt(np.diag(S))
    C = S / np.outer(std, std)                        # correlation space, entries O(1)
    adj = np.zeros((n, n), bool)
    for i, j in [(0, 1), (0, 2), (1, 2), (3, 4), (3, 5), (4, 5)]:
        adj[i, j] = adj[j, i] = True
    penalty = np.where(adj, 0.02, 0.2)                # moderate, comparable to entry scale
    np.fill_diagonal(penalty, 0.0)
    Theta, Z, conv = risk._glasso_admm(C, penalty, max_iter=500)
    assert conv                                       # actually converges (not a fallback)
    assert np.allclose(Theta, Z, atol=1e-2)           # dense Θ matches the sparse iterate Z
    assert abs(Z[0, 3]) < 1e-6                         # cross-block partial corr zeroed
    assert abs(Z[0, 1]) > 10 * abs(Z[0, 3])           # within-block partial corr genuinely survives


def test_graph_glasso_falls_back_on_nonconvergence():
    rng = np.random.default_rng(4)
    R = rng.normal(0, 0.01, size=(60, 8))
    adj = np.zeros((8, 8), bool)
    adj[0, 1] = adj[1, 0] = True
    cov = risk.graph_glasso(R, adj, edge_penalty=0.001, offedge_penalty=1.0, max_iter=1)
    # max_iter=1 forces non-convergence -> must FALL BACK to estimator A exactly (not just
    # return some finite PD matrix from a 1-iteration glasso)
    assert np.allclose(cov, risk.graph_masked_cov(R, adj))
    assert np.linalg.eigvalsh(cov).min() > -1e-10


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

    v_sample = risk.realized_vol(risk.rolling_gmvp_returns(R, risk.sample_cov, window=60, hold=5))
    v_true = risk.realized_vol(risk.rolling_gmvp_returns(
        R, lambda w: risk.graph_masked_cov(w, adj), window=60, hold=5))
    # average the rewire null over several seeds (one Maslov-Sneppen draw is noisy) so the
    # comparison isn't a lucky single rewire — matches evaluate_estimators' multi-seed null
    v_rewire = np.mean([
        risk.realized_vol(risk.rolling_gmvp_returns(
            R, lambda w, a=_rewire_adjacency(adj, np.random.default_rng(s)): risk.graph_masked_cov(w, a),
            window=60, hold=5))
        for s in range(5)])

    assert v_true < v_sample, f"graph {v_true:.4f} should beat sample {v_sample:.4f}"
    assert v_true < v_rewire, f"true graph {v_true:.4f} should beat mean rewire {v_rewire:.4f}"


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


def test_format_estimator_table_preserves_tiny_ci_columns():
    # the CI/MDE columns live on the daily variance-difference scale (~1e-7); a naive
    # +.4f renders them all as +0.0000 (unreadable). The formatter must keep them legible
    # while the human-scale columns stay fixed-point.
    tbl = pd.DataFrame([{"estimator": "x", "realized_vol": 0.1487, "qlike": 1.55,
                         "vol_vs_lw_ratio": 1.0, "paired_t_vs_lw": -1.55,
                         "ci_lo": 2.9e-7, "ci_hi": 1.06e-6, "mde": 6.4e-7}])
    out = risk.format_estimator_table(tbl)
    assert "2.9e-07" in out and "1.06e-06" in out         # tiny values survive
    assert "+0.1487" in out and "-1.5500" in out           # human columns fixed-point
