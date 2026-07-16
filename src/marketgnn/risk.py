"""Run 10 — graphs for risk, not alpha: graph-structured covariance estimation and
the global minimum-variance portfolio.

Runs 1–9 showed a stock-relationship graph adds no cross-sectional *return* signal.
Covariance estimation is a different problem where structural priors are known to help
(that is why Ledoit–Wolf shrinkage exists). This module asks whether a graph — a
*holdings-based* 13F co-holding graph, so using it on a *return* covariance is a genuine
external prior, not a tautology — improves out-of-sample covariance estimation, measured
by the realized volatility of the global minimum-variance portfolio (GMVP). Two ways the
graph enters: (A) a graph-masked shrinkage target, (B) a per-edge graphical lasso. Every
graph estimator must beat a degree-preserving rewire of the same graph (the topology null).

See docs/superpowers/specs/2026-07-16-graph-risk-covariance-design.md.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- covariance
def _complete_case(returns: np.ndarray) -> np.ndarray:
    """Drop rows (dates) with any missing return — a single NaN otherwise poisons the
    covariance and its inverse. Assumes all columns are present (liquid universe)."""
    R = np.asarray(returns, float)
    return R[~np.isnan(R).any(axis=1)]


def sample_cov(returns: np.ndarray) -> np.ndarray:
    """Complete-case sample covariance (ddof=1)."""
    R = _complete_case(returns)
    return np.cov(R, rowvar=False, ddof=1)


def _constant_correlation_target(S: np.ndarray) -> np.ndarray:
    """Ledoit–Wolf constant-correlation target F: diagonal = sample variances, off-diagonal
    = average sample correlation scaled by the individual volatilities."""
    var = np.diag(S)
    std = np.sqrt(var)
    denom = np.outer(std, std)
    with np.errstate(divide="ignore", invalid="ignore"):
        corr = np.where(denom > 0, S / denom, 0.0)
    n = S.shape[0]
    rbar = (corr.sum() - n) / (n * (n - 1)) if n > 1 else 0.0
    F = rbar * denom
    np.fill_diagonal(F, var)
    return F


def ledoit_wolf_cc(returns: np.ndarray) -> tuple[np.ndarray, float]:
    """Ledoit–Wolf (2004) linear shrinkage toward the constant-correlation target.
    Returns (cov, shrinkage∈[0,1])."""
    X = _complete_case(returns)
    T, n = X.shape
    Xc = X - X.mean(0)
    S = (Xc.T @ Xc) / T                       # MLE covariance (1/T, LW convention)
    var = np.diag(S)
    std = np.sqrt(var)
    F = _constant_correlation_target(S)

    # pi: sum of asymptotic variances of the sample covariance entries
    Xc2 = Xc**2
    pi_mat = (Xc2.T @ Xc2) / T - S**2
    pi = pi_mat.sum()

    # rho: sum of asymptotic covariances of F and S entries
    denom = np.outer(std, std)
    with np.errstate(divide="ignore", invalid="ignore"):
        corr = np.where(denom > 0, S / denom, 0.0)
        rbar = (corr.sum() - n) / (n * (n - 1)) if n > 1 else 0.0
        P3 = (Xc**3).T @ Xc / T
        theta_ii = P3 - var[:, None] * S       # (i,j) uses var_i
        theta_jj = P3.T - var[None, :] * S      # (i,j) uses var_j
        ratio = np.where(denom > 0, np.sqrt(np.outer(var, 1.0 / var)), 0.0)  # sqrt(var_i/var_j)
    rho_off = (rbar / 2.0) * ((1.0 / ratio) * theta_ii + ratio * theta_jj)
    np.fill_diagonal(rho_off, 0.0)
    rho = np.diag(pi_mat).sum() + rho_off[~np.eye(n, dtype=bool)].sum()

    gamma = float(np.sum((F - S) ** 2))
    kappa = (pi - rho) / gamma if gamma > 0 else 0.0
    shrink = float(max(0.0, min(1.0, kappa / T)))
    cov = shrink * F + (1.0 - shrink) * S
    return cov, shrink


def gmvp_weights(cov: np.ndarray) -> np.ndarray:
    """Unconstrained global minimum-variance weights w ∝ Σ⁻¹1, normalized to 1ᵀw = 1.
    Solves rather than inverts; ridges on a singular covariance so weights stay finite."""
    n = cov.shape[0]
    ones = np.ones(n)
    try:
        x = np.linalg.solve(cov, ones)
    except np.linalg.LinAlgError:
        ridge = max(1e-12, 1e-8 * np.trace(cov) / n)
        x = np.linalg.solve(cov + ridge * np.eye(n), ones)
    return x / x.sum()


# --------------------------------------------------------------------------- graph → cov
def dense_adjacency(graph, nodes) -> np.ndarray:
    """Aligned, symmetrized boolean adjacency [n,n] from a Graph. Hard-asserts the graph's
    node order matches ``nodes`` (guards the silent wrong-asset corruption when a provider
    is built in a different order than the covariance's columns). Symmetrizes because
    correlation_knn is a directed top-k graph."""
    assert list(graph.nodes) == list(nodes), "graph node order must match the covariance columns"
    n = len(nodes)
    A = np.zeros((n, n), dtype=bool)
    src, dst = graph.edge_index
    A[np.asarray(src), np.asarray(dst)] = True
    A = A | A.T
    np.fill_diagonal(A, False)
    return A


def graph_masked_cov(returns: np.ndarray, adj: np.ndarray, *, shrink: float | None = None) -> np.ndarray:
    """Estimator A — graph-informed shrinkage. Σ̂ = δ·T + (1−δ)·S with LW intensity δ applied
    everywhere, where the target T treats the graph as a **conditional-independence prior**:
    on-graph pairs shrink toward the constant-correlation target, off-graph pairs shrink toward
    **zero** (unlinked ⇒ low covariance), diagonal = sample variances. This is the fair test —
    if the graph correctly flags weakly-related pairs, shrinking them to zero beats shrinking
    everything to the same constant correlation (plain Ledoit–Wolf)."""
    S = sample_cov(returns)
    F = _constant_correlation_target(S)
    T = np.where(adj, F, 0.0)              # on-graph -> const-corr, off-graph -> 0
    np.fill_diagonal(T, np.diag(S))         # diagonal -> sample variance
    if shrink is None:
        _, shrink = ledoit_wolf_cc(returns)
    return shrink * T + (1.0 - shrink) * S


def _soft_threshold(a: np.ndarray, lam: np.ndarray) -> np.ndarray:
    return np.sign(a) * np.maximum(np.abs(a) - lam, 0.0)


def _glasso_admm(S: np.ndarray, penalty: np.ndarray, *, rho: float = 1.0, tol: float = 1e-4,
                 rel_tol: float = 1e-2, max_iter: int = 500) -> tuple[np.ndarray, np.ndarray, bool]:
    """SPD-guarded ADMM graphical lasso with a per-edge penalty matrix. Minimizes
    −logdet Θ + tr(SΘ) + ‖Λ∘Θ‖₁. Returns (Theta [PD precision], Z [sparse iterate carrying
    the exact off-graph zeros], converged). The Θ-update's analytic prox of −logdet always
    yields a PD Θ even on a rank-deficient S.

    Stopping uses the standard Boyd size-scaled primal/dual criterion (`tol` = absolute,
    `rel_tol` = relative) — a fixed Frobenius `tol` is meaningless for a 90×90 matrix (the norm
    sums over n² entries)."""
    n = S.shape[0]
    S = S + 1e-3 * np.trace(S) / n * np.eye(n)     # SPD precondition
    Z = np.zeros((n, n))
    U = np.zeros((n, n))
    Theta = np.eye(n)
    converged = False
    sqrt_p = float(n)                               # sqrt(n²) elements
    for _ in range(max_iter):
        M = rho * (Z - U) - S
        M = (M + M.T) / 2.0
        eig, Q = np.linalg.eigh(M)
        theta_eig = (eig + np.sqrt(eig**2 + 4.0 * rho)) / (2.0 * rho)   # prox of −logdet, > 0
        Theta = (Q * theta_eig) @ Q.T
        Theta = (Theta + Theta.T) / 2.0
        Zold = Z
        A = Theta + U
        Z = _soft_threshold(A, penalty / rho)
        np.fill_diagonal(Z, np.diag(A))            # diagonal unpenalized
        U = U + Theta - Z
        primal = np.linalg.norm(Theta - Z)
        dual = np.linalg.norm(rho * (Z - Zold))
        eps_pri = sqrt_p * tol + rel_tol * max(np.linalg.norm(Theta), np.linalg.norm(Z))
        eps_dual = sqrt_p * tol + rel_tol * np.linalg.norm(rho * U)
        if primal < eps_pri and dual < eps_dual:
            converged = True
            break
    return Theta, Z, converged


def graph_glasso(returns: np.ndarray, adj: np.ndarray, *, edge_penalty: float,
                 offedge_penalty: float, rho: float = 1.0, tol: float = 1e-4,
                 max_iter: int = 500) -> np.ndarray:
    """Estimator B — graph-penalized graphical lasso. Low L1 penalty on graph edges, high
    off-graph. Returns Σ̂ = Θ⁻¹; falls back to estimator A on non-convergence.

    Solved in **correlation space** (unit-diagonal): daily-return precisions are O(1e4), so a
    fixed `tol`/penalty on the raw covariance is meaningless — standardizing to the correlation
    matrix makes both scale-free. Σ̂ = D · corr-cov · D with D = diag(sample stdevs)."""
    S = sample_cov(returns)
    std = np.sqrt(np.diag(S))
    D = np.outer(std, std)
    with np.errstate(divide="ignore", invalid="ignore"):
        C = np.where(D > 0, S / D, 0.0)
    penalty = np.where(adj, edge_penalty, offedge_penalty)
    np.fill_diagonal(penalty, 0.0)
    Theta, _Z, converged = _glasso_admm(C, penalty, rho=rho, tol=tol, max_iter=max_iter)
    if not converged:
        return graph_masked_cov(returns, adj)
    corr_cov = np.linalg.solve(Theta, np.eye(S.shape[0]))   # (correlation-space) covariance
    return D * corr_cov


# --------------------------------------------------------------------------- evaluation
def rolling_gmvp_returns(returns: np.ndarray, estimator, *, window: int, hold: int) -> np.ndarray:
    """Walk forward: at each rebalance, estimate Σ̂ from the trailing ``window`` (strictly
    before the rebalance), form the GMVP, and record the realized daily portfolio returns
    over the next ``hold`` days. PIT by construction (window ends before the holding period).
    Returns the concatenated daily out-of-sample portfolio returns."""
    R = np.asarray(returns, float)
    T = R.shape[0]
    out = []
    for start in range(window, T - hold + 1, hold):
        cov = estimator(R[start - window:start])
        w = gmvp_weights(cov)
        out.append(R[start:start + hold] @ w)
    return np.concatenate(out) if out else np.array([])


def realized_vol(daily_returns: np.ndarray, *, periods: int = 252) -> float:
    """Annualized realized volatility of a daily return series."""
    d = np.asarray(daily_returns, float)
    d = d[np.isfinite(d)]
    return float(np.std(d, ddof=1) * np.sqrt(periods))


def portfolio_qlike(realized_var: np.ndarray, forecast_var: np.ndarray) -> float:
    """QLIKE loss of a variance forecast vs a realized-variance proxy (lower = better).
    Thin wrapper over evaluate.qlike so the horizon-matched forecast/realized pair is scored
    with the standard proper loss."""
    from .evaluate import qlike
    return qlike(realized_var, forecast_var)


def paired_squared_diff(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Per-period paired difference of squared DEMEANED returns, `(a−ā)² − (b−b̄)²` — the
    volatility-comparison estimand (raw squared returns would test Var+mean²). Its mean < 0
    iff series a has lower realized variance than b."""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    return (a - a.mean()) ** 2 - (b - b.mean()) ** 2


def _logvar_ratio(graph_ret: np.ndarray, bench_ret: np.ndarray) -> float:
    """log( var(graph) / var(bench) ) — scale-free; < 0 means the graph estimator gives a
    lower-variance portfolio than the benchmark."""
    vg = np.var(graph_ret, ddof=1)
    vb = np.var(bench_ret, ddof=1)
    return float(np.log(vg / vb)) if vg > 0 and vb > 0 else np.nan


def regime_spread(bench_ret: np.ndarray, graph_ret: np.ndarray, regime_high: np.ndarray) -> dict:
    """Is the graph's covariance benefit concentrated in the high-correlation regime? Compare
    the scale-free log-variance ratio (graph vs benchmark) in the high vs low regime. A more
    negative ratio = more benefit. ``spread = high − low`` < 0 means the graph helps *more*
    in the high regime. Scale-free by construction (uses a ratio, not a variance difference,
    so it is not driven by the higher absolute vol of crisis periods)."""
    bench_ret = np.asarray(bench_ret, float)
    graph_ret = np.asarray(graph_ret, float)
    hi = np.asarray(regime_high, bool)
    r_high = _logvar_ratio(graph_ret[hi], bench_ret[hi])
    r_low = _logvar_ratio(graph_ret[~hi], bench_ret[~hi])
    return {"logvar_ratio_high": r_high, "logvar_ratio_low": r_low,
            "spread": r_high - r_low, "n_high": int(hi.sum()), "n_low": int((~hi).sum())}


# --------------------------------------------------------------------------- Run 10 orchestration
def _ar1(x: np.ndarray) -> float:
    x = np.asarray(x, float)
    x = x - x.mean()
    d = float(x @ x)
    return float(np.clip((x[1:] @ x[:-1]) / d, 0.0, 0.9)) if d > 0 else 0.0


def evaluate_estimators(prices, provider, *, window: int = 252, rebal_freq: str = "M",
                        corr_window: int = 60, corr_k: int = 10,
                        edge_penalty: float = 0.0005, offedge_penalty: float = 0.05,
                        rewire_seeds=(0, 1, 2)):
    """Run 10: rolling monthly GMVP realized-vol + QLIKE comparison across covariance
    estimators, with paired significance vs Ledoit–Wolf and the multi-seed rewire null.
    Returns (estimator_table, regime_table, meta)."""
    from . import graph as G
    from .dataset import rebalance_dates
    from .evaluate import block_bootstrap_ci, newey_west_tstat, qlike, two_sided_p
    from .power import min_detectable_effect

    nodes = list(prices.columns)
    rets_df = prices.pct_change().dropna()
    idx = rets_df.index
    R = rets_df.to_numpy()
    reb = [d for d in rebalance_dates(idx, rebal_freq)]
    reb_pos = [p for p in (idx.get_indexer([d])[0] for d in reb) if p >= window]

    daily: dict[str, list] = {}
    fvar: dict[str, list] = {}
    rvar: dict[str, list] = {}
    conds, regime_corr, reb_dates = [], [], []

    def _record(name, cov, hold):
        w = gmvp_weights(cov)
        port = hold @ w
        daily.setdefault(name, []).append(port)
        fvar.setdefault(name, []).append(float(w @ cov @ w))
        rvar.setdefault(name, []).append(float(np.mean(port**2)))

    for k in range(len(reb_pos) - 1):
        p, p_next = reb_pos[k], reb_pos[k + 1]
        d = idx[p]
        win, hold = R[p - window:p], R[p:p_next]
        adj_co = dense_adjacency(provider["coholding"](d), nodes)
        adj_corr = dense_adjacency(
            G.correlation_knn(rets_df, d, nodes, window=corr_window, k=corr_k), nodes)
        S = sample_cov(win)
        _record("sample", S, hold)
        _record("ledoit_wolf", ledoit_wolf_cc(win)[0], hold)
        _record("diagonal", np.diag(np.diag(S)), hold)
        _record("masked_coholding", graph_masked_cov(win, adj_co), hold)
        _record("masked_correlation", graph_masked_cov(win, adj_corr), hold)
        _record("glasso_coholding",
                graph_glasso(win, adj_co, edge_penalty=edge_penalty,
                             offedge_penalty=offedge_penalty, max_iter=500), hold)
        for s in rewire_seeds:
            adj_rw = dense_adjacency(provider["rewire"](d, s), nodes)
            _record(f"__rewire{s}", graph_masked_cov(win, adj_rw), hold)
        conds.append(np.arange(len(hold)))  # placeholder for period lengths
        regime_corr.append(_avg_pairwise_corr(win))
        reb_dates.append(d)

    lw = np.concatenate(daily["ledoit_wolf"])
    rows = []
    display = ["sample", "diagonal", "ledoit_wolf", "masked_correlation",
               "masked_coholding", "glasso_coholding"]
    for name in display:
        dd = np.concatenate(daily[name])
        rv = realized_vol(dd)
        ql = qlike(np.array(rvar[name]), np.array(fvar[name]))
        # paired vol test vs Ledoit–Wolf (lower is better)
        diff = paired_squared_diff(dd, lw)
        t, _ = newey_west_tstat(diff)
        lo, hi = block_bootstrap_ci(diff, block=3)
        mde = min_detectable_effect(n_dates=len(diff), ic_sd=max(diff.std(ddof=1), 1e-12),
                                    phi=_ar1(diff), n_sims=300)
        rows.append({"estimator": name, "realized_vol": rv, "qlike": ql,
                     "vol_vs_lw_ratio": rv / realized_vol(lw), "paired_t_vs_lw": t,
                     "ci_lo": lo, "ci_hi": hi, "mde": mde})
    # rewire null: average realized vol over seeds
    rw_vols = [realized_vol(np.concatenate(daily[f"__rewire{s}"])) for s in rewire_seeds]
    rows.append({"estimator": "rewire_coholding(null)", "realized_vol": float(np.mean(rw_vols)),
                 "qlike": np.nan, "vol_vs_lw_ratio": float(np.mean(rw_vols)) / realized_vol(lw),
                 "paired_t_vs_lw": np.nan, "ci_lo": np.nan, "ci_hi": np.nan, "mde": np.nan})
    table = pd.DataFrame(rows)

    regime_table = regime_conditioning(daily, rvar, regime_corr, conds,
                                       graph="glasso_coholding", bench="ledoit_wolf")
    meta = {"n_rebalances": len(reb_dates), "window": window,
            "start": str(reb_dates[0].date()), "end": str(reb_dates[-1].date())}
    return table, regime_table, meta


def _avg_pairwise_corr(window: np.ndarray) -> float:
    """Average off-diagonal correlation over a return window — the regime proxy (PIT)."""
    c = np.corrcoef(window, rowvar=False)
    n = c.shape[0]
    return float((c.sum() - n) / (n * (n - 1)))


def regime_conditioning(daily, rvar, regime_corr, conds, *, graph, bench):
    """Split the OOS rebalances at the median trailing-correlation into high/low regimes and
    compare the scale-free log-variance ratio (graph vs benchmark) across them. A more
    negative ratio in the high regime = the graph helps risk more when correlations spike."""
    regime_corr = np.asarray(regime_corr)
    hi_reb = regime_corr >= np.median(regime_corr)          # per-rebalance high-corr flag
    # tag each daily observation with its rebalance's regime
    hi_daily, g_daily, b_daily = [], [], []
    for k, per in enumerate(conds):
        m = len(per)
        hi_daily.append(np.full(m, hi_reb[k]))
    hi_daily = np.concatenate(hi_daily)
    g = np.concatenate(daily[graph])
    b = np.concatenate(daily[bench])
    out = regime_spread(b, g, hi_daily)
    return pd.DataFrame([{"regime": "high_corr", "logvar_ratio": out["logvar_ratio_high"], "n": out["n_high"]},
                         {"regime": "low_corr", "logvar_ratio": out["logvar_ratio_low"], "n": out["n_low"]},
                         {"regime": "spread(high-low)", "logvar_ratio": out["spread"], "n": np.nan}])


def main():
    from .coholding import cusip_map, make_provider
    from .data.download import load_market

    nodes = list(cusip_map())
    provider, _graphs = make_provider(nodes, k=10)
    prices, _vol, _sectors, _mkt = load_market(
        synthetic=False, tickers=nodes, start="2014-01-01", end="2024-12-31")
    prices = prices.reindex(columns=nodes).dropna(how="all")
    # rebuild the provider on the exact price-column order so adjacency aligns
    provider, _graphs = make_provider(list(prices.columns), k=10)

    print("=== Run 10: graph-structured covariance & the GMVP (real data) ===")
    table, regime, meta = evaluate_estimators(prices, provider)
    print(f"universe {prices.shape[1]} names, {meta['n_rebalances']} monthly rebalances "
          f"({meta['start']}..{meta['end']}), 252d window\n")
    with pd.option_context("display.float_format", lambda v: f"{v:+.4f}"):
        print(table.to_string(index=False))
    print("\n=== regime conditioning (does the graph help more in high-correlation regimes?) ===")
    with pd.option_context("display.float_format", lambda v: f"{v:+.4f}"):
        print(regime.to_string(index=False))


if __name__ == "__main__":
    main()
