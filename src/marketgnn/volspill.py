"""Run 11 — volatility spillover over the co-holding graph (Diebold–Yilmaz-inspired).

Does a neighbour's volatility INNOVATION (short vol vs its own long-run level) predict a
stock's forward volatility beyond the stock's own vol history? The innovation design is the
load-bearing choice: a naive neighbour-vol-level signal is structurally confounded by
graph-clustered vol levels (the neighbour mean is a less noisy proxy of the shared level than
the own short-window vol — errors-in-variables), and the rewire null cannot catch that.
Innovations cancel shared levels; the residualized reading controls own σ20 AND σ250.

Identification limit (pre-registered): heterogeneous dynamic factor-vol EXPOSURE (block-level
vol regimes) is observationally equivalent to transmission in this design — as in standard DY
connectedness — so a positive is reported as "incremental predictive information along the
topology", never as causal spillover; a null is the stronger statement (the confound biases
positive). See docs/superpowers/specs/2026-07-16-vol-spillover-design.md.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import graph as G
from .evaluate import benjamini_hochberg, ic_summary, per_date_ic, rank_ic, two_sided_p
from .leadlag import _estimate_phi
from .power import min_detectable_effect


# --------------------------------------------------------------------------- primitives
def trailing_vol(returns: pd.DataFrame, *, lookback: int) -> pd.DataFrame:
    """Rolling std over the trailing ``lookback`` rows, strictly (t−lookback, t].
    ``min_periods = lookback``: a short history yields NaN, never a 2-observation vol."""
    return returns.rolling(lookback, min_periods=lookback).std(ddof=1)


def neighbour_innovation(innov_row: pd.Series, graph: G.Graph) -> pd.Series:
    """Edge-weighted mean of NEIGHBOURS' vol innovations (own value never enters).
    Non-finite neighbours are excluded and the weights renormalized over the finite ones;
    NaN only when no finite neighbour remains (isolated node / all neighbours missing) —
    one long-NaN name must not silently poison its ~k graph neighbours."""
    nodes = list(graph.nodes)
    vals = innov_row.reindex(nodes).to_numpy(float)
    n = len(nodes)
    out = np.full(n, np.nan)
    src, dst = graph.edge_index
    if src.size:
        w = np.abs(graph.edge_weight).astype(float)
        finite = np.isfinite(vals[dst])
        acc = np.zeros(n)
        sw = np.zeros(n)
        np.add.at(acc, src[finite], w[finite] * vals[dst[finite]])
        np.add.at(sw, src[finite], w[finite])
        nz = sw > 0
        out[nz] = acc[nz] / sw[nz]
    return pd.Series(out, index=nodes)


def _residualize_multi(y: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Cross-sectional OLS residual of y on the columns of X (with intercept), computed on
    the jointly-finite rows; other rows are NaN in the output (dropped by rank_ic). The
    multi-regressor sibling of ``leadlag._residualize`` (which is single-regressor)."""
    y = np.asarray(y, float)
    X = np.asarray(X, float)
    if X.ndim == 1:
        X = X[:, None]
    out = np.full_like(y, np.nan)
    m = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    if m.sum() < X.shape[1] + 2:
        return out
    Xc = X[m] - X[m].mean(0)
    yc = y[m] - y[m].mean()
    beta, *_ = np.linalg.lstsq(Xc, yc, rcond=None)
    out[m] = yc - Xc @ beta
    return out


# --------------------------------------------------------------------------- the experiment
def run_volspill(prices: pd.DataFrame, *, graph_provider, lookback: int = 20,
                 level_lookback: int = 250, horizon: int = 20, warmup: int = 260,
                 rewire_seeds=(0, 1, 2)) -> pd.DataFrame:
    """Per-date rank-IC of the neighbour vol-INNOVATION signal against forward vol, raw and
    residualized on own σ_short and σ_long (two regressors), over the real graph and its
    degree-preserving rewires, plus the own-vol persistence yardstick. Non-overlapping
    sampling (step = horizon); MDE with ic_sd/phi re-estimated per series; BH-FDR over the
    discovery family only (real-graph rows), controls reported with fdr_sig=False, q=NaN."""
    rets = prices.pct_change()
    vol_s = trailing_vol(rets, lookback=lookback)
    vol_l = trailing_vol(rets, lookback=level_lookback)
    innov = vol_s - vol_l
    idx = prices.index
    eval_pos = list(range(warmup, len(idx) - horizon, horizon))

    def collect(kind: str, seed: int) -> pd.DataFrame:
        recs = []
        for i in eval_pos:
            t = idx[i]
            g = graph_provider[kind](t, seed)
            sig = neighbour_innovation(innov.iloc[i], g)
            fwd = rets.iloc[i + 1:i + 1 + horizon].std(ddof=1).reindex(sig.index)
            recs.append(pd.DataFrame({
                "date": t, "asset": sig.index, "sig": sig.to_numpy(),
                "own_s": vol_s.iloc[i].reindex(sig.index).to_numpy(),
                "own_l": vol_l.iloc[i].reindex(sig.index).to_numpy(),
                "fwd": fwd.to_numpy(),
            }))
        return pd.concat(recs, ignore_index=True)

    def ic_pair(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
        ic_raw = per_date_ic(df["sig"], df["fwd"], df["date"]).dropna()
        resid = df.groupby("date", sort=True, group_keys=False).apply(
            lambda gdf: pd.Series(
                _residualize_multi(gdf["fwd"].to_numpy(),
                                   np.column_stack([gdf["own_s"], gdf["own_l"]])),
                index=gdf.index),
            include_groups=False)
        ic_res = per_date_ic(df["sig"], resid.reindex(df.index), df["date"]).dropna()
        return ic_raw, ic_res

    rows = []
    own_row = None
    for kind in ("coholding", "rewire"):
        seeds = rewire_seeds if kind == "rewire" else (0,)
        raw_list, res_list = [], []
        for seed in seeds:
            df = collect(kind, seed)
            ic_raw, ic_res = ic_pair(df)
            raw_list.append(ic_raw)
            res_list.append(ic_res)
            if own_row is None:  # edge-independent persistence yardstick (own σ20 -> fwd)
                ic_own = per_date_ic(df["own_s"], df["fwd"], df["date"]).dropna()
                s = ic_summary(ic_own)
                own_row = {"edges": "(none)", "signal": "own_vol",
                           "mean_ic": float(ic_own.mean()), "seed_std": 0.0,
                           "hac_t": s["hac_t"], "p": two_sided_p(s["hac_t"]),
                           "mde_80": _mde(ic_own), "n_dates": s["n"]}
        for name, series_list in (("spill_raw", raw_list), ("spill_resid", res_list)):
            means = [s.mean() for s in series_list]
            ic = series_list[0]
            s = ic_summary(ic)
            rows.append({"edges": kind, "signal": name,
                         "mean_ic": float(np.mean(means)),
                         "seed_std": float(np.std(means)) if len(means) > 1 else 0.0,
                         "hac_t": s["hac_t"], "p": two_sided_p(s["hac_t"]),
                         "mde_80": _mde(ic), "n_dates": s["n"]})
    rows.append(own_row)
    table = pd.DataFrame(rows)

    # BH-FDR over the discovery family only: the real-graph rows. Rewire and own_vol are
    # diagnostics (fdr_sig=False, q=NaN) — the leadlag.py convention.
    table["fdr_sig"] = False
    table["q"] = np.nan
    fam = table["edges"].eq("coholding")
    if fam.any():
        reject, q = benjamini_hochberg(table.loc[fam, "p"].fillna(1.0).to_numpy())
        table.loc[fam, "fdr_sig"] = reject
        table.loc[fam, "q"] = q
    return table


def _mde(ic: pd.Series) -> float:
    return min_detectable_effect(n_dates=len(ic), ic_sd=max(float(ic.std(ddof=1)), 1e-6),
                                 phi=_estimate_phi(ic), n_sims=400)


def confound_probe(prices: pd.DataFrame, graph: G.Graph, *, lookback: int = 20,
                   level_lookback: int = 250, horizon: int = 20,
                   warmup: int = 260) -> tuple[float, float]:
    """The control-validation probe (test 4): on a γ=0 clustered-level market, return
    (naive_ic, innov_ic) — the naive LEVEL signal residualized on own σ_short only (should
    show the spurious level confound) vs the innovation signal with the two-regressor
    control (should be ~null). Validates the CONTROL, not the signal."""
    rets = prices.pct_change()
    vol_s = trailing_vol(rets, lookback=lookback)
    vol_l = trailing_vol(rets, lookback=level_lookback)
    innov = vol_s - vol_l
    idx = prices.index
    naive_ics, innov_ics = [], []
    for i in range(warmup, len(idx) - horizon, horizon):
        fwd = rets.iloc[i + 1:i + 1 + horizon].std(ddof=1)
        own_s = vol_s.iloc[i]
        own_l = vol_l.iloc[i]
        naive_sig = neighbour_innovation(own_s, graph)          # LEVEL signal (neighbour σ20)
        innov_sig = neighbour_innovation(innov.iloc[i], graph)  # innovation signal
        r1 = _residualize_multi(fwd.to_numpy(), own_s.to_numpy())
        r2 = _residualize_multi(fwd.to_numpy(), np.column_stack([own_s, own_l]))
        naive_ics.append(rank_ic(naive_sig.to_numpy(), r1))
        innov_ics.append(rank_ic(innov_sig.to_numpy(), r2))
    return float(np.nanmean(naive_ics)), float(np.nanmean(innov_ics))


# --------------------------------------------------------------------------- planted market
def make_synthetic_spatial_arch(*, n_assets: int = 60, n_blocks: int = 6, n_days: int = 1500,
                                alpha: float = 0.08, beta: float = 0.5, gamma: float = 0.35,
                                block_omega_spread: float = 0.0, omega_base: float = 4e-6,
                                seed: int = 0) -> tuple[pd.DataFrame, G.Graph]:
    """Spatial-ARCH market with a PLANTED volatility spillover along a block graph:
    σ²_i(t) = ω_i + α·r²_i(t−1) + β·σ²_i(t−1) + γ·mean_N r²_j(t−1). With a row-normalized
    neighbour mean the aggregate persistence is α+β+γ, so stationarity requires < 1
    (asserted). ``block_omega_spread > 0`` draws per-BLOCK base variances
    ω_b = ω̄·exp(N(0, spread²)) — shared within a block — creating the clustered-level
    confound for the control-validation test. The γ-term is 0 for edgeless nodes."""
    assert alpha + beta + gamma < 1, "spatial-ARCH stationarity requires alpha+beta+gamma < 1"
    rng = np.random.default_rng(seed)
    assets = [f"V{i:03d}" for i in range(n_assets)]
    block = rng.integers(0, n_blocks, size=n_assets)
    sectors = pd.Series([f"blk{b}" for b in block], index=assets, name="sector")
    graph = G.sector_graph(sectors, assets)

    # row-normalized neighbour-mean matrix; zero row (γ-term 0) for edgeless nodes
    A = np.zeros((n_assets, n_assets))
    src, dst = graph.edge_index
    if src.size:
        np.add.at(A, (src, dst), 1.0)
    rs = A.sum(1, keepdims=True)
    A = np.divide(A, rs, out=np.zeros_like(A), where=rs > 0)

    omega_b = omega_base * np.exp(rng.normal(0.0, block_omega_spread, size=n_blocks))
    omega = omega_b[block]
    sig2 = omega / (1.0 - alpha - beta - gamma)      # start at ~unconditional variance
    r = np.zeros((n_days, n_assets))
    z = rng.normal(size=(n_days, n_assets))
    for t in range(n_days):
        r[t] = np.sqrt(sig2) * z[t]
        sig2 = omega + alpha * r[t] ** 2 + beta * sig2 + gamma * (A @ r[t] ** 2)
    prices = pd.DataFrame(100 * np.exp(np.cumsum(r, axis=0)),
                          index=pd.bdate_range("2015-01-01", periods=n_days), columns=assets)
    return prices, graph


# --------------------------------------------------------------------------- runner
def main():
    import argparse

    ap = argparse.ArgumentParser(description="Run 11: vol spillover over the co-holding graph")
    ap.add_argument("--synthetic-planted", action="store_true",
                    help="planted spatial-ARCH recovery (power proof)")
    args = ap.parse_args()

    if args.synthetic_planted:
        prices, graph = make_synthetic_spatial_arch()

        def coholding(asof, seed=0):
            return graph

        def rewire(asof, seed=0):
            return G.degree_preserving_rewire(graph, seed=seed)

        provider = {"coholding": coholding, "rewire": rewire}
        print("=== Run 11: PLANTED spatial-ARCH spillover recovery (synthetic) ===")
        table = run_volspill(prices, graph_provider=provider)
    else:
        from .coholding import cusip_map, make_provider
        from .data.download import load_market

        nodes = list(cusip_map())
        prices, _vol, _sectors, _mkt = load_market(
            synthetic=False, tickers=nodes, start="2014-01-01", end="2024-12-31")
        prices = prices.reindex(columns=nodes).dropna(how="all")
        provider, _graphs = make_provider(list(prices.columns), k=10)
        print("=== Run 11: vol spillover over the REAL co-holding graph (2014-2024) ===")
        table = run_volspill(prices, graph_provider=provider)

    cols = ["edges", "signal", "mean_ic", "seed_std", "hac_t", "p", "mde_80", "n_dates", "fdr_sig"]
    print(table[cols].to_string(index=False, float_format=lambda v: f"{v:+.3f}"))


if __name__ == "__main__":
    main()
