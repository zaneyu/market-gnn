"""Lead-lag / momentum-spillover experiment -- the channel by which a graph can
actually add cross-sectional *return* signal (Cohen-Frazzini economic links,
Menzly-Ozbas cross-industry lead-lag, Moskowitz-Grinblatt industry momentum).

The signal is strictly lagged: at date t, a name's predictor is the graph-weighted
mean of its neighbours' *trailing* return over (t-h, t]; the target is the name's
*forward* return over (t, t+h]. Neighbours' past -> own future. Zero parameters, so
there is nothing to overfit -- the only question is whether the linkage carries
information, and whether it survives (a) a degree-preserving rewire null, (b) a
control for the name's own momentum, and (c) purged/PIT evaluation.

Crucially, `make_synthetic_leadlag` plants a KNOWN lead-lag effect so we can prove
the pipeline *recovers* it (power) before trusting a null on real data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import graph as G
from .dataset import make_graph, rebalance_dates
from .evaluate import benjamini_hochberg, ic_summary, per_date_ic, two_sided_p
from .features import forward_return
from .power import min_detectable_effect


def leadlag_signal(returns: pd.DataFrame, asof, graph, *, lookback: int) -> pd.Series:
    """Edge-weighted mean of neighbours' trailing `lookback`-day return (PIT)."""
    nodes = list(graph.nodes)
    hist = returns.loc[:asof].iloc[-lookback:].reindex(columns=nodes)
    node_ret = ((1 + hist).prod() - 1).to_numpy()
    src, dst = graph.edge_index
    w = np.abs(graph.edge_weight)
    out = np.full(len(nodes), np.nan)
    if src.size:
        acc = np.zeros(len(nodes))
        sw = np.zeros(len(nodes))
        np.add.at(acc, src, w * node_ret[dst])
        np.add.at(sw, src, w)
        nz = sw > 0
        out[nz] = acc[nz] / sw[nz]
    return pd.Series(out, index=nodes)


def own_momentum(returns: pd.DataFrame, asof, nodes, *, lookback: int) -> pd.Series:
    """The name's OWN trailing return -- the control the lead-lag signal must beat."""
    hist = returns.loc[:asof].iloc[-lookback:].reindex(columns=nodes)
    return (1 + hist).prod() - 1


def _estimate_phi(ic: pd.Series) -> float:
    """Lag-1 autocorrelation of the IC series, clamped to [0, 0.9], so the MDE's
    AR(1) assumption is self-consistent with the series it describes."""
    x = ic.dropna().to_numpy()
    if len(x) < 5:
        return 0.3
    x = x - x.mean()
    denom = float(x @ x)
    if denom == 0:
        return 0.0
    return float(np.clip((x[1:] @ x[:-1]) / denom, 0.0, 0.9))


def _residualize(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Cross-sectional residual of y after removing x's linear fit (both demeaned)."""
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 3:
        return y
    xc = x - np.nanmean(x[m])
    yc = y - np.nanmean(y[m])
    denom = np.nansum(xc[m] ** 2)
    b = np.nansum(xc[m] * yc[m]) / denom if denom else 0.0
    return yc - b * xc


def run_leadlag(
    prices, sectors, market=None, *, edge_kinds=("sector", "correlation", "rewire"),
    label_horizon=5, lookback=5, corr_window=60, k=10, warmup=260, rebal_freq="W",
    rewire_seeds=(0, 1, 2),
) -> pd.DataFrame:
    """Per-date rank-IC of the lead-lag signal (raw and own-momentum-residualized),
    with a degree-preserving rewire null and each row's minimum detectable effect."""
    rets = prices.pct_change()
    rebal = [d for d in rebalance_dates(prices.index, rebal_freq)
             if prices.index.get_indexer([d])[0] >= warmup
             and prices.index.get_indexer([d])[0] + label_horizon < len(prices.index)]
    nodes = list(prices.columns)

    def graph_for(kind, asof, seed=0):
        if kind == "rewire":
            real = make_graph("correlation", rets, sectors, asof, nodes, corr_window=corr_window, k=k)
            return G.degree_preserving_rewire(real, seed=seed)
        return make_graph(kind, rets, sectors, asof, nodes, corr_window=corr_window, k=k)

    rows = []
    own_ic_once = None  # own-momentum control is edge-independent -> emit one row
    for kind in edge_kinds:
        seeds = rewire_seeds if kind == "rewire" else (0,)
        seed_ic_raw, seed_ic_res, seed_ic_own = [], [], []
        for seed in seeds:
            recs = []
            for asof in rebal:
                g = graph_for(kind, asof, seed)
                ll = leadlag_signal(rets, asof, g, lookback=lookback)
                om = own_momentum(rets, asof, nodes, lookback=lookback).reindex(ll.index)
                fwd = forward_return(prices, asof, list(ll.index), label_horizon)
                recs.append(pd.DataFrame({"date": asof, "asset": ll.index, "ll": ll.to_numpy(),
                                          "om": om.to_numpy(), "fwd": fwd.to_numpy()}))
            df = pd.concat(recs, ignore_index=True)
            ic_raw = per_date_ic(df["ll"], df["fwd"], df["date"]).dropna()
            ic_own = per_date_ic(df["om"], df["fwd"], df["date"]).dropna()
            # lead-lag beyond own momentum: IC of ll vs the own-momentum-residualized fwd return
            resid = df.groupby("date", sort=True, group_keys=False).apply(
                lambda gdf: pd.Series(_residualize(gdf["fwd"].to_numpy(), gdf["om"].to_numpy()), index=gdf.index),
                include_groups=False,
            )
            ic_res = per_date_ic(df["ll"], resid.reindex(df.index), df["date"]).dropna()
            seed_ic_raw.append(ic_raw); seed_ic_res.append(ic_res); seed_ic_own.append(ic_own)

        if own_ic_once is None:  # edge-independent control, computed once
            own_ic_once = seed_ic_own[0]
        for name, series_list in (("leadlag", seed_ic_raw), ("leadlag_resid", seed_ic_res)):
            means = [s.mean() for s in series_list]
            ic = series_list[0]
            s = ic_summary(ic)
            mde = min_detectable_effect(n_dates=s["n"], ic_sd=max(ic.std(ddof=1), 1e-6),
                                        phi=_estimate_phi(ic), n_sims=400)
            rows.append({
                "edges": kind, "signal": name,
                "mean_ic": float(np.mean(means)), "seed_std": float(np.std(means)) if len(means) > 1 else 0.0,
                "hac_t": s["hac_t"], "p": two_sided_p(s["hac_t"]), "mde_80": mde, "n_dates": s["n"],
            })

    # single own-momentum control row (does not depend on the graph)
    s = ic_summary(own_ic_once)
    rows.append({
        "edges": "(none)", "signal": "own_mom", "mean_ic": float(own_ic_once.mean()), "seed_std": 0.0,
        "hac_t": s["hac_t"], "p": two_sided_p(s["hac_t"]),
        "mde_80": min_detectable_effect(n_dates=s["n"], ic_sd=max(own_ic_once.std(ddof=1), 1e-6),
                                        phi=_estimate_phi(own_ic_once), n_sims=400),
        "n_dates": s["n"],
    })
    table = pd.DataFrame(rows)
    reject, q = benjamini_hochberg(table["p"].fillna(1.0).to_numpy())
    table["fdr_sig"], table["q"] = reject, q
    return table


def make_synthetic_leadlag(
    *, n_days=1600, n_assets=60, n_blocks=6, beta_ll=0.35, h=5, noise=0.012, seed=0
):
    """Synthetic market with a PLANTED lead-lag effect: each name's daily return
    depends on its block-neighbours' return h days earlier. The block graph is the
    'true' economic-link graph the pipeline should recover (and the rewire null
    should not). Returns (prices, volume, sectors, market, true_graph)."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2015-01-01", periods=n_days)
    assets = [f"L{i:03d}" for i in range(n_assets)]
    block = rng.integers(0, n_blocks, size=n_assets)
    sectors = pd.Series([f"blk{b}" for b in block], index=assets, name="sector")

    # block-neighbour averaging matrix (row-normalized, no self)
    A = np.zeros((n_assets, n_assets))
    for b in range(n_blocks):
        idx = np.where(block == b)[0]
        for i in idx:
            others = idx[idx != i]
            if len(others):
                A[i, others] = 1.0 / len(others)

    mkt = rng.normal(0.0003, 0.008, size=n_days)
    idio = rng.normal(0, noise, size=(n_days, n_assets))
    r = np.zeros((n_days, n_assets))
    for t in range(n_days):
        r[t] = mkt[t] + idio[t]
        if t >= h:
            r[t] += beta_ll * (A @ r[t - h])  # neighbours' return h days ago -> me now

    prices = pd.DataFrame(100 * np.exp(np.cumsum(r, axis=0)), index=dates, columns=assets)
    volume = pd.DataFrame(rng.lognormal(13, 0.4, size=r.shape), index=dates, columns=assets)
    market = pd.Series(mkt, index=dates, name="MKT")

    # the true graph: block co-membership (what a perfect economic-link graph would be)
    true_graph = G.sector_graph(sectors, assets)
    return prices, volume, sectors, market, true_graph


def main():
    import argparse

    ap = argparse.ArgumentParser(description="lead-lag / spillover experiment")
    ap.add_argument("--synthetic-planted", action="store_true", help="planted-signal recovery test")
    args = ap.parse_args()

    if args.synthetic_planted:
        prices, volume, sectors, market, _ = make_synthetic_leadlag()
        print("=== PLANTED lead-lag recovery (synthetic) ===")
        table = run_leadlag(prices, sectors, market, edge_kinds=("sector", "rewire"),
                            label_horizon=5, lookback=5, warmup=260)
    else:
        from .data.download import load_market
        from .data.universe import default_universe

        prices, volume, sectors, market = load_market(
            synthetic=False, tickers=default_universe(), start="2014-01-01", end="2024-12-31")
        print("=== lead-lag on real data ===")
        table = run_leadlag(prices, sectors, market, edge_kinds=("sector", "correlation", "rewire"))

    cols = ["edges", "signal", "mean_ic", "seed_std", "hac_t", "p", "mde_80", "n_dates", "fdr_sig"]
    print(table[cols].to_string(index=False, float_format=lambda v: f"{v:+.3f}"))


if __name__ == "__main__":
    main()
