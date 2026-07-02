"""Point-in-time node features, cross-sectional normalization, and labels.

Two invariants, both asserted in tests:

1. Features at date t use only data up to and including t (``.loc[:asof]``).
   Labels are the *only* thing allowed to look forward.
2. Normalization is cross-sectional: ranks / z-scores are taken across the
   universe on a single date, never pooled across time.

Post-review fixes baked in here: 12-1 momentum is the true t-21..t-252 return;
beta is measured against an exogenous market series (SPY), never a survivor
equal-weight mean; ``forward_return`` is duplicate/absent-date safe; a
neighbor-aggregated trailing-return feature carries the (contemporaneous)
relative-strength channel that both the GNN and the MLP can consume.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

BASE_FEATURES = ["mom_1m", "mom_3m", "mom_12_1", "rev_1w", "vol_20d", "turnover", "beta", "size"]
FEATURES = BASE_FEATURES + ["nbr_ret"]


def compute_features(
    prices: pd.DataFrame,
    volume: pd.DataFrame,
    asof,
    nodes,
    *,
    market: pd.Series | None = None,
) -> pd.DataFrame:
    """Raw (un-normalized) node features for a single cross-section at ``asof``.

    ``market`` is an exogenous benchmark *return* series (e.g. SPY) used for beta.
    If omitted, beta falls back to the equal-weight universe mean -- acceptable
    only for synthetic tests, NOT for real runs (survivorship). See PLAN.md.
    """
    px = prices.loc[:asof].reindex(columns=nodes)
    vol = volume.loc[:asof].reindex(columns=nodes)
    rets = px.pct_change()
    dollar = px * vol

    def perf(n: int) -> pd.Series:
        if len(px) <= n:
            return pd.Series(np.nan, index=nodes)
        return px.iloc[-1] / px.iloc[-1 - n] - 1

    out = pd.DataFrame(index=pd.Index(nodes, name="asset"))
    out["mom_1m"] = perf(21)
    out["mom_3m"] = perf(63)
    # True 12-1 momentum: return from t-252 to t-21 (skips the recent month).
    out["mom_12_1"] = (px.iloc[-22] / px.iloc[-253] - 1) if len(px) > 253 else np.nan
    out["rev_1w"] = perf(5)
    out["vol_20d"] = rets.iloc[-20:].std() * np.sqrt(252)
    out["turnover"] = np.log(dollar.iloc[-20:].mean().replace(0, np.nan))

    win = rets.iloc[-120:]
    if market is not None:
        mkt = market.loc[:asof].iloc[-120:]
        mkt = mkt.reindex(win.index)
    else:
        mkt = win.mean(axis=1)  # synthetic-only fallback (see docstring)
    mkt_var = mkt.var()
    out["beta"] = win.apply(lambda c: c.cov(mkt)) / mkt_var if mkt_var else np.nan
    out["size"] = np.log(dollar.iloc[-1].replace(0, np.nan))
    return out.reindex(nodes)


def neighbor_return_feature(returns: pd.DataFrame, asof, graph, *, lookback: int) -> pd.Series:
    """Mean trailing return of each node's out-neighbours (PIT).

    This is the (contemporaneous) relative-strength / lead-lag channel: it lets a
    *static* GNN-or-MLP see how a name is doing relative to its graph neighbours,
    without any temporal recurrence. PIT because it reads only ``.loc[:asof]`` and
    a graph that was itself built as-of ``asof``.
    """
    nodes = list(graph.nodes)
    hist = returns.loc[:asof].iloc[-lookback:].reindex(columns=nodes)
    node_ret = ((1 + hist).prod() - 1).to_numpy()  # trailing cumulative return per node
    src, dst = graph.edge_index
    out = np.full(len(nodes), np.nan)
    if src.size:
        sums = np.zeros(len(nodes))
        counts = np.zeros(len(nodes))
        np.add.at(sums, src, node_ret[dst])
        np.add.at(counts, src, 1.0)
        nz = counts > 0
        out[nz] = sums[nz] / counts[nz]
    return pd.Series(out, index=pd.Index(nodes, name="asset"))


def cross_sectional_normalize(feat: pd.DataFrame, method: str = "zscore") -> pd.DataFrame:
    """Normalize each feature across assets *within this one date*.

    Missing values are filled with the cross-sectional median (PIT-safe: uses only
    this date). A column that is entirely missing normalizes to 0 (not silent NaN
    that would flow to models undetected). Both z-score and rank are shift- and
    scale-invariant, which the scope test exploits to prove no cross-date pooling.
    """
    x = feat.astype(float).copy()
    x = x.fillna(x.median(axis=0)).fillna(0.0)
    if method == "rank":
        r = x.rank(axis=0)
        mu, sd = r.mean(axis=0), r.std(axis=0, ddof=0).replace(0, 1.0)
        return (r - mu) / sd
    mu, sd = x.mean(axis=0), x.std(axis=0, ddof=0).replace(0, 1.0)
    return (x - mu) / sd


def forward_return(prices: pd.DataFrame, asof, nodes, horizon: int) -> pd.Series:
    """Label: simple return from ``asof`` to ``asof + horizon`` steps ahead.

    The one function permitted to read the future. Duplicate/absent-date safe.
    """
    if not prices.index.is_unique:
        raise ValueError("price index must be unique for unambiguous labelling")
    i = prices.index.get_indexer([asof])[0]
    if i == -1 or i + horizon >= len(prices.index):
        return pd.Series(np.nan, index=nodes)
    fwd = prices.iloc[i + horizon].reindex(nodes) / prices.iloc[i].reindex(nodes) - 1
    return fwd


def forward_volatility(prices: pd.DataFrame, asof, nodes, horizon: int) -> pd.Series:
    """Label: log annualized realized volatility over the next ``horizon`` steps.

    The predictable *anchor* target (H3). Reads the future, like ``forward_return``.
    """
    if not prices.index.is_unique:
        raise ValueError("price index must be unique for unambiguous labelling")
    i = prices.index.get_indexer([asof])[0]
    if i == -1 or i + horizon >= len(prices.index):
        return pd.Series(np.nan, index=nodes)
    fwd = prices.iloc[i : i + horizon + 1].reindex(columns=nodes)
    daily = fwd.pct_change().iloc[1:]
    rv = daily.std(ddof=0) * np.sqrt(252)
    return np.log(rv.replace(0, np.nan))


def leakage_canary(y: pd.Series) -> pd.Series:
    """A feature identical to the label. Injected only in tests to prove the
    evaluation *would* light up if the future leaked in."""
    return y.copy()
