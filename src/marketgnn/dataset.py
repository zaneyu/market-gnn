"""Assemble a leak-free panel: per rebalance date, build the as-of graph, the
point-in-time node features (incl. the neighbour-return channel), and the two
forward labels. Everything routes through the PIT-guarded primitives in
``features``/``graph``; nothing here reads the future except the label calls.

Also provides a synthetic factor market so the pipeline (and a demo run) works
offline, with real cross-sectional structure the graph can actually recover.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import graph as G
from .features import (
    BASE_FEATURES,
    FEATURES,
    compute_features,
    cross_sectional_normalize,
    forward_return,
    forward_volatility,
    neighbor_return_feature,
)

GRAPH_KINDS = ["correlation", "shrinkage", "frozen", "sector", "both", "random", "rewire", "none"]


@dataclass
class Dataset:
    X: pd.DataFrame  # MultiIndex (date, asset) -> normalized FEATURES
    y_ret: pd.Series
    y_vol: pd.Series
    graphs: dict  # date -> marketgnn.graph.Graph (the real edges; None for "none")
    dates: np.ndarray
    graph_kind: str

    @property
    def feature_cols(self) -> list[str]:
        return list(self.X.columns)


def rebalance_dates(index: pd.DatetimeIndex, freq: str = "W") -> pd.DatetimeIndex:
    """Last available trading day in each period (weekly by default)."""
    per = index.to_period(freq)
    last = pd.Series(index, index=per).groupby(level=0).last()
    return pd.DatetimeIndex(sorted(last.values))


def make_graph(kind, returns, sectors, asof, nodes, *, corr_window, k, seed=0):
    """Dispatch the graph builders for an ablation kind. Returns a Graph or None."""
    if kind == "none":
        return None
    if kind == "correlation":
        return G.correlation_knn(returns, asof, nodes, window=corr_window, k=k)
    if kind == "shrinkage":  # denoised correlation (Ledoit-Wolf) -> less spurious churn
        return G.correlation_knn(returns, asof, nodes, window=corr_window, k=k, shrinkage="lw")
    if kind == "sector":
        return G.sector_graph(sectors, nodes, max_degree=k)
    if kind == "both":
        c = G.correlation_knn(returns, asof, nodes, window=corr_window, k=k)
        s = G.sector_graph(sectors, nodes, max_degree=k)
        # Dedup parallel edges: a neighbour reached by BOTH edge types must count
        # once, else neighbor_return_feature double-counts it (biasing nbr_ret).
        merged: dict = {}
        for gi in (c, s):
            for idx in range(gi.num_edges):
                merged.setdefault((int(gi.edge_index[0, idx]), int(gi.edge_index[1, idx])), float(gi.edge_weight[idx]))
        if not merged:
            return G.Graph(np.zeros((2, 0), np.int64), np.zeros(0, np.float32), np.asarray(nodes))
        keys = list(merged)
        ei = np.array([[a for a, _ in keys], [b for _, b in keys]], np.int64)
        w = np.array([merged[key] for key in keys], np.float32)
        return G.Graph(ei, w, np.asarray(nodes))
    if kind == "random":  # weak null (avg-degree matched)
        real = G.correlation_knn(returns, asof, nodes, window=corr_window, k=k)
        return G.match_random(real, seed=seed)
    if kind == "rewire":  # fair null (degree-sequence preserving)
        real = G.correlation_knn(returns, asof, nodes, window=corr_window, k=k)
        return G.degree_preserving_rewire(real, seed=seed)
    raise ValueError(f"unknown graph kind: {kind}")


def build_dataset(
    prices: pd.DataFrame,
    volume: pd.DataFrame,
    sectors: pd.Series,
    market: pd.Series | None,
    rebal: pd.DatetimeIndex,
    *,
    graph_kind: str = "correlation",
    label_horizon: int = 5,
    corr_window: int = 60,
    k: int = 10,
    nbr_lookback: int = 21,
    warmup: int = 260,
    membership: "pd.DataFrame | None" = None,
) -> Dataset:
    """Build one Dataset for a given graph ablation kind.

    ``membership`` (optional): boolean DataFrame [date x asset], True where a name
    is a point-in-time index member. When given, each cross-section uses only the
    members as-of that date -- the real defence against survivorship bias.
    """
    rets = prices.pct_change()
    rows, y_ret_rows, y_vol_rows, graphs = [], [], [], {}
    frozen_graph = None  # for graph_kind == "frozen": estimated once, then reused

    for asof in rebal:
        if prices.index.get_indexer([asof])[0] < warmup:
            continue
        if membership is not None:
            m = membership.reindex(index=[asof]).iloc[0]
            nodes = list(m.index[m.fillna(False).to_numpy()])
        else:
            nodes = list(prices.columns)
        if len(nodes) < k + 2:
            continue

        if graph_kind == "frozen":
            # Static topology: estimate ONCE (first eligible date, long window) and
            # freeze. Still point-in-time -- the neighbour feature keeps updating
            # daily over fixed edges. Assumes a constant universe.
            if membership is not None:
                raise NotImplementedError("frozen graph assumes a constant universe (membership=None)")
            if frozen_graph is None:
                long_w = min(len(prices), max(4 * corr_window, 504))
                frozen_graph = G.correlation_knn(rets, asof, nodes, window=long_w, k=k, shrinkage="lw")
            graph = frozen_graph
        else:
            graph = make_graph(graph_kind, rets, sectors, asof, nodes, corr_window=corr_window, k=k)
        feat = compute_features(prices, volume, asof, nodes, market=market)
        if graph is not None:
            feat["nbr_ret"] = neighbor_return_feature(rets, asof, graph, lookback=nbr_lookback)
        else:
            feat["nbr_ret"] = 0.0  # blindfolded baseline: no graph, no neighbour signal
        feat = feat.reindex(columns=FEATURES)
        norm = cross_sectional_normalize(feat)

        y_r = forward_return(prices, asof, nodes, label_horizon)
        y_v = forward_volatility(prices, asof, nodes, label_horizon)

        idx = pd.MultiIndex.from_product([[asof], nodes], names=["date", "asset"])
        norm.index = idx
        rows.append(norm)
        y_ret_rows.append(pd.Series(y_r.to_numpy(), index=idx))
        y_vol_rows.append(pd.Series(y_v.to_numpy(), index=idx))
        graphs[asof] = graph

    X = pd.concat(rows)
    y_ret = pd.concat(y_ret_rows)
    y_vol = pd.concat(y_vol_rows)
    dates = np.array(sorted(graphs.keys()))
    return Dataset(X, y_ret, y_vol, graphs, dates, graph_kind)


def make_synthetic(
    n_days: int = 1400, n_assets: int = 60, n_sectors: int = 6, seed: int = 0
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Synthetic factor market: return = market beta + sector factor + idiosyncratic.

    Real cross-sectional structure means the correlation graph recovers sectors and
    the neighbour-return feature carries mild, HONEST signal -- so a demo run isn't a
    trivial null, but nothing here is tuned to make the GNN win.
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2015-01-01", periods=n_days)
    assets = [f"S{i:03d}" for i in range(n_assets)]
    sector_id = rng.integers(0, n_sectors, size=n_assets)
    sectors = pd.Series([f"sec{s}" for s in sector_id], index=assets, name="sector")

    mkt = rng.normal(0.0003, 0.01, size=n_days)
    sector_f = rng.normal(0, 0.008, size=(n_days, n_sectors))
    beta = rng.uniform(0.5, 1.5, size=n_assets)
    idio = rng.normal(0, 0.012, size=(n_days, n_assets))
    rets = beta[None, :] * mkt[:, None] + sector_f[:, sector_id] + idio

    prices = pd.DataFrame(100 * np.exp(np.cumsum(rets, axis=0)), index=dates, columns=assets)
    volume = pd.DataFrame(rng.lognormal(13, 0.4, size=rets.shape), index=dates, columns=assets)
    market = pd.Series(mkt, index=dates, name="SPY")  # exogenous benchmark for beta
    return prices, volume, sectors, market
