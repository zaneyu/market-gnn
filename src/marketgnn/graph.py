"""Point-in-time graph construction.

Every builder is a pure function of data at or before an as-of date; it must
never read a row dated after ``asof``. The correlation builder slices
``returns.loc[:asof]`` internally, so appending or mutating future rows in the
input frame cannot change its output -- the property the PIT test asserts.

Graphs are returned as plain numpy arrays (no torch dependency) so the hygiene
tests and CI run without the GNN stack installed. The GNN model converts them to
tensors at the edge.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Graph:
    edge_index: np.ndarray  # int64 [2, E], directed src -> dst
    edge_weight: np.ndarray  # float32 [E]
    nodes: np.ndarray  # asset id at each node position

    @property
    def num_edges(self) -> int:
        return int(self.edge_index.shape[1])

    @property
    def avg_degree(self) -> float:
        n = len(self.nodes)
        return self.num_edges / n if n else 0.0


def _empty(nodes) -> Graph:
    return Graph(np.zeros((2, 0), np.int64), np.zeros(0, np.float32), np.asarray(nodes))


def _corr_matrix(hist: pd.DataFrame, nodes, min_overlap: int, shrinkage) -> np.ndarray:
    """Correlation matrix aligned to ``nodes``, optionally denoised.

    A 60-day correlation over hundreds of names is badly under-determined, so most
    of its week-to-week churn is estimation noise. ``shrinkage`` counters that:
    ``"lw"`` = Ledoit-Wolf (built for p >= n), or a float in (0,1] = linear blend of
    the raw correlation toward the identity (shrink spurious off-diagonals to 0).
    """
    raw = np.array(hist.corr(min_periods=min_overlap).reindex(index=nodes, columns=nodes), dtype=float)
    if not shrinkage:
        return raw
    if shrinkage == "lw":
        X = hist.reindex(columns=nodes).to_numpy()
        Xc = X[~np.isnan(X).any(axis=1)]  # Ledoit-Wolf needs complete cases
        if len(Xc) >= max(5, X.shape[1] // 2):
            from sklearn.covariance import ledoit_wolf

            cov, _ = ledoit_wolf(Xc)
            d = np.sqrt(np.clip(np.diag(cov), 1e-12, None))
            return cov / np.outer(d, d)
        shrinkage = 0.2  # too sparse for LW -> light identity shrink of the raw estimate
    delta = float(shrinkage)
    out = (1 - delta) * np.nan_to_num(raw)
    np.fill_diagonal(out, 1.0)
    return out


def correlation_knn(
    returns: pd.DataFrame, asof, nodes, *, window: int, k: int, min_overlap: int | None = None, shrinkage=None
) -> Graph:
    """kNN graph on trailing return correlation. Each node points at its k
    strongest-|correlation| neighbours; edge weight is the signed correlation.
    ``shrinkage`` (see ``_corr_matrix``) denoises the estimate to reduce spurious churn."""
    hist = returns.loc[:asof].iloc[-window:].reindex(columns=nodes)
    if len(hist) < 2:
        return _empty(nodes)
    min_overlap = min_overlap or max(5, window // 4)
    corr = _corr_matrix(hist, nodes, min_overlap, shrinkage)
    np.fill_diagonal(corr, np.nan)
    absc = np.abs(corr)

    n = len(nodes)
    kk = min(k, n - 1)
    src, dst, w = [], [], []
    for i in range(n):
        row = absc[i]
        valid = np.flatnonzero(~np.isnan(row))
        if valid.size == 0:
            continue
        # Deterministic across numpy versions/platforms: sort by descending
        # |corr|, breaking ties by ascending node index. lexsort's last key is
        # primary, so (-|corr|) ascending == |corr| descending.
        order = np.lexsort((valid, -row[valid]))
        top = valid[order[:kk]]
        for j in top:
            src.append(i)
            dst.append(int(j))
            w.append(float(corr[i, j]))
    if not src:
        return _empty(nodes)
    return Graph(np.array([src, dst], np.int64), np.array(w, np.float32), np.asarray(nodes))


def sector_graph(sectors: pd.Series, nodes, *, max_degree: int | None = None, seed: int = 0) -> Graph:
    """Fully connect same-sector names (optionally degree-capped for tractability)."""
    s = sectors.reindex(nodes)
    rng = np.random.default_rng(seed)
    groups: dict = {}
    for i, node in enumerate(nodes):
        key = s.iloc[i]
        if pd.isna(key):
            continue
        groups.setdefault(key, []).append(i)

    src, dst = [], []
    for members in groups.values():
        for i in members:
            others = [j for j in members if j != i]
            if max_degree and len(others) > max_degree:
                others = rng.choice(others, max_degree, replace=False).tolist()
            for j in others:
                src.append(i)
                dst.append(int(j))
    if not src:
        return _empty(nodes)
    ei = np.array([src, dst], np.int64)
    return Graph(ei, np.ones(ei.shape[1], np.float32), np.asarray(nodes))


def random_graph(nodes, *, degree: int, seed: int = 0) -> Graph:
    """Degree-matched random control: the null the real graph must beat."""
    n = len(nodes)
    d = min(degree, n - 1)
    if d < 1:
        return _empty(nodes)
    rng = np.random.default_rng(seed)
    src, dst = [], []
    for i in range(n):
        choices = rng.choice(np.delete(np.arange(n), i), d, replace=False)
        for j in choices:
            src.append(i)
            dst.append(int(j))
    ei = np.array([src, dst], np.int64)
    return Graph(ei, np.ones(ei.shape[1], np.float32), np.asarray(nodes))


def match_random(graph: Graph, *, seed: int = 0) -> Graph:
    """Weak random control matched only to *average* degree. Kept for comparison;
    the real null is ``degree_preserving_rewire`` (matches the full degree sequence)."""
    return random_graph(graph.nodes, degree=round(graph.avg_degree), seed=seed)


def degree_preserving_rewire(graph: Graph, *, n_swaps: int | None = None, seed: int = 0) -> Graph:
    """Degree-sequence-preserving random null (directed Maslov-Sneppen edge swaps).

    Repeatedly pick edges (a->b), (c->d) and swap to (a->d), (c->b). This preserves
    every node's in- AND out-degree exactly, so the GNN sees the same aggregation
    fan-in/out as the real graph -- the only thing randomized is *which* neighbours.
    That makes it the fair null for H2 ("does topology carry signal, beyond degree?").
    Edge weights ride along with their (new) destination.
    """
    ei = graph.edge_index.copy()
    w = graph.edge_weight.copy()
    m = ei.shape[1]
    if m < 2:
        return Graph(ei, w, np.asarray(graph.nodes))
    rng = np.random.default_rng(seed)
    n_swaps = n_swaps if n_swaps is not None else 10 * m
    existing = {(int(s), int(d)) for s, d in ei.T}
    for _ in range(n_swaps):
        e1, e2 = rng.integers(0, m, size=2)
        a, b = ei[0, e1], ei[1, e1]
        c, d = ei[0, e2], ei[1, e2]
        if a == d or c == b:  # would create a self-loop
            continue
        if (int(a), int(d)) in existing or (int(c), int(b)) in existing:  # would multi-edge
            continue
        existing.discard((int(a), int(b)))
        existing.discard((int(c), int(d)))
        existing.add((int(a), int(d)))
        existing.add((int(c), int(b)))
        ei[1, e1], ei[1, e2] = d, b
        w[e1], w[e2] = w[e2], w[e1]
    return Graph(ei, w, np.asarray(graph.nodes))


def symmetrize(graph: Graph) -> Graph:
    """Undirected union: for every edge i->j add j->i (correlation is symmetric).

    Applied in the model pipeline, not at construction time, so the directed top-k
    graph stays available as an explicit ablation.
    """
    ei, w = graph.edge_index, graph.edge_weight
    both = {}
    for k in range(ei.shape[1]):
        i, j, wt = int(ei[0, k]), int(ei[1, k]), float(w[k])
        both[(i, j)] = wt
        both.setdefault((j, i), wt)
    if not both:
        return graph
    keys = sorted(both)
    src = np.array([i for i, _ in keys], np.int64)
    dst = np.array([j for _, j in keys], np.int64)
    wt = np.array([both[k] for k in keys], np.float32)
    return Graph(np.vstack([src, dst]), wt, np.asarray(graph.nodes))


def add_self_loops(graph: Graph, *, weight: float = 1.0) -> Graph:
    """Add i->i so a node keeps its own features through message passing (else
    2-hop aggregation over dense sector cliques washes out the node's own signal)."""
    n = len(graph.nodes)
    loops_src = np.arange(n, dtype=np.int64)
    ei = np.hstack([graph.edge_index, np.vstack([loops_src, loops_src])])
    w = np.concatenate([graph.edge_weight, np.full(n, weight, np.float32)])
    return Graph(ei, w, np.asarray(graph.nodes))


def edges_from_pairs(links: pd.DataFrame, nodes, *, symmetric: bool = False) -> Graph:
    """Build a graph from an explicit link table -- the drop-in for a REAL
    economic-link graph (supplier-customer from 10-K disclosures, shared-analyst
    coverage, common 13F ownership). ``links`` has columns [src, dst, weight?];
    tickers absent from ``nodes`` are dropped. This is the interface that lets the
    lead-lag experiment run on true economic links instead of the correlation/sector
    proxy -- provide the CSV, the harness does the rest.
    """
    pos = {t: i for i, t in enumerate(nodes)}
    w_col = "weight" in links.columns
    # accumulate into a dict keyed by directed pair so a mutual kNN pair (both i->j and
    # j->i present in ``links``) plus symmetrization does not emit the same edge twice
    # and double its weight in aggregation. Keep the max weight seen for a pair.
    edges: dict[tuple[int, int], float] = {}
    for r in links.itertuples(index=False):
        a, b = pos.get(r.src), pos.get(r.dst)
        if a is None or b is None or a == b:
            continue
        wt = float(getattr(r, "weight")) if w_col else 1.0
        edges[(a, b)] = max(edges.get((a, b), wt), wt)
        if symmetric:
            edges[(b, a)] = max(edges.get((b, a), wt), wt)
    if not edges:
        return _empty(nodes)
    keys = sorted(edges)
    src = np.array([a for a, _ in keys], np.int64)
    dst = np.array([b for _, b in keys], np.int64)
    w = np.array([edges[k] for k in keys], np.float32)
    return Graph(np.vstack([src, dst]), w, np.asarray(nodes))


def in_out_degrees(graph: Graph) -> tuple[np.ndarray, np.ndarray]:
    """(out_degree, in_degree) per node position -- used to verify the null matches."""
    n = len(graph.nodes)
    out = np.bincount(graph.edge_index[0], minlength=n)
    inn = np.bincount(graph.edge_index[1], minlength=n)
    return out, inn
