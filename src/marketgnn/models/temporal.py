"""Temporal (spatiotemporal) GNN — the LEARNED counterpart to the zero-parameter
lead-lag signal.

Same-date message passing (the static GNN in ``gnn.py``) cannot express lead-lag:
neighbours' *past* returns predicting a name's *future* return is inherently temporal.
This model is the honest v2 the plan promised: a **graph-conv spatial layer per date
whose embeddings feed a GRU over the sequence of dates** (Seo et al., "Structured
Sequence Modeling with GConvGRU", 2018, in its GConv-then-GRU form). At date t the
per-node hidden state summarises everything up to t — including neighbours' history —
so if linkage carries a lead-lag signal, a *learned* model has the capacity to find it.

The A/B is identical to the static one: ``use_graph=False`` swaps the real edges for
self-loops, leaving the architecture, parameter count, GRU, head, loss and training
loop untouched — the only difference is whether graph topology is visible. So this
answers "does the graph help a model that CAN see time?" cleanly. As always, the
synthetic planted-lead-lag market is recovered first (power) before any null on real
data is trusted. Torch-only; aggregation hand-coded (no torch-geometric).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch import nn

from .. import graph as G
from .gnn import _mean_aggregate
from .losses import soft_spearman_loss


class _GConv(nn.Module):
    """One GraphSAGE-mean spatial layer: h' = relu(W_self x + W_neigh mean_N(x))."""

    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.lin_self = nn.Linear(dim_in, dim_out)
        self.lin_neigh = nn.Linear(dim_in, dim_out)

    def forward(self, x, edge_index, n):
        return torch.relu(self.lin_self(x) + self.lin_neigh(_mean_aggregate(x, edge_index, n)))


class _GConvGRU(nn.Module):
    """Spatial graph-conv per date -> GRU cell over dates. Hidden state is per node."""

    def __init__(self, n_features, hidden=32):
        super().__init__()
        self.hidden = hidden
        self.spatial = _GConv(n_features, hidden)
        self.cell = nn.GRUCell(hidden, hidden)
        self.head = nn.Linear(hidden, 1)

    def unroll(self, x_seq, edge_seq, n):
        """Carry the per-node GRU state across dates; return a score tensor [n] per date."""
        h = torch.zeros(n, self.hidden, device=x_seq[0].device)
        preds = []
        for x, ei in zip(x_seq, edge_seq):
            h = self.cell(self.spatial(x, ei, n), h)
            preds.append(self.head(h).squeeze(-1))
        return preds


def _seq_loss(preds, ys, ms):
    terms = [soft_spearman_loss(p[m], y[m]) for p, y, m in zip(preds, ys, ms) if int(m.sum()) >= 3]
    if not terms:
        return torch.zeros((), requires_grad=True)
    return torch.stack(terms).mean()


class TemporalGNN:
    """Purged single-split trainer: unroll the GConvGRU over the ordered date
    sequence, rank-loss at each date over the present names, early stop on a purged
    val tail (val hidden state reflects the full train history). ``predict`` continues
    the unroll from the trained state onto the test dates."""

    def __init__(self, *, use_graph=True, hidden=32, lr=1e-3, epochs=300, patience=25,
                 val_frac=0.2, val_gap=1, seed=0):
        self.use_graph = use_graph
        self.hidden, self.lr, self.epochs, self.patience = hidden, lr, epochs, patience
        self.val_frac, self.val_gap, self.seed = val_frac, val_gap, seed
        self.net = None
        self._nodes = None

    def _edges(self, graphs, date, n, device):
        g = graphs.get(date) if graphs else None
        if self.use_graph and g is not None and g.num_edges:
            g = self._align(g)
            ei = G.add_self_loops(G.symmetrize(g)).edge_index
        else:
            loops = np.arange(n)
            ei = np.vstack([loops, loops])
        return torch.as_tensor(ei, dtype=torch.long, device=device)

    def _align(self, g):
        """Remap a provider graph's edges onto ``self._nodes`` positions. The feature
        rows are ordered by ``self._nodes`` (= prices.columns); an externally-built
        graph (e.g. the 13F provider, ordered by its own node list) must be reindexed
        or every edge would silently connect the wrong pair. Edges touching a name not
        in the fixed universe are dropped. No-op fast path when the order already
        matches."""
        gnodes = list(g.nodes)
        if gnodes == self._nodes:
            return g
        pos = {t: i for i, t in enumerate(self._nodes)}
        remap = np.array([pos.get(t, -1) for t in gnodes], dtype=np.int64)
        src, dst = g.edge_index
        ns, nd = remap[src], remap[dst]
        keep = (ns >= 0) & (nd >= 0)
        return G.Graph(np.vstack([ns[keep], nd[keep]]),
                       g.edge_weight[keep], np.asarray(self._nodes))

    def _seq(self, X, y, graphs, dates, device):
        """Aligned [n]-node tensors per date over the FIXED universe, with a mask of
        present-and-labelled names (missing names carry zeros; they still hold GRU
        state but never enter the loss)."""
        pos = {t: i for i, t in enumerate(self._nodes)}
        n = len(self._nodes)
        xs, ys, ms, eis = [], [], [], []
        for d in dates:
            xd = X.loc[d]
            present = [a for a in xd.index if a in pos]
            idx = [pos[a] for a in present]
            xv = np.zeros((n, X.shape[1]), np.float32)
            xv[idx] = np.nan_to_num(xd.loc[present].to_numpy(), nan=0.0)
            yv = np.full(n, np.nan, np.float32)
            yv[idx] = y.loc[d].reindex(present).to_numpy()
            m = np.isfinite(yv)
            xs.append(torch.as_tensor(xv, device=device))
            ys.append(torch.as_tensor(np.nan_to_num(yv, nan=0.0), device=device))
            ms.append(torch.as_tensor(m, device=device))
            eis.append(self._edges(graphs, d, n, device))
        return xs, ys, ms, eis

    def fit(self, X, y, graphs, train_dates, nodes):
        torch.manual_seed(self.seed)
        device = "cpu"
        self._nodes = list(nodes)
        n = len(self._nodes)
        train_dates = list(train_dates)
        cut = max(2, int(len(train_dates) * (1 - self.val_frac)))
        tr_dates, va_dates = train_dates[:cut], train_dates[cut + self.val_gap:]

        self.net = _GConvGRU(X.shape[1], self.hidden).to(device)
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        xs, ys, ms, eis = self._seq(X, y, graphs, tr_dates, device)
        # for val, unroll train+val contiguously so the val state reflects full history
        xva, yva, mva, eiva = self._seq(X, y, graphs, va_dates, device) if va_dates else ([], [], [], [])
        n_tr = len(xs)

        best, best_state, bad = np.inf, None, 0
        for _ in range(self.epochs):
            self.net.train()
            opt.zero_grad()
            loss = _seq_loss(self.net.unroll(xs, eis, n), ys, ms)
            loss.backward()
            opt.step()
            if va_dates:
                self.net.eval()
                with torch.no_grad():
                    preds = self.net.unroll(xs + xva, eis + eiva, n)
                    val = float(_seq_loss(preds[n_tr:], yva, mva).item())
            else:
                val = float(loss.item())
            if val < best - 1e-5:
                best, best_state, bad = val, {k: v.clone() for k, v in self.net.state_dict().items()}, 0
            else:
                bad += 1
                if bad >= self.patience:
                    break
        if best_state is not None:
            self.net.load_state_dict(best_state)
        return self

    @torch.no_grad()
    def predict(self, X, y, graphs, train_dates, test_dates):
        self.net.eval()
        device = "cpu"
        n = len(self._nodes)
        alld = list(train_dates) + list(test_dates)
        xs, _ys, _ms, eis = self._seq(X, y, graphs, alld, device)
        preds = self.net.unroll(xs, eis, n)
        pos = {t: i for i, t in enumerate(self._nodes)}
        test_set = set(pd.Timestamp(t) for t in test_dates)
        rows = []
        for k, d in enumerate(alld):
            if pd.Timestamp(d) not in test_set:
                continue
            p = preds[k].cpu().numpy()
            present = [a for a in X.loc[d].index if a in pos]
            rows.append(pd.DataFrame({"date": d, "asset": present,
                                      "score": [float(p[pos[a]]) for a in present]}))
        return pd.concat(rows, ignore_index=True)


# --------------------------------------------------------------------------- runner
def _feature_frame(prices, dates, nodes, lookbacks=(5, 20, 60)):
    """Per-date cross-sectionally-standardized trailing returns — the raw material a
    spatial-then-temporal model needs to (potentially) learn lead-lag: the graph conv
    mixes neighbours' trailing returns, the GRU carries them forward."""
    rets = prices.pct_change()
    xs, ys = [], []
    for d in dates:
        hist = rets.loc[:d]
        feat = {}
        for lb in lookbacks:
            r = (1 + hist.iloc[-lb:]).prod() - 1
            feat[f"r{lb}"] = r.reindex(nodes)
        F = pd.DataFrame(feat)
        F = (F - F.mean()) / (F.std(ddof=0) + 1e-9)
        F = F.fillna(0.0)
        F.index = pd.MultiIndex.from_product([[d], nodes], names=["date", "asset"])
        xs.append(F)
    return pd.concat(xs)


def run_temporal(prices, graphs, *, label_horizon=5, warmup=260, rebal_freq="W",
                 test_frac=0.35, hidden=32, epochs=300, seed=0):
    """Purged split A/B for the GConvGRU: fit graph vs no-graph (self-loops), report
    out-of-sample per-date rank-IC with a Newey-West/HAC t. ``graphs`` is a dict
    date->Graph or a callable(asof)->Graph (PIT)."""
    from ..dataset import rebalance_dates
    from ..evaluate import ic_summary, per_date_ic, two_sided_p
    from ..features import forward_return

    nodes = list(prices.columns)
    idx = prices.index
    rebal = [d for d in rebalance_dates(idx, rebal_freq)
             if idx.get_indexer([d])[0] >= warmup and idx.get_indexer([d])[0] + label_horizon < len(idx)]
    X = _feature_frame(prices, rebal, nodes)
    yparts, gmap = [], {}
    for d in rebal:
        fwd = forward_return(prices, d, nodes, label_horizon)
        yparts.append(pd.Series(fwd.to_numpy(),
                                index=pd.MultiIndex.from_product([[d], nodes], names=["date", "asset"])))
        gmap[d] = graphs(d) if callable(graphs) else graphs
    y = pd.concat(yparts)

    cut = int(len(rebal) * (1 - test_frac))
    # purge the train/test boundary by at least one label horizon, measured in
    # rebalance steps from the ACTUAL date spacing (not a hardcoded weekly cadence),
    # so a monthly or daily rebal still removes the right overlap.
    locs = idx.get_indexer(rebal)
    step_len = int(np.median(np.diff(locs))) if len(locs) > 1 else 5
    gap = max(1, int(np.ceil(label_horizon / max(1, step_len))))
    train_dates, test_dates = rebal[:cut], rebal[cut + gap:]

    rows = []
    for use_graph in (True, False):
        m = TemporalGNN(use_graph=use_graph, hidden=hidden, epochs=epochs, seed=seed)
        m.fit(X, y, gmap, train_dates, nodes)
        pred = m.predict(X, y, gmap, train_dates, test_dates)
        fwd = pd.Series(
            np.concatenate([forward_return(prices, d, list(X.loc[d].index), label_horizon).to_numpy()
                            for d in test_dates]),
            index=pd.MultiIndex.from_tuples([(d, a) for d in test_dates for a in X.loc[d].index]),
        ).to_numpy()
        ic = per_date_ic(pd.Series(pred["score"].to_numpy()), pd.Series(fwd), pd.Series(pred["date"].to_numpy())).dropna()
        s = ic_summary(ic)
        rows.append({"graph": "yes" if use_graph else "no (self-loops)",
                     "mean_ic": float(ic.mean()), "hac_t": s["hac_t"],
                     "p": two_sided_p(s["hac_t"]), "n_dates": s["n"]})
    return pd.DataFrame(rows)


def main():
    import argparse

    ap = argparse.ArgumentParser(description="temporal (GConvGRU) A/B experiment")
    ap.add_argument("--synthetic-planted", action="store_true",
                    help="planted lead-lag recovery (power) on the synthetic market")
    ap.add_argument("--graph", default="coholding", choices=["coholding", "correlation"],
                    help="graph the temporal model runs over (real data)")
    args = ap.parse_args()

    if args.synthetic_planted:
        from ..leadlag import make_synthetic_leadlag

        prices, _vol, sectors, _mkt, true_graph = make_synthetic_leadlag(n_assets=60)
        graphs = {}  # static true block graph applied at every date

        def prov(asof):
            return true_graph

        print("=== temporal GConvGRU: planted lead-lag recovery (power) ===")
        table = run_temporal(prices, prov, warmup=260)
    else:
        from ..data.download import load_market
        from ..data.universe import default_universe

        if args.graph == "coholding":
            from ..coholding import cusip_map, make_provider
            nodes = list(cusip_map())
            provider, _g = make_provider(nodes, k=10)
            graphs = provider["coholding"]
            tickers = nodes
        else:
            from .. import graph as GG
            from ..dataset import make_graph
            tickers = default_universe()
            graphs = None  # built per date below via correlation

        prices, _vol, sectors, _mkt = load_market(
            synthetic=False, tickers=tickers, start="2014-01-01", end="2024-12-31")
        if args.graph == "correlation":
            from ..dataset import make_graph
            rets = prices.pct_change()

            def graphs(asof):
                return make_graph("correlation", rets, sectors, asof, list(prices.columns), corr_window=60, k=10)

        print(f"=== temporal GConvGRU A/B over the {args.graph} graph, real data (2014-2024) ===")
        table = run_temporal(prices, graphs, warmup=260)

    print(table.to_string(index=False, float_format=lambda v: f"{v:+.3f}"))


if __name__ == "__main__":
    main()

