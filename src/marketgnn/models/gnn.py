"""The controlled A/B: MLP vs GNN are the SAME network. GraphSAGE-mean layers
aggregate over each node's neighbours; setting ``use_graph=False`` swaps the real
edges for self-loops only, so the architecture, parameter count, head, loss, and
training loop are identical and the ONLY difference is whether graph topology is
visible. That isolates "does structure help" cleanly (H1), the way H2 isolates
"beyond degree". Torch-only (aggregation is hand-coded, no torch-geometric).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch import nn

from .. import graph as G
from .losses import soft_spearman_loss


def _mean_aggregate(h, edge_index, num_nodes):
    src, dst = edge_index  # message src -> dst, pooled at dst
    agg = torch.zeros_like(h)
    agg.index_add_(0, dst, h[src])
    deg = torch.zeros(num_nodes, device=h.device).index_add_(0, dst, torch.ones(dst.shape[0], device=h.device))
    return agg / deg.clamp(min=1.0)[:, None]


class _SageLayer(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.lin_self = nn.Linear(dim_in, dim_out)
        self.lin_neigh = nn.Linear(dim_in, dim_out)

    def forward(self, h, edge_index, n):
        return torch.relu(self.lin_self(h) + self.lin_neigh(_mean_aggregate(h, edge_index, n)))


class _Net(nn.Module):
    def __init__(self, n_features, hidden=32, layers=2):
        super().__init__()
        dims = [n_features] + [hidden] * layers
        self.convs = nn.ModuleList(_SageLayer(dims[i], dims[i + 1]) for i in range(layers))
        self.head = nn.Linear(hidden, 1)

    def forward(self, x, edge_index, n):
        h = x
        for conv in self.convs:
            h = conv(h, edge_index, n)
        return self.head(h).squeeze(-1)


class NeuralModel:
    """Trains one net across all dates with the shared ranking loss and early
    stopping on a time-held-out (purged) tail of the training dates."""

    def __init__(self, *, use_graph=True, hidden=32, layers=2, lr=1e-3, epochs=200, patience=20,
                 val_frac=0.2, val_gap=1, seed=0):
        self.use_graph = use_graph
        self.hidden, self.layers = hidden, layers
        self.lr, self.epochs, self.patience, self.val_frac = lr, epochs, patience, val_frac
        self.val_gap = val_gap  # rebalance steps dropped between train and val (purge)
        self.seed = seed
        self.net = None

    def _edges(self, dataset, date, n, device):
        """Symmetrized + self-looped edges for a date; self-loops-only if blindfolded."""
        g = dataset.graphs.get(date)
        if self.use_graph and g is not None and g.num_edges:
            g = G.add_self_loops(G.symmetrize(g))
        else:
            loops = np.arange(n)
            g = G.Graph(np.vstack([loops, loops]), np.ones(n, np.float32), np.arange(n))
        return torch.as_tensor(g.edge_index, dtype=torch.long, device=device)

    def _date_tensors(self, dataset, dates, target, device):
        y_all = dataset.y_ret if target == "ret" else dataset.y_vol
        out = []
        for d in dates:
            x = torch.as_tensor(dataset.X.loc[d].to_numpy(copy=True), dtype=torch.float32, device=device)
            y = torch.as_tensor(y_all.loc[d].to_numpy(copy=True), dtype=torch.float32, device=device)
            keep = torch.isfinite(y)
            if keep.sum() < 3:
                continue
            out.append((d, x, y, keep, self._edges(dataset, d, x.shape[0], device)))
        return out

    def fit(self, dataset, train_dates, target="ret"):
        torch.manual_seed(self.seed)
        device = "cpu"
        n_feat = dataset.X.shape[1]
        self.net = _Net(n_feat, self.hidden, self.layers).to(device)
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr)

        train_dates = list(train_dates)
        cut = max(1, int(len(train_dates) * (1 - self.val_frac)))
        # Purge the val split like everything else: drop val_gap steps between the
        # train tail and the val head so the last train label can't overlap val.
        tr = self._date_tensors(dataset, train_dates[:cut], target, device)
        va = self._date_tensors(dataset, train_dates[cut + self.val_gap :], target, device)
        if not tr:
            return self

        best, best_state, bad = np.inf, None, 0
        for _ in range(self.epochs):
            self.net.train()
            opt.zero_grad()
            loss = sum(
                soft_spearman_loss(self.net(x, ei, x.shape[0])[keep], y[keep]) for _, x, y, keep, ei in tr
            ) / len(tr)
            loss.backward()
            opt.step()

            val = self._val_loss(va, target) if va else float(loss.item())
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
    def _val_loss(self, va, target):
        self.net.eval()
        return float(
            sum(soft_spearman_loss(self.net(x, ei, x.shape[0])[keep], y[keep]) for _, x, y, keep, ei in va) / len(va)
        )

    @torch.no_grad()
    def predict(self, dataset, dates):
        self.net.eval()
        device = "cpu"
        parts = []
        for d in dates:
            x = torch.as_tensor(dataset.X.loc[d].to_numpy(copy=True), dtype=torch.float32, device=device)
            ei = self._edges(dataset, d, x.shape[0], device)
            s = self.net(x, ei, x.shape[0]).cpu().numpy()
            parts.append(pd.Series(s, index=dataset.X.loc[d].index))
        idx = pd.MultiIndex.from_tuples(
            [(d, a) for d in dates for a in dataset.X.loc[d].index], names=["date", "asset"]
        )
        return pd.Series(np.concatenate([p.to_numpy() for p in parts]), index=idx)
