"""Locks the core scientific property: with no graph, the GNN IS the MLP.

Torch-gated (skips in the core CI run, which stays torch-free). When the graph has
no real edges, the GNN falls back to self-loops only, so GNN(use_graph=True) and
MLP(use_graph=False) must produce identical predictions -- proving the A/B differs
by graph topology and nothing else.
"""

import numpy as np
import pytest

pytest.importorskip("torch")

from marketgnn.dataset import build_dataset, make_synthetic, rebalance_dates
from marketgnn.models.gnn import NeuralModel


def _dataset(kind):
    prices, volume, sectors, market = make_synthetic(n_days=700, n_assets=24, n_sectors=4, seed=0)
    rebal = rebalance_dates(prices.index, "W")
    return build_dataset(prices, volume, sectors, market, rebal, graph_kind=kind,
                         label_horizon=5, corr_window=60, k=5, warmup=260)


def test_gnn_equals_mlp_when_graph_is_empty():
    ds = _dataset("none")  # graphs are all None -> GNN uses self-loops only
    dates = list(ds.dates)
    tr, te = dates[:30], dates[30:40]

    gnn = NeuralModel(use_graph=True, epochs=15, patience=15, seed=0).fit(ds, tr, target="vol")
    mlp = NeuralModel(use_graph=False, epochs=15, patience=15, seed=0).fit(ds, tr, target="vol")
    p_gnn = gnn.predict(ds, te).to_numpy()
    p_mlp = mlp.predict(ds, te).to_numpy()

    assert np.allclose(p_gnn, p_mlp, atol=1e-6), "with no edges, GNN must reduce to the MLP"


def test_gnn_diverges_from_mlp_with_real_edges():
    ds = _dataset("correlation")
    dates = list(ds.dates)
    tr, te = dates[:30], dates[30:40]
    gnn = NeuralModel(use_graph=True, epochs=15, patience=15, seed=0).fit(ds, tr, target="vol")
    mlp = NeuralModel(use_graph=False, epochs=15, patience=15, seed=0).fit(ds, tr, target="vol")
    # real edges must actually change the function (message passing does something)
    assert not np.allclose(gnn.predict(ds, te).to_numpy(), mlp.predict(ds, te).to_numpy())
