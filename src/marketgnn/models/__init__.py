"""Models share a minimal interface: ``fit(X, y) -> self`` and ``predict(X) -> np.ndarray``
for the tabular baselines; the neural models add a ``fit_graph``/``predict_graph`` path.

The scientific A/B is GNN vs MLP: the GNN is literally the MLP plus graph-conv
layers, sharing the exact same head, loss, and training loop, so zeroing the graph
recovers the MLP. Ridge and GBM are the "any-other-method" bar and may differ.
"""

from .ridge import RidgeModel

__all__ = ["RidgeModel", "build_model"]


def build_model(name: str, **kw):
    if name == "ridge":
        return RidgeModel(**kw)
    if name == "gbm":
        from .gbm import GBMModel

        return GBMModel(**kw)
    if name in ("mlp", "gnn"):
        from .gnn import NeuralModel

        return NeuralModel(use_graph=(name == "gnn"), **kw)
    raise ValueError(f"unknown model: {name}")
