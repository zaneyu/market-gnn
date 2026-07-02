"""LightGBM baseline -- the honest tough benchmark. If the GNN can't beat a
gradient-boosted tree on the same features, that's a finding, not a failure.
Import is lazy (via models.build_model) so the core stays torch/lightgbm-free."""

from __future__ import annotations

import numpy as np


class GBMModel:
    def __init__(self, n_estimators: int = 300, learning_rate: float = 0.03, num_leaves: int = 31, seed: int = 0):
        import lightgbm as lgb

        self.model = lgb.LGBMRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            num_leaves=num_leaves,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.8,
            random_state=seed,
            verbosity=-1,
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GBMModel":
        self.model.fit(np.nan_to_num(np.asarray(X, float)), np.asarray(y, float))
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(np.nan_to_num(np.asarray(X, float)))
