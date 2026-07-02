"""Ridge baseline. Features arrive already cross-sectionally normalized, so this is
a plain linear map; NaNs (should be none post-normalization) are zero-filled."""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import Ridge


class RidgeModel:
    def __init__(self, alpha: float = 1.0, seed: int = 0):
        self.model = Ridge(alpha=alpha)  # seed accepted for a uniform constructor; unused

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RidgeModel":
        X = np.nan_to_num(np.asarray(X, float))
        y = np.asarray(y, float)
        self.model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(np.nan_to_num(np.asarray(X, float)))
