"""Purged, embargoed walk-forward cross-validation.

Forward-return labels span a horizon H (in trading steps): a sample stamped at
date d only *realizes* at d+H. If d+H reaches into the test window, that training
row has effectively seen the future and must be purged. Overlapping labels are
the single most common way a market backtest leaks and silently inflates its IC,
so this splitter is the backbone of the whole project.

Everything is index-based over the sorted array of unique rebalance dates, which
makes the purge a pure integer computation the tests can assert on exactly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Sequence

import numpy as np


@dataclass(frozen=True)
class Fold:
    """One walk-forward fold, as integer positions into the sorted date array
    (for exact assertions) and the corresponding dates (for slicing panels)."""

    train_pos: np.ndarray
    test_pos: np.ndarray
    train_dates: np.ndarray
    test_dates: np.ndarray


class PurgedWalkForward:
    """Forward-only walk-forward with label purging and an embargo.

    Parameters
    ----------
    label_horizon : int
        Steps H the forward label spans. A train sample at position p realizes
        at p + H; it is eligible only if p + H < test_start - embargo.
    n_test : int
        Number of consecutive rebalance dates in each test block.
    embargo : int
        Extra gap (in steps) held out immediately before the test block, on top
        of the horizon purge. Guards against residual autocorrelation leaking
        across the train/test boundary.
    min_train : int
        Position of the first test block (i.e. minimum calendar history before
        testing begins).
    step : int | None
        Stride between successive test blocks. Defaults to ``n_test`` (disjoint,
        tiling test blocks).
    expanding : bool
        True -> training window grows from position 0. False -> rolling window
        of length ``min_train`` ending at the (purged) boundary.
    """

    def __init__(
        self,
        label_horizon: int,
        n_test: int,
        *,
        embargo: int = 0,
        min_train: int = 252,
        step: int | None = None,
        expanding: bool = True,
    ):
        if label_horizon < 0:
            raise ValueError("label_horizon must be >= 0")
        if n_test < 1:
            raise ValueError("n_test must be >= 1")
        if embargo < 0:
            raise ValueError("embargo must be >= 0")
        if min_train < 1:
            raise ValueError("min_train must be >= 1")
        self.label_horizon = label_horizon
        self.n_test = n_test
        self.embargo = embargo
        self.min_train = min_train
        self.step = step or n_test
        self.expanding = expanding

    def split(self, dates: Sequence) -> Iterator[Fold]:
        arr = np.asarray(dates)
        if len(arr) < self.min_train + self.n_test:
            return
        deltas = np.diff(arr.astype("datetime64[ns]"))
        if np.any(deltas <= np.timedelta64(0)):
            raise ValueError("dates must be strictly increasing and unique")

        n = len(arr)
        pos = np.arange(n)
        for test_start in range(self.min_train, n - self.n_test + 1, self.step):
            test_pos = pos[test_start : test_start + self.n_test]
            # Purge: a train label must realize strictly before the test window
            # opens, less the embargo. p + H < test_start - embargo.
            cutoff = test_start - self.embargo
            eligible = pos[(pos + self.label_horizon) < cutoff]
            if not self.expanding:
                eligible = eligible[eligible >= test_start - self.min_train]
            if eligible.size == 0:
                continue
            yield Fold(eligible, test_pos, arr[eligible], arr[test_pos])
