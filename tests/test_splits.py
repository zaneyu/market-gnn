"""Hygiene test 4 -- purge correctness: no training label window may overlap the
test window (plus embargo), and train/test never intersect."""

import numpy as np
import pandas as pd
import pytest

from marketgnn.splits import PurgedWalkForward

DATES = pd.bdate_range("2015-01-01", periods=1000)


def test_purge_and_embargo_hold():
    H, embargo = 21, 5
    cv = PurgedWalkForward(label_horizon=H, n_test=63, embargo=embargo, min_train=252, step=63)
    folds = list(cv.split(DATES))
    assert folds, "expected at least one fold"
    for f in folds:
        test_start = f.test_pos.min()
        # exact boundary: the last eligible train row is at test_start-embargo-H-1,
        # and the whole purge+embargo band immediately before test is empty.
        assert f.train_pos.max() == test_start - embargo - H - 1
        band = set(range(test_start - embargo - H, test_start))
        assert band.isdisjoint(f.train_pos.tolist())
        # every train label realizes strictly before the test window (+embargo)
        assert (f.train_pos + H).max() < test_start - embargo
        # train and test are disjoint; positions and dates agree
        assert set(f.train_pos.tolist()).isdisjoint(f.test_pos.tolist())
        assert (DATES[f.test_pos] == f.test_dates).all()
        assert (DATES[f.train_pos] == f.train_dates).all()


def test_rolling_window_is_actually_bounded():
    # Use a LATER fold so the rolling lower bound is strictly positive -- the
    # original assertion (>= -63) was vacuously true and never distinguished
    # rolling from expanding.
    min_train = 252
    roll = list(PurgedWalkForward(21, 63, min_train=min_train, step=63, expanding=False).split(DATES))
    exp = list(PurgedWalkForward(21, 63, min_train=min_train, step=63, expanding=True).split(DATES))
    later = next(f for f in roll if f.test_pos.min() > min_train)
    assert later.train_pos.min() == later.test_pos.min() - min_train
    # same fold under expanding still starts at 0 -> proves the bound bites
    exp_same = next(f for f in exp if f.test_pos.min() == later.test_pos.min())
    assert exp_same.train_pos.min() == 0


def test_first_test_respects_min_train():
    cv = PurgedWalkForward(21, 63, min_train=300)
    first = next(cv.split(DATES))
    assert first.test_pos.min() == 300


def test_rejects_unsorted_dates():
    bad = DATES[::-1]
    with pytest.raises(ValueError):
        list(PurgedWalkForward(21, 63).split(bad))


def test_too_short_yields_nothing():
    assert list(PurgedWalkForward(21, 63, min_train=252).split(DATES[:100])) == []
