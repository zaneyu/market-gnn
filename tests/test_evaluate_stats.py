"""The significance machinery the review flagged as the #1 issue: HAC inference,
block bootstrap, and FDR must behave correctly on autocorrelated input."""

import numpy as np

from marketgnn.evaluate import (
    benjamini_hochberg,
    block_bootstrap_ci,
    iid_bootstrap_ci,
    long_short_spread,
    newey_west_tstat,
    two_sided_p,
)


def _ar1(n=400, phi=0.6, seed=0):
    rng = np.random.default_rng(seed)
    e = rng.normal(size=n)
    x = np.empty(n)
    x[0] = e[0]
    for t in range(1, n):
        x[t] = phi * x[t - 1] + e[t]
    return x + 0.05  # small positive mean


def test_hac_is_more_conservative_than_naive_under_autocorrelation():
    x = _ar1()
    naive_t = x.mean() / (x.std(ddof=1) / np.sqrt(len(x)))
    hac_t, hac_se = newey_west_tstat(x, lag=20)
    # positive autocorrelation inflates the naive t-stat; HAC must shrink it
    assert abs(hac_t) < abs(naive_t)
    assert hac_se > x.std(ddof=1) / np.sqrt(len(x))


def test_block_bootstrap_ci_is_wider_than_iid_under_autocorrelation():
    x = _ar1()
    lo_b, hi_b = block_bootstrap_ci(x, block=20, seed=1)
    lo_i, hi_i = iid_bootstrap_ci(x, seed=1)
    assert (hi_b - lo_b) > (hi_i - lo_i)


def test_benjamini_hochberg_controls_and_is_monotone():
    p = np.array([0.001, 0.009, 0.04, 0.2, 0.7])
    reject, q = benjamini_hochberg(p, alpha=0.05)
    # BH rejects a prefix of the sorted p-values; here the two smallest
    assert reject.tolist() == [True, True, False, False, False]
    assert np.all(np.diff(q[np.argsort(p)]) >= -1e-12)  # q-values non-decreasing in p


def test_hac_does_not_over_reject_pure_noise():
    # Any single draw can be spuriously significant ~5% of the time; test the
    # RATE over many seeds, which HAC should hold near the nominal level.
    rejects = 0
    trials = 200
    for s in range(trials):
        x = np.random.default_rng(s).normal(size=300)  # zero mean, iid
        t, _ = newey_west_tstat(x, lag=10)
        if two_sided_p(t) < 0.05:
            rejects += 1
    assert rejects / trials < 0.12  # near the 5% nominal, with HAC finite-sample slack


def test_long_short_spread_nan_on_constant_predictions():
    assert np.isnan(long_short_spread(np.ones(50), np.random.default_rng(0).normal(size=50)))


def test_qlike_minimized_at_perfect_forecast():
    from marketgnn.evaluate import qlike
    rng = np.random.default_rng(0)
    realized = rng.uniform(0.01, 0.5, size=200)
    assert abs(qlike(realized, realized)) < 1e-9           # perfect forecast -> 0
    assert qlike(realized, realized * 2) > 0               # wrong forecast -> positive
    assert qlike(realized, realized * 0.5) > 0
