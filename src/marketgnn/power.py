"""Power calculation: can this study even detect the IC gap it hopes to find?

The reviewers flagged that ~weekly cadence over a few years gives few *effective*
observations once labels overlap, so a null could just mean "underpowered." This
simulates the per-date IC-difference series (GNN minus MLP) as an AR(1) with a
given mean effect and estimates the probability the HAC t-test rejects -- and the
minimum detectable effect (MDE) at 80% power.
"""

from __future__ import annotations

import numpy as np

from .evaluate import newey_west_tstat, two_sided_p


def _ar1(n, phi, mean, sd, rng):
    # AR(1) with the target mean as a genuine offset. Do NOT re-center to the
    # sample mean: that would pin the t-stat numerator to effect/se on every draw,
    # making the numerator deterministic and the "power" a near-step function
    # (understated at small effects, overstated at large). The sampling variation
    # in the mean is exactly what a Monte-Carlo power estimate must keep.
    e = rng.normal(0, sd, size=n)
    x = np.empty(n)
    x[0] = e[0]
    for t in range(1, n):
        x[t] = phi * x[t - 1] + e[t]
    return x + mean


def power(
    effect: float,
    *,
    n_dates: int,
    ic_sd: float = 0.08,
    phi: float = 0.4,
    hac_lag: int | None = None,
    alpha: float = 0.05,
    n_sims: int = 2000,
    seed: int = 0,
) -> float:
    """Probability the HAC t-test rejects H0 when the true mean IC gap == ``effect``."""
    rng = np.random.default_rng(seed)
    lag = hac_lag if hac_lag is not None else max(1, int(0.1 * n_dates))
    rejects = 0
    for _ in range(n_sims):
        x = _ar1(n_dates, phi, effect, ic_sd, rng)
        t, _ = newey_west_tstat(x, lag=lag)
        if two_sided_p(t) < alpha:
            rejects += 1
    return rejects / n_sims


def min_detectable_effect(
    *, n_dates: int, target_power: float = 0.8, ic_sd: float = 0.08, phi: float = 0.4, **kw
) -> float:
    """Smallest mean IC gap detectable at ``target_power`` (bisection over effect)."""
    cap = 0.2
    lo, hi = 0.0, cap
    for _ in range(24):
        mid = (lo + hi) / 2
        if power(mid, n_dates=n_dates, ic_sd=ic_sd, phi=phi, **kw) < target_power:
            lo = mid
        else:
            hi = mid
    if hi >= cap - 1e-6 and power(cap, n_dates=n_dates, ic_sd=ic_sd, phi=phi, **kw) < target_power:
        import warnings

        warnings.warn(
            f"target power {target_power} not reached even at IC gap {cap}; "
            f"study is underpowered at n_dates={n_dates} (returning the cap).",
            stacklevel=2,
        )
    return hi
