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
    # Stationary AR(1) whose MARGINAL sd equals ``sd`` (the empirical IC sd the caller
    # measured). A raw AR(1) driven by innovations of sd ``sd`` has marginal sd
    # sd/sqrt(1-phi^2) -- up to ~2.3x too wide at phi=0.9 -- which would over-disperse
    # the series and inflate the MDE. So scale the innovation to sd*sqrt(1-phi^2) and
    # start from the stationary distribution.
    #   Do NOT re-center to the sample mean: that would pin the t-stat numerator to
    # effect/se on every draw, making "power" a near-step function. The sampling
    # variation in the mean is exactly what a Monte-Carlo power estimate must keep.
    sigma_e = sd * np.sqrt(max(1e-12, 1.0 - phi * phi))
    x = np.empty(n)
    x[0] = rng.normal(0, sd)                      # stationary start
    for t in range(1, n):
        x[t] = phi * x[t - 1] + rng.normal(0, sigma_e)
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
    # use the SAME HAC lag the real evaluation uses (Newey-West rule of thumb), so the
    # MDE describes the test actually run -- not a larger, more conservative lag.
    lag = hac_lag if hac_lag is not None else int(np.floor(4 * (n_dates / 100) ** (2 / 9)))
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
