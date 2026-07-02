"""Power analysis sanity: power rises with effect size and sample, and the MDE is
larger under stronger autocorrelation (fewer effective observations)."""

from marketgnn.power import min_detectable_effect, power


def test_power_increases_with_effect_and_sample():
    small = power(0.01, n_dates=120, n_sims=400, seed=0)
    big = power(0.05, n_dates=120, n_sims=400, seed=0)
    more = power(0.03, n_dates=400, n_sims=400, seed=0)
    less = power(0.03, n_dates=120, n_sims=400, seed=0)
    assert big > small
    assert more > less
    assert 0.0 <= small <= 1.0


def test_autocorrelation_raises_min_detectable_effect():
    mde_iid = min_detectable_effect(n_dates=150, phi=0.0, n_sims=300, seed=1)
    mde_ac = min_detectable_effect(n_dates=150, phi=0.6, n_sims=300, seed=1)
    assert mde_ac > mde_iid


def test_null_rejection_rate_near_nominal():
    # Zero true effect -> rejection rate must sit near alpha=0.05. The earlier
    # sample-mean-pinning bug drove this to exactly 0 (a near-step power curve),
    # so this is the assertion that actually catches it.
    assert 0.02 < power(0.0, n_dates=200, ic_sd=0.08, phi=0.4, n_sims=2000, seed=3) < 0.11
