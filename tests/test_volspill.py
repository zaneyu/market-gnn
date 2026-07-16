"""Run 11 — volatility spillover over the co-holding graph (DY-inspired).

Tests the innovation-based spillover estimand. The two load-bearing guards:
test 3 (planted spatial-ARCH recovery — power) and test 4 (the level-confound
market — validates the CONTROL: a naive level signal shows a large spurious IC
there while the innovation design is ~null). Torch-free."""

import numpy as np
import pandas as pd
import pytest

from marketgnn import graph as G
from marketgnn import volspill


# --------------------------------------------------------------- primitives
def test_trailing_vol_is_pit():
    rng = np.random.default_rng(0)
    rets = pd.DataFrame(rng.normal(0, 0.01, size=(80, 4)),
                        index=pd.bdate_range("2020-01-01", periods=80),
                        columns=list("ABCD"))
    v1 = volspill.trailing_vol(rets, lookback=20)
    t = rets.index[50]
    rets2 = rets.copy()
    rets2.iloc[51:] += 5.0                    # perturb strictly-future rows
    v2 = volspill.trailing_vol(rets2, lookback=20)
    assert np.allclose(v1.loc[t], v2.loc[t])  # vol at t unchanged
    # short history -> NaN, never a 2-observation "vol"
    assert v1.iloc[:19].isna().all().all()
    assert np.isfinite(v1.loc[t]).all()


def test_neighbour_innovation_excludes_self_and_handles_nan():
    nodes = ["A", "B", "C", "D"]
    # A-B, A-C edges; D isolated
    g = G.edges_from_pairs(pd.DataFrame({"src": ["A", "A"], "dst": ["B", "C"]}),
                           nodes, symmetric=True)
    innov = pd.Series({"A": 100.0, "B": 0.02, "C": 0.04, "D": 0.5})
    out = volspill.neighbour_innovation(innov, g)
    # A's own huge innovation never enters its signal: mean of B,C = 0.03
    assert out["A"] == pytest.approx(0.03)
    assert np.isnan(out["D"])                 # isolated -> NaN, not 0
    # NaN neighbour excluded with weights renormalized (not poisoning the mean)
    innov2 = innov.copy()
    innov2["B"] = np.nan
    out2 = volspill.neighbour_innovation(innov2, g)
    assert out2["A"] == pytest.approx(0.04)   # only C remains
    innov3 = innov.copy()
    innov3["B"] = innov3["C"] = np.nan
    assert np.isnan(volspill.neighbour_innovation(innov3, g)["A"])  # all-NaN -> NaN
    # edge WEIGHTS are actually applied (a plain mean would return 0.03 here)
    gw = G.edges_from_pairs(pd.DataFrame({"src": ["A", "A"], "dst": ["B", "C"],
                                          "weight": [3.0, 1.0]}), nodes, symmetric=True)
    outw = volspill.neighbour_innovation(innov, gw)
    assert outw["A"] == pytest.approx((3 * 0.02 + 1 * 0.04) / 4)  # 0.025, not 0.03


# --------------------------------------------------------------- planted power
def test_planted_spatial_arch_is_recovered():
    """Positive control: the residualized innovation IC over the TRUE graph recovers the
    planted spatial-ARCH spillover, and the multi-seed rewire null sits well below it —
    asserting the contrast, not just 'signal exists' (the Run 9 blindfold lesson)."""
    prices, graph = volspill.make_synthetic_spatial_arch(seed=0)
    table = volspill.run_volspill(prices, graph_provider=_static_provider(graph),
                                  warmup=260)
    row = {(r["edges"], r["signal"]): r for r in table.to_dict("records")}
    ic_true = row[("coholding", "spill_resid")]["mean_ic"]
    ic_rewire = row[("rewire", "spill_resid")]["mean_ic"]
    assert ic_true > 0.06, f"planted spillover not recovered: {ic_true:.4f}"
    assert row[("coholding", "spill_resid")]["hac_t"] > 3.0
    assert ic_true > ic_rewire + 0.05, f"contrast too small: {ic_true:.4f} vs rewire {ic_rewire:.4f}"


def _static_provider(graph):
    """A run_volspill provider serving one fixed graph (+ its rewires) at every date."""
    def coholding(asof, seed=0):
        return graph

    def rewire(asof, seed=0):
        return G.degree_preserving_rewire(graph, seed=seed)

    return {"coholding": coholding, "rewire": rewire}


def test_level_confound_is_killed_by_innovation_design():
    """The control-validation guard: gamma=0 (ZERO transmission) but per-block base
    variances (neighbours share persistent vol levels — the real-world confound). The
    naive LEVEL signal residualized on own sigma20 alone shows a large spurious IC
    (asserting the confound is real and present); the innovation signal with the
    two-regressor control is ~null."""
    prices, graph = volspill.make_synthetic_spatial_arch(
        gamma=0.0, beta=0.85, block_omega_spread=1.0, seed=1)
    naive_ic, innov_ic = volspill.confound_probe(prices, graph, warmup=260)
    assert naive_ic > 0.15, f"confound absent — the probe market is broken: {naive_ic:.4f}"
    assert abs(innov_ic) < 0.05, f"innovation design leaks the level confound: {innov_ic:.4f}"


def test_fdr_family_excludes_controls():
    # full-length market: a short one (~26 eval dates) trips power.py's underpowered-cap
    # warning, which the suite promotes to an error
    prices, graph = volspill.make_synthetic_spatial_arch(seed=2)
    table = volspill.run_volspill(prices, graph_provider=_static_provider(graph),
                                  warmup=260)
    ctrl = table[(table["edges"] == "rewire") | (table["signal"] == "own_vol")]
    assert (~ctrl["fdr_sig"]).all()
    assert ctrl["q"].isna().all()
    fam = table[(table["edges"] == "coholding")]
    assert fam["q"].notna().all()             # the discovery family gets real q-values


def test_run_volspill_kills_confound_in_production_path():
    """Test 4b — the PRODUCTION-path confound guard. Test 4 validates confound_probe, a
    parallel implementation; mutation testing showed run_volspill's own signal/control
    wiring could be silently degraded (level signal, dropped sigma250 regressor) with all
    other tests green — the fully-confounded mutation reports a spurious FDR-significant
    +0.27 on this very market. So the real table-producing loop must itself be run on the
    gamma=0 clustered-level market and come back ~null."""
    prices, graph = volspill.make_synthetic_spatial_arch(
        gamma=0.0, beta=0.85, block_omega_spread=1.0, seed=1)
    table = volspill.run_volspill(prices, graph_provider=_static_provider(graph),
                                  warmup=260)
    row = {(r["edges"], r["signal"]): r for r in table.to_dict("records")}
    resid = row[("coholding", "spill_resid")]
    assert abs(resid["mean_ic"]) < 0.05, f"production loop leaks the confound: {resid['mean_ic']:.4f}"
    assert not resid["fdr_sig"]
