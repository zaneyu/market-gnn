# Run 12 — backtest-overfitting hardening: PBO (CSCV) + deflated Sharpe for the surviving signals

**Status:** draft design (2026-07-16). Single-module extension to market-gnn.

## Motivation

The repo's nulls are powered and control-checked, but its two *positive* real-data results —
the short-term reversal control (Run 6: IC +0.015, t 3.4) and the vol-information row
(Run 11: +0.038, t 2.65, borderline-MDE) — have so far faced only single-path walk-forward
inference. The standard López-de-Prado objection remains open: **a result selected from (or
even implicitly tuned over) a configuration space overstates its significance**, and one
walk-forward path gives a point estimate with no distribution. Run 12 closes that with the
standard machinery: **CSCV/PBO** (probability of backtest overfitting) over the signal's
natural configuration grid, and the **deflated Sharpe ratio** (DSR) for the reversal
long-short, adjusting for multiple testing, non-normality, and track length.

## Scientific question

> After accounting for the configuration search space (PBO via CSCV) and multiple-testing /
> non-normality deflation (DSR), do the repo's two surviving positives still stand — and at
> what confidence?

Honest scope: this is **evaluation hardening of already-reported results**, pre-registered
before running. If reversal's DSR collapses or PBO is high, that gets reported with the same
prominence as the original positive — the repo's whole thesis.

## Method (standard constructions, stated precisely)

### PBO via CSCV (Bailey–Borwein–López de Prado–Zhu)

- **Configuration grid:** the reversal signal's natural knobs, pre-registered:
  `lookback ∈ {1, 3, 5}` × `horizon ∈ {1, 3, 5}` trading days (9 configs; skip=0 fixed) —
  the space someone tuning "short-term reversal" would actually search. Portfolio: **quintile**
  long-short, `q = 0.2` (18 names/leg on 90) — pinned here because that is what
  `costs.longshort_portfolio` defaults to and what Run 6 actually reported (RESULTS.md's
  "decile" label for Run 6 is a mislabel this run also corrects). Gross returns.
- **Common time axis (the load-bearing construction):** CSCV requires a T×N P&L matrix on ONE
  time axis (the paper's M matrix). Naive step=horizon sampling puts the 9 configs on three
  different date grids (measured on the real panel: 2,506 / 835 / 501 periods; intersection a
  degenerate 167 dates) with incommensurable per-period returns (SR scales ~√h, mechanically
  favouring long horizons in-sample). So every config is converted to a **daily** return
  series via **Jegadeesh–Titman overlapping tranches**: 1/h of the book is re-formed each day
  and each tranche held h days, the combined book marked daily. All 9 configs then share the
  ~2,506-day post-warmup axis with 1-day returns, and cross-config Sharpe comparisons (and V
  in the DSR) are apples-to-apples.
- **CSCV:** partition the daily axis into `S = 16` contiguous, equal-length blocks (~157
  days/block), with an **(h_max − 1) = 4-day embargo** at every block boundary (daily-marked
  h=5 books straddle boundaries by up to 4 days; the embargo costs ≤ ~3% of each block). For
  every combination of S/2 blocks as in-sample (C(16,8) = 12,870 splits, all enumerated;
  per-block precomputed sums / sums-of-squares make this ~1s — do not recompute per split):
  pick the config with the best IS Sharpe, record its OOS **rank** among all 9.
  **Convention (pinned, since N=9 is odd):** relative rank ω̄ = r/(N+1), λ = logit(ω̄),
  **PBO = P(λ < 0) ⇔ OOS rank ≤ 4 of 9** (strictly below the median, per the paper's
  "underperforms the median OOS"). Under pure noise the theoretical PBO is therefore
  **4/9 ≈ 0.444**, not 0.5. Report PBO and the OOS-rank histogram over ranks 1..9 under this
  convention. NOTE: single-realization PBO is highly dispersed (the 12,870 splits reuse the
  same 16 block statistics; measured seed-to-seed range 0.26–0.94 on noise) — the real-data
  PBO is reported as one realization with that dispersion caveat stated, and the tests
  calibrate on seed-averages (below).

### Deflated Sharpe ratio (Bailey–López de Prado)

- **PSR** (probabilistic Sharpe): `PSR(SR*) = Φ( (SR − SR*)·√(T−1) / √(1 − γ₃·SR + (γ₄−1)/4·SR²) )`
  with skew γ₃ and **raw (non-excess) kurtosis** γ₄ (m₄/m₂², normal = 3 — pinned, because the
  classic implementation bug is feeding excess kurtosis) estimated from the strategy's
  per-period returns via plain numpy moments. Note the normal-case reduction is
  `Φ((SR−SR*)·√(T−1)/√(1+SR²/2))` (Lo/Mertens) — the SR²/2 term does NOT vanish at skew 0 /
  kurt 3, and the formula test must hand-compute exactly that.
- **HAC-aware effective sample size (the repo's own standard applied to itself):** PSR assumes
  serial independence, but the repo's methodological spine is that these series are
  autocorrelated and iid inference overstates significance. So PSR/DSR are computed with
  `T_eff = T · (se_naive / se_HAC)²` in place of T — `se_HAC = newey_west_tstat(x)[1]` and
  `se_naive = newey_west_tstat(x, lag=0)[1]` (at lag 0 the Bartlett terms vanish, giving the
  naive √(γ₀/n) in the SAME 1/n variance convention; do NOT mix in `std(ddof=1)/√T`, which
  differs by √(n/(n−1))) — and the iid-T variant is shown alongside as the optimistic bound. Without this, "hardening" would
  be *softer* than the repo's HAC t-stats on the same series.
- **DSR** = PSR evaluated at the **expected maximum Sharpe under the null** across N trials:
  `SR* = √V · ((1−γ)·Φ⁻¹(1−1/N) + γ·Φ⁻¹(1−1/(N·e)))`, V = cross-config variance of the (daily,
  common-axis) Sharpe estimates, γ = Euler–Mascheroni. Domain: **N ≥ 2** (at N=1 the formula is
  −∞; `expected_max_sharpe` asserts n_trials ≥ 2). Caveats stated: the E[max] approximation
  assumes independent trials — the 9 nested-lookback configs are heavily correlated, so
  effective N < 9 and SR* is conservative-to-uncertain there.
- **Trial-count sensitivity (pre-registered):** DSR reported at **N ∈ {9, 25, 100}** — N=9 is
  the honest count for THIS grid (headline); 25 and 100 bound the repo's realized wider search
  (Run 6 picked reversal from 3 controls; Run 7 ran liquidity-conditioned variants). This turns
  the "wider search" caveat into a number instead of a paragraph.
- Applied to: the pre-registered reversal config (lookback=1, horizon=1 — the Run 6 headline)
  quintile long-short gross daily returns. Reported: SR, PSR(0), DSR at each N, skew, kurtosis,
  T, T_eff.

### Vol-information row (Run 11)

CSCV/PBO needs a config grid; Run 11 was pre-registered as a single configuration (no grid was
searched), so PBO does not apply — fabricating a grid post-hoc would *manufacture* an
overfitting test. Instead, the non-normality-aware significance applies at the **IC level**:
`ic_psr(ic) ≡ psr(ic, sr_star=0.0)` — plain PSR with the HAC-aware T_eff, **never touching
`expected_max_sharpe`** (whose N=1 value is −∞; nothing was searched, so there is no maximum
to deflate against). This is stated plainly: Run 11's protection is pre-registration, not CSCV.

## Pre-registered reading of outcomes

- **Reversal:** PBO low (< 0.2) and DSR > 0.95 → the control survives hardening (still gross,
  still survivorship-exposed — those caveats stand unchanged). PBO high or DSR < 0.95 → the
  repo's one "real" return effect is itself partially a selection artifact; reported with the
  same prominence as Run 6, and Run 6's framing updated to match.
- Either way, the grid, S, and all thresholds above are fixed BEFORE the real-data run.

## Module & interfaces

New module `src/marketgnn/overfit.py` (torch-free):
- `daily_strategy_returns(prices, *, lookback, horizon, q=0.2) -> pd.Series` — **daily** gross
  return series of the reversal quintile long-short via Jegadeesh–Titman overlapping tranches:
  each day a new tranche (1/horizon of the book) is formed from the signal and held `horizon`
  days; the combined book is marked daily. **Marking conventions (pinned, so the hand-computed
  test has one right answer):** within a tranche, fixed equal formation weights held for the h
  days (tranche daily return = mean of long-leg daily returns − mean of short-leg daily
  returns at formation membership; no intra-tranche re-weighting); during the first h−1 ramp-up
  days the book holds fewer than h tranches and is **renormalized over the live tranches**
  (mean of live-tranche returns), not zero-padded. All configs share the daily post-warmup
  axis. Signal construction reuses the `signals.py` reversal definition (sign, lookback,
  skip=0); the leg construction mirrors `costs.longshort_portfolio`'s quantile logic with q
  pinned 0.2.
- `cscv_pbo(returns_matrix, *, n_blocks=16, embargo=4) -> dict` — CSCV over a [n_days ×
  n_configs] DataFrame on one common axis; contiguous blocks, `embargo` days dropped at each
  block boundary; per-block precomputed sums/sums-of-squares (never recompute per split);
  enumerates all C(n_blocks, n_blocks/2) splits. Convention pinned: PBO = fraction of splits
  with the IS-best config's OOS rank **strictly below the median** (rank ≤ (N−1)/2 for odd N;
  noise ⇒ 4/9). Returns {pbo, oos_rank_freq (ranks 1..N), n_splits, **n_days_used**} —
  `n_days_used` exposes the post-embargo day count (`n_days − (n_blocks−1)·embargo`) so the
  embargo is externally testable (a no-op embargo would otherwise pass every calibration test).
  Asserts n_blocks even and **n_days ≥ 10·n_blocks** (blocks of 2 periods would make IS Sharpe
  pure noise).
- `sharpe(returns) -> float` (per-period, native cadence), `psr(returns, sr_star=0.0, *,
  t_eff=None) -> float` (uses `t_eff` in place of T when given — the HAC-aware variant),
  `expected_max_sharpe(n_trials, var_sr) -> float` (**asserts n_trials ≥ 2**),
  `dsr(returns, n_trials, var_sr, *, t_eff=None) -> float` — the pinned formulas, raw
  (non-excess) kurtosis via numpy moments, T < 10 → NaN guard.
- `hac_t_eff(returns) -> float` — `T · (se_naive/se_HAC)²` from `newey_west_tstat`.
- `run_overfit(prices) -> (pd.DataFrame, dict)` — builds the 9-config daily matrix, runs
  CSCV/PBO, computes SR/PSR/DSR (N ∈ {9, 25, 100}, iid-T and T_eff variants) for the
  pre-registered headline config; returns (per-config table, summary dict).
- `ic_psr(ic_series) -> float` — `psr(ic, sr_star=0.0, t_eff=hac_t_eff(ic))`; never calls
  `expected_max_sharpe`.
- `main()` — real-data run printing the config table, the PBO summary, and the DSR block;
  `--synthetic-planted` mode for the positive/negative controls below.

Reuses: `data.download.load_market` (raise-not-fallback), `data.universe.default_universe`,
`evaluate` helpers where applicable. No new dependencies (numpy/pandas only; Φ and Φ⁻¹ via
`scipy.stats.norm` which is already a repo dependency through `evaluate.py`).

## Tests (`tests/test_overfit.py`, TDD)

1. `test_cscv_pbo_detects_pure_noise` — **negative control**: 9 iid-noise "configs" (no true
   skill). Single-realization PBO is highly dispersed (measured 0.26–0.94), so the test asserts
   the **mean PBO over ≥20 seeds** ≈ the theoretical noise value under the pinned convention,
   **4/9 ≈ 0.444** (strictly-below-median, odd N), within a modest band. This is the test that
   PBO measures overfitting rather than anything else — and that the convention is implemented
   as pinned.
2. `test_cscv_pbo_low_for_true_skill` — **positive control**: one config with a genuine mean
   shift among noise configs, sized well above the noise floor, seed-averaged → mean PBO well
   below 4/9 (the skilled config keeps winning OOS).
3. `test_sharpe_psr_dsr_formulas` — PSR against the hand-computed normal case
   `Φ((SR−SR*)·√(T−1)/√(1+SR²/2))` (skew 0, raw kurt 3 — the SR²/2 term does NOT vanish);
   DSR < PSR(0) when n_trials ≥ 2 and V > 0; `expected_max_sharpe` increasing in n_trials **over
   its domain n ≥ 2** and asserts on n_trials < 2; T<10 → NaN; feeding excess kurtosis (0 for a
   normal) would change the answer — the test pins raw-kurtosis semantics.
4. `test_daily_strategy_returns_is_pit_and_tranched` — perturbing future prices doesn't change
   earlier daily returns; the series is daily (consecutive dates); for horizon h the book at
   day t reflects signals formed on days t−h+1..t (tranche structure), verified on a small
   hand-computable panel.
5. `test_run_overfit_grid_shape` — the config table has exactly the 9 pre-registered rows on a
   common daily axis, and the summary carries pbo / dsr (N ∈ {9,25,100}) / psr / t_eff keys
   (on synthetic data); `cscv_pbo`'s `n_days_used == n_days − (n_blocks−1)·embargo` (the
   embargo actually fires — a no-op embargo would pass the calibration tests).

## Deliverable

`overfit.py` + `test_overfit.py`; **Run 12** in RESULTS.md (config table, PBO, DSR at
N ∈ {9,25,100} with iid-T and T_eff variants, honest read), README bullet + roadmap + counts,
paper/note.md bullet; `python -m marketgnn.overfit` prints the tables; **the "decile" mislabel corrected at its SOURCE and downstream** — `costs.py`
says "decile" in its module docstring, `longshort_portfolio` docstring, and printed header while
computing q=0.2 quintiles, and RESULTS.md's Run 6 inherited it; all corrected to "quintile" (else
the confusion regenerates); Run 6's framing updated further if the PBO/DSR verdict demands it;
site card updated after validation.

## Out of scope (YAGNI)

Full CPCV path-wise walk-forward re-evaluation of every historical run (CSCV on the two
positives is the load-bearing piece); costs/net-of-fee re-analysis (Run 6's ~0.9 bp breakeven
already covers it); applying PBO to pre-registered single-config results (would fabricate a
search that never happened); any change to the underlying signals.
