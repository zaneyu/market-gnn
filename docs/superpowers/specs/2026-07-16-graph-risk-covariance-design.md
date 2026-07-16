# Run 10 — Graphs for risk, not alpha: graph-structured covariance & the minimum-variance portfolio

**Status:** approved design (2026-07-16). Single-module extension to market-gnn.

## Motivation

Runs 1–9 established that a stock-relationship graph adds **no cross-sectional return
signal** that survives leak-free, powered, control-checked evaluation. But covariance
estimation is a *different* problem where structural priors are known to help (that is the
entire reason Ledoit–Wolf shrinkage exists): with n≈90 assets and short trailing windows the
sample covariance is ill-conditioned. A graph is a structural prior. So the honest question
is whether graphs help **risk** even though they don't help **alpha** — a nuanced, truthful
reframe that is a *positive-result* direction rather than another null.

## Scientific question

> Does knowing a stock-relationship graph improve **out-of-sample covariance estimation** —
> measured by the realized volatility of the global minimum-variance portfolio (GMVP) —
> beyond standard estimators (sample, Ledoit–Wolf)? And does any benefit **concentrate in
> high-correlation / crisis regimes**?

## Critical design decision — which graph (non-circularity)

The **13F co-holding graph is the headline structure**, because it is built from *holdings,
not returns* — so using it to regularize a *return* covariance is a genuine external prior,
not a tautology. A correlation-kNN graph regularizing a return covariance is near-circular
(same information), so it appears **only as a clearly-labeled reference ceiling**, never as
the reported result. This is the choice that makes the finding meaningful.

## Graph → dense adjacency (prerequisite for A and B)

Both graph estimators need a **symmetric dense boolean adjacency aligned to the covariance's
asset order**, which the `Graph` object does not provide (it exposes only `edge_index [2,E]` into
`graph.nodes`). Specify a helper:

- `dense_adjacency(graph, nodes) -> np.ndarray[n,n] (bool)`: assert `list(graph.nodes) == list(nodes)`
  (hard fail on mismatch — prevents the silent wrong-asset corruption when a provider is built in
  `cusip_map` order but the covariance uses `prices.columns` order); scatter `edge_index` into a
  matrix; **symmetrize** `A = A | A.T` (required because `correlation_knn` is a *directed* top-k
  graph — an asymmetric mask would make the target/penalty non-symmetric and `Σ̂` not a valid
  covariance). The graph provider is built with `nodes = list(prices.columns)` so alignment holds.

## Estimators

Baselines (no graph):
- **sample** — S (trailing-window sample covariance; complete-case, see NaN policy below).
- **ledoit_wolf** — linear shrinkage of S toward the constant-correlation target (the standard
  to beat).
- **diagonal** — diagonal of S (a floor; ignores all covariance).

Graph estimators (headline graph = co-holding; reference = correlation-kNN):
- **A — graph-informed shrinkage (conditional-independence prior).** `Σ̂ = δ·T + (1−δ)·S` with
  the Ledoit–Wolf-optimal intensity δ applied *everywhere*, where target `T` treats the graph as
  a conditional-independence prior: **on-graph** pairs shrink toward the constant-correlation
  target, **off-graph** pairs shrink toward **zero** (unlinked ⇒ low covariance), diagonal =
  sample variances. **`T` is then PSD-projected** (eigen-clip negatives to a small floor) before
  combining — zeroing arbitrary off-diagonals of the constant-correlation matrix makes it
  *indefinite* at realistic equity correlations, and a convex combination `δT+(1−δ)S` is only
  guaranteed PSD when both operands are; without the projection the GMVP optimizes over an
  indefinite non-covariance and returns extreme leverage artifacts (the pre-merge review team
  caught exactly this). (An earlier "keep raw S on edges" variant was also rejected during
  implementation: it under-shrinks exactly the noisiest entries. The conditional-independence
  version is the fair test — it can beat plain LW iff the graph correctly flags weakly-related pairs.)
- **B — graph-penalized graphical lasso.** Sparse precision matrix Θ with a **per-edge** L1
  penalty matrix Λ (low `edge_penalty` on graph edges, high `offedge_penalty` off-graph; diagonal
  unpenalized). Solved by a **numpy-only ADMM** with a fully specified numerical contract:
  - **Solve in correlation space** (unit-diagonal): daily-return precisions are O(1e4), so a
    fixed penalty/tol on the raw covariance is meaningless. Standardize `C = D⁻¹SD⁻¹`
    (`D = diag(std)`), solve, and return `Σ̂ = D · (correlation-space Σ) · D`.
  - Pre-condition: `C ← C + ε·tr(C)/n · I` (ε=1e-3) so it is SPD before solving.
  - ADMM: **Θ-update** = eigendecompose the symmetric `M = ρ(Z−U) − C` and map its eigenvalues
    `d_i ↦ (d_i + √(d_i²+4ρ))/(2ρ)` (analytic prox of −logdet; always yields PD Θ); **Z-update** by
    element-wise soft-threshold of `Θ+U` at `Λ/ρ` (diagonal unpenalized); scaled dual update
    `U ← U + Θ − Z`; `rho=1.0`, `max_iter=500`.
  - **Convergence:** the standard Boyd size-scaled primal/dual criterion — `eps = √p·abs_tol +
    rel_tol·max(‖Θ‖,‖Z‖)` with `abs_tol=1e-4`, `rel_tol=1e-2`, `p=n²` (a fixed Frobenius tol never
    triggers for a 90×90 matrix; this converges in 11/11 real windows).
  - Guards: symmetrize each iterate; on non-convergence at `max_iter`, **fall back to estimator A**,
    never returning a non-invertible matrix.

## The null (same rigor as the rest of the repo)

Every graph estimator is re-run with a **degree-preserving rewired** co-holding graph
(`graph.degree_preserving_rewire`), **averaged over ≥3 seeds** (matching `run_leadlag`'s
`rewire_seeds=(0,1,2)` — one Maslov–Sneppen draw is a noisy null). The real graph must beat its
own rewire, or the benefit is just sparsity/degree, not *topology*.

## Evaluation protocol

- **Universe / data:** the existing 90 large-caps (`default_universe`, cached parquet); the
  co-holding graph from the cached 13F snapshots (PIT public-date logic already built). The
  graph provider is built with `nodes = list(prices.columns)` so adjacency aligns to the cov.
- **Windows — a manual rolling rebalance loop** (NOT `PurgedWalkForward`, whose expanding
  train/test-block + overlapping-label-purge API is built for cross-sectional IC and cannot
  express a fixed trailing window + monthly rebalance + hold-to-next; the repo's own holding-
  period evaluations in `leadlag.py`/`conditioning.py` roll a manual loop, and this follows that
  pattern). Concretely: `rebal = [d for d in rebalance_dates(prices.index, "M") if pos(d) >= window]`;
  at each `d`, estimate `Σ̂ = estimator(rets.loc[:d].iloc[-window:])` (returns strictly ≤ d),
  form `w = gmvp(Σ̂)`, and record the realized portfolio return over `(d, next_rebal]` (strictly
  forward). `window = 252` trading days, monthly rebalance, no lookahead by construction (window
  ends at d, holding starts after d — no overlap, so no purge needed).
- **Portfolio:** unconstrained GMVP, `w ∝ Σ̂⁻¹ 1`, normalized to `1ᵀw = 1`. Unconstrained
  because long-only constraints mask covariance-quality differences. Long-only GMVP reported as
  a robustness check only. NaN/inversion policy in the Portfolio section below.
- **Primary metric — QLIKE** (the repo already has `evaluate.qlike`, the standard noise-robust
  proper loss for variance forecasts; it sidesteps the mean-vs-variance confound below).
  **Horizon must match** (QLIKE differentials are not scale-invariant): the forecast `wᵀΣ̂w` is a
  **daily** variance (Σ̂ is estimated from daily returns), so the realized proxy is the **mean
  daily squared portfolio return over the holding period** `(d, next_rebal]` (same daily
  horizon) — NOT the squared ~21-day holding return. Lower QLIKE = better. **Co-primary:**
  annualized OOS realized volatility of each estimator's GMVP (the interpretable number).
  Note: `evaluate.qlike` returns the *aggregated* mean loss; the paired DM test (below) needs the
  **per-period** QLIKE summand `r/f − log(r/f) − 1`, computed inline, not the aggregate.
- **Significance (paired, correctly specified):** neither `block_bootstrap_ci` nor
  `newey_west_tstat` is paired — each tests the *mean of one series* — so the paired **difference
  series is formed by hand first** (as `conditioning.py` does with `spread = low − high`):
  - Realized-vol comparison: `d_t = (r^est_t − r̄^est)² − (r^LW_t − r̄^LW)²` — squared **demeaned**
    returns (raw squared returns test `Var + mean²`, not volatility), then `newey_west_tstat(d)`
    (HAC) and `block_bootstrap_ci(d, block=3)` — a small block, not iid (`block=1`), since squared
    returns exhibit volatility clustering; the non-overlapping monthly holdings mute but don't
    remove the autocorrelation, so the CI matches the HAC treatment.
  - QLIKE comparison: Diebold–Mariano-style — `newey_west_tstat(qlike^est − qlike^LW)`.
  - Vol **ratio** `√(mean a²/mean b²)`: a small dedicated ratio-bootstrap (the existing CI helper
    cannot produce a ratio CI); otherwise report the variance-difference CI and don't call it a
    ratio. Same paired construction for graph-vs-rewire.
- **Power:** report an **MDE** for the vol comparison over the ~120 monthly OOS periods.
  `power.min_detectable_effect` is really "MDE for the mean of a HAC-tested serially-correlated
  series", so it transfers to the paired-difference series **only if** `ic_sd` and `phi` are
  **re-estimated from the actual `d_t` series** (its std and lag-1 autocorrelation), not left at
  the IC-calibrated defaults (0.08 / 0.4). Report it as an order-of-magnitude power check, not a
  precise threshold, with two stated caveats: its Gaussian-AR(1) model understates `d_t`'s fat
  tails (optimistic), and its internal `cap=0.2` effect ceiling can trip a spurious "underpowered"
  warning on the QLIKE-difference scale.
- **Diagnostics:** average condition number of Σ̂, effective N, turnover, and the count of periods
  dropped for insufficient data (informational, not the metric).

## Portfolio & numerical policy (`sample_cov`, `gmvp_weights`)

- **NaN policy:** estimate on **complete cases** within the trailing window (drop any asset with a
  missing return in the window, or use pairwise min-overlap), mirroring `graph._corr_matrix`'s
  complete-case guard — a single NaN column otherwise poisons `S` and `Σ⁻¹` fails. Report the
  count of assets/periods dropped.
- **Inversion:** `gmvp_weights(cov)` computes `w = solve(cov, 1)` (not explicit `inv`) after a
  **`np.linalg.cholesky` positive-definiteness check** — `solve` raises only on exact singularity,
  not on an *indefinite* matrix, which would silently return sign-flipped garbage weights. On a
  non-PD covariance it ridges by the **magnitude of the most-negative eigenvalue**
  (`cov + max(1e-10, −λ_min+1e-10)·I`) so the fix actually lifts the matrix to PD, not a cosmetic
  `ε·tr` floor. With the PSD-projected target (Estimator A) the compared covariances are already PD,
  so this is defense-in-depth, not part of the estimator.

## Regime conditioning (the paired second-tier add-on)

Classify each OOS period by a strictly-**trailing** (PIT) regime proxy — trailing average pairwise
return correlation (high = crisis / everything-moves-together). Hypothesis: the graph's covariance
benefit **concentrates in high-correlation regimes**, because structure matters most when
idiosyncratic diversification fails.

Test on a **scale-free** statistic — the **log-variance ratio** (or QLIKE difference), NOT the raw
variance *difference*: high-correlation regimes are high-vol regimes, so any covariance edge
produces a larger *absolute* variance gap there mechanically; a ratio/log removes that confound.
Compare the graph-vs-Ledoit–Wolf log-variance ratio across the **high/low median split** (2 groups,
~60 periods each — not terciles) with a HAC test on the spread (reuses the `conditioning.py`
low-minus-high pattern). The regime **threshold** is set on a trailing/expanding basis where
feasible; if a full-sample median is used for the split, that in-sample-threshold peek is stated
explicitly as a caveat. The MDE is reported at the overall n≈120, not per regime group.

## Module & interfaces

New module `src/marketgnn/risk.py`:
- `dense_adjacency(graph, nodes) -> np.ndarray` — aligned, symmetrized boolean adjacency (asserts
  node order == `nodes`).
- `sample_cov(returns) -> np.ndarray` — complete-case sample covariance.
- `ledoit_wolf_cc(returns) -> (cov, shrinkage)` — constant-correlation shrinkage target.
- `graph_masked_cov(returns, adj, *, shrink=None) -> np.ndarray` — estimator A.
- `_glasso_admm(S, penalty, *, rho=1.0, tol=1e-4, rel_tol=1e-2, max_iter=500) -> (Theta, Z, converged)`
  — the SPD-guarded ADMM solver, returning the PD precision `Theta`, the **sparse iterate `Z`**
  (carries the exact off-graph zeros), and a convergence flag. Exposed (not private-only) so test 5
  can assert on `Z`/`Theta` directly.
- `graph_glasso(returns, adj, *, edge_penalty, offedge_penalty, rho=1.0, tol=1e-4, max_iter=500)
  -> np.ndarray` — estimator B: standardizes to correlation space, builds the per-edge penalty
  matrix, calls `_glasso_admm`, and returns `Σ̂ = D · solve(Theta, I) · D`; falls back to A
  (`graph_masked_cov`) when `converged` is False.
- `gmvp_weights(cov) -> np.ndarray` — unconstrained GMVP via `solve` + ridge fallback.
- `portfolio_qlike(...)`, and the paired-difference helpers used for inference.
- `evaluate_estimators(prices, provider, *, window=252, rebal_freq="M", rewire_seeds=(0,1,2), ...)
  -> (pd.DataFrame, pd.DataFrame, dict)` — the manual rolling-rebalance loop; returns
  (estimator_table, regime_table, meta): QLIKE + realized-vol per estimator, paired significance vs
  Ledoit–Wolf, the multi-seed rewire null, condition-number/turnover diagnostics, and an MDE for
  the vol comparison.
- `regime_conditioning(...) -> pd.DataFrame` — trailing-correlation regime split + the scale-free
  (log-variance-ratio) spread test.
- `main()` — runs Run 10 on real data (co-holding headline + correlation ceiling + rewire null)
  and prints the estimator table and the regime table.

Reuses: `graph.py` (co-holding via `coholding.make_provider`, `correlation_knn`,
`degree_preserving_rewire`), `dataset.rebalance_dates` (the rolling loop), `evaluate` (`qlike`,
`newey_west_tstat`, `block_bootstrap_ci`), `power.min_detectable_effect`,
`data.download.load_market`, `data.universe.default_universe`. **Not** `PurgedWalkForward` (see
Evaluation protocol). No new dependencies (numpy-only ADMM glasso).

## Tests (`tests/test_risk.py`, TDD)

1. `test_gmvp_weights_analytic` — known 2–3-asset covariance; GMVP matches the closed form
   `Σ⁻¹1 / 1ᵀΣ⁻¹1`; weights sum to 1; singular input → ridge fallback returns finite weights.
2. `test_dense_adjacency_aligned_and_symmetric` — scatters `edge_index` correctly, symmetrizes a
   *directed* `correlation_knn` graph (`A == A.T`), and **raises** on a node-order mismatch.
3. `test_ledoit_wolf_shrinks_and_conditions` — shrinkage intensity ∈ [0,1]; Σ̂ better-conditioned
   than the sample covariance on a poorly-sampled draw.
4. `test_graph_masked_recovers_block_structure` — **positive control**: a synthetic block-factor
   market whose true covariance IS the block structure; the graph-masked estimator with the
   *true block graph* gives lower OOS GMVP realized vol than sample, and lower than a
   degree-preserving-rewired graph. Proves the method can exploit real structure.
5. `test_glasso_spd_and_recovers_sparsity` — call `_glasso_admm` directly: it returns an **SPD**
   `Theta` even on a rank-deficient/ill-conditioned `S` (the SPD safeguard), and with a high
   off-edge penalty the off-graph entries of the **sparse iterate `Z`** are exactly ~0 (the dense
   `Theta` matches `Z` only to `tol`) while on-graph structure is retained (planted sparse
   precision recovered). Separately, `graph_glasso` returns a finite PD `Σ̂` and, on a forced
   non-convergence (`max_iter=1`), falls back to `graph_masked_cov` without raising.
6. `test_estimator_is_pit` — Σ̂ at t is unchanged when future return rows are perturbed.
7. `test_qlike_and_paired_diff` — `portfolio_qlike` penalizes a worse variance forecast; the
   paired-difference construction (demeaned squared-return diff) has the expected sign on a
   synthetic where one estimator is truly lower-variance.
8. `test_regime_spread` — on synthetic data where structure helps only in a high-corr regime, the
   scale-free regime spread test recovers the concentration and stays null when it shouldn't.

Torch-free (pure numpy/pandas); the block-structure and glasso tests are the power/positive
controls, mirroring the planted-recovery discipline used everywhere else in the repo.

## Deliverable

- `risk.py` + `test_risk.py` (raises the test count).
- **Run 10** written up in `RESULTS.md` (estimator table, regime table, honest read) and
  summarized in `README.md` (a "How it works" bullet + a findings line), `paper/note.md`, and
  the roadmap; `REVIEW.md` unaffected.
- `python -m marketgnn.risk` prints the real-data tables.
- Personal-site `content.ts` market-gnn card updated once real numbers exist.

## Outcome (realized — 90 large-caps, 101 monthly rebalances, 2016-07…2024-11, 252d window)

Every estimator clusters near Ledoit–Wolf, and no graph structure significantly beats it:

- **Ledoit–Wolf is the benchmark** (annualized GMVP realized vol **14.9%**, QLIKE 1.55). Plain
  **sample** (16.5%, paired t **+6.06**) and **diagonal** (16.2%, t **+3.09**) are *significantly
  worse* — shrinkage genuinely helps.
- **Every graph estimator is statistically indistinguishable from LW** (all |t| < 1.96): the
  precision-space **graphical lasso** (14.6%, ratio **0.98**, t **−1.55**) marginally under it, the
  covariance-space **masked** estimators (co-holding 15.1%, t +1.03; correlation 15.1%, t +0.86)
  marginally over it, and all marginally *worse* on QLIKE (glasso 1.82, masked 1.92/2.03 vs LW 1.55).
- **The real graph edges its own rewire, but not significantly** — masked 15.1% / glasso 14.6% vs
  the degree-preserving rewire null at 15.4%. So topology carries a *little* information; the margin
  is inside noise.
- **Regime:** glasso's tiny edge over LW is ~uniform across correlation regimes (log-var ratio
  −0.038 high-corr vs −0.034 low-corr, spread −0.005) — no meaningful crisis concentration.

**Net:** on liquid large-caps a graph-structured covariance adds no *significant* out-of-sample
improvement over standard constant-correlation shrinkage, whether the graph enters as a shrinkage
target or as precision-matrix sparsity — the null extends from alpha to risk. Bracketed by the
block-structure **positive control** (the masked estimator DOES beat sample and the mean rewire when
the graph genuinely is the covariance structure — proving power) and the **rewire null** (the real
graph edges random topology). Glasso penalties are fixed a-priori (not OOS-tuned).

*Correction history: an earlier draft reported estimator A at 8–12× LW's vol (a "50×+ injection-
method" headline). The pre-merge review team found the masked target was **indefinite** (zeroing
off-graph entries breaks PSD), so the unconstrained GMVP was optimizing over a non-covariance and
returning leverage artifacts. PSD-projecting the target (see Estimator A) corrected it to the clean
~1.02× cluster above; the qualitative conclusion — no significant graph improvement — is unchanged
and now rests on valid covariances.*

## Out of scope (YAGNI)

Trading costs / net Sharpe (covariance-quality study only), factor-model covariances, the
small-cap universe (a separate spec), supplier–customer graphs, and any return-prediction change.
