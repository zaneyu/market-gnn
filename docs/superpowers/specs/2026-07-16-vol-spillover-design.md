# Run 11 — volatility spillover over the co-holding graph (Diebold–Yilmaz-inspired)

**Status:** draft design (2026-07-16). Single-module extension to market-gnn.

## Motivation

Runs 1–10 close the *return* channels (contemporaneous, lead-lag, learned temporal) and the
*covariance-level* risk channel (Run 10). One risk channel remains untested: **volatility
transmission** — does a shock to a neighbour's volatility propagate to a stock's *future*
volatility beyond what the stock's own vol history already predicts? This is the
cross-sectional core of the Diebold–Yilmaz "connectedness" question. Crucially, Run 3 already
showed that ~90% of vol "predictability" is **own persistence** — so the own-vol control is
load-bearing, not optional: a naive neighbour-vol signal will look predictive simply because
neighbours' vol proxies one's own.

## Scientific question

> At date t, does the graph-weighted mean of **neighbours'** trailing realized volatility
> predict a stock's **forward** realized volatility, *after controlling for the stock's own
> trailing volatility*? Cross-sectionally, per date, out-of-sample, PIT.

Honest scope note: full Diebold–Yilmaz connectedness is a VAR forecast-error-variance
decomposition. This experiment is its **zero-parameter cross-sectional analogue** (neighbour
shock → own future vol), deliberately matching the repo's lead-lag design so the answer is
comparable — it is *inspired by* DY, not a VAR FEVD, and is labelled as such everywhere.

## Signal / target / control (innovation-based — the level-confound fix)

A naive "neighbour trailing vol" signal is **structurally confounded**: co-held stocks share
persistent vol *levels* (size/style clustering), and the neighbour mean is a *less noisy* proxy
of that shared level than the stock's own short-window vol (errors-in-variables), so it predicts
the own-vol residual with **zero transmission** — verified by simulation (γ=0 clustered market:
naive residualized IC +0.36; the rewire null does NOT catch this, because rewiring destroys the
clustering, so "beats rewire" holds under the pure confound). The design therefore measures
**innovations**, not levels:

- **Node state at t:** short trailing vol `σ20_i(t) = std(returns over (t−20, t])` and long
  trailing vol `σ250_i(t) = std(returns over (t−250, t])` — both strictly ≤ t.
- **Signal (spillover):** `spill_i(t) = Σ_j w_ij (σ20_j(t) − σ250_j(t)) / Σ_j w_ij` — the
  edge-weighted mean of neighbours' **vol innovations** (how elevated each neighbour's recent
  vol is vs its own long-run level). Innovations are level-free, so shared levels cancel.
- **Target:** forward realized vol `σ_i(t→t+h) = std(returns over (t, t+h])`.
- **Controls (two regressors):** own `σ20_i(t)` and own `σ250_i(t)`. The residualized reading
  regresses the forward vol per date on both own-vol terms (a small multi-regressor
  cross-sectional residualizer in `volspill.py`; `leadlag._residualize` is single-regressor)
  and computes the rank-IC of the spillover signal against that residual. No "market mean
  innovation" regressor is added: a per-date cross-sectional constant is **absorbed by the
  demeaned per-date regression by construction** (it would be vacuous), which also means
  *homogeneously-loaded* common vol shocks are automatically controlled; what cannot be
  controlled this way is heterogeneous factor **exposure** — the stated identification limit
  below. Verified by simulation: pure level-confound → **−0.003** (clean null); a true planted
  γ=0.06 spillover → **+0.065**, with or without clustering — the innovation design separates
  transmission from *static* similarity (dynamic factor exposure is the residual limit).
- Per-date **rank**-IC removes any *homogeneous* common vol-regime shift or scale (a common
  monotone transform preserves cross-sectional ranks). It does NOT remove heterogeneous
  persistent vol levels — that is exactly the clustered-level channel above, which the
  innovation signal + two-regressor control are there to kill.
- Two readings reported, as in `leadlag`: `raw` (the innovation signal vs forward vol with no
  own-vol control — the persistence pathway included, a yardstick) and `resid` (the endpoint).

## The null and the power proof (repo standard)

- **Rewire null:** degree-preserving rewire of the co-holding graph, averaged over seeds
  (0,1,2). The real graph must beat its rewire or "spillover" is just diffuse cross-sectional
  vol similarity, not topology.
- **Planted recovery (power):** a synthetic market with a **spatial ARCH** effect planted along
  a known graph: each name's conditional variance responds to its *neighbours'* lagged squared
  shocks over and above its own GARCH-style persistence —
  `σ²_i(t) = ω_i + α·r²_i(t−1) + β·σ²_i(t−1) + γ·mean_{j∈N(i)} r²_j(t−1)` with γ > 0 planted.
  **Stationarity:** with a row-normalized neighbour mean the aggregate persistence is α+β+γ, so
  the guard is `α+β+γ < 1`, asserted inside the generator. Defaults `α=0.08, β=0.5, γ=0.35`
  (sum 0.93 — an earlier draft's β=0.85 made the process explosive, σ²→10¹⁵⁰). The γ-term is 0
  for a node with no neighbours (singleton blocks under `sector_graph` are edgeless).
  The pipeline must recover the planted spillover in the *residualized* reading (the own-vol
  controls must NOT absorb it, since the plant is genuinely cross-sectional), and the rewire
  null must stay flat. This proves the residualized estimand has power before any real-data
  null is interpreted.

## Evaluation protocol

- **Universe/data:** the 90 large-caps (cached parquet), co-holding graph via
  `coholding.make_provider` with `nodes = list(prices.columns)` (PIT public-date logic as-is).
- **Sampling:** non-overlapping — `for i in range(warmup, len(idx) − h, h)` over trading-day
  positions (the `conditioning.py`/`signals.py` pattern), `lookback = 20`,
  `level_lookback = 250`, `h = 20` trading days, `warmup = 260` (covers the 250d level window).
  Non-overlapping windows keep the per-date IC series clean for HAC. On the cached panel
  (2,767 trading days, 2014–2024) this yields **n = 125 evaluation dates** — pre-registered so
  the MDE claim is checkable (expected MDE ~0.08–0.10 at ic_sd≈0.2, φ≈0.5).
- **Statistics:** per-date rank-IC (`evaluate.per_date_ic`) → `ic_summary` (Newey–West/HAC),
  `two_sided_p`; **MDE** via `power.min_detectable_effect` with `ic_sd`/`phi` re-estimated from
  the actual IC series (the Run-2 correction); BH-FDR over the **discovery family only**
  (raw + resid on the real graph); rewire rows and the own-vol control row are diagnostics
  outside the family (`fdr_sig=False, q=NaN`), the established convention.
- **Rows reported:** coholding/raw, coholding/resid (the endpoint), rewire/raw, rewire/resid,
  and `(none)/own_vol` — own **σ20** (the short trailing vol) → forward vol, the persistence
  yardstick from Run 3.

## Identification limit (stated up front, not discovered later)

A **dynamic common-factor confound survives the innovation design, and the rewire null cannot
catch it** — verified by simulation: with ZERO transmission but a persistent per-block vol
factor (log-AR(1), φ=0.97 — sector vol regimes, which certainly exist among these 90 names),
the residualized innovation IC is **+0.10 to +0.22** (as large as the γ=0.35 plant) while the
rewire null stays flat. Mechanism: errors-in-variables one level up — the neighbour mean of
innovations is a less noisy proxy of the shared *factor innovation* than the own term, so it
predicts the residual with no propagation; rewiring destroys block membership, so the null is
blind to it. *Homogeneously-loaded* common vol shocks are absorbed automatically by the
per-date cross-sectional design (demeaning removes any per-date constant), but heterogeneous
factor **exposure** — block-level vol factors, or a global factor with graph-clustered
loadings — remains **observationally equivalent** to transmission in this design (as it is in
standard Diebold–Yilmaz connectedness without identification assumptions).

## Pre-registered reading of outcomes

- Residualized co-holding IC significant, above MDE, rewire flat → **neighbour vol innovations
  carry incremental predictive information about own future vol along the real ownership
  topology** — consistent with EITHER directed transmission OR shared factor-vol exposure that
  co-holding proxies; the design cannot separate these, and the write-up must not claim
  "causal spillover".
- Residualized IC null with adequate power (MDE reported) → the vol channel joins the other
  nulls: neighbour vol adds nothing beyond own persistence and the common regime — a *stronger*
  statement than the positive, since the confound biases *toward* finding signal, making a
  null conservative. Either outcome is reported with the same machinery.

## Module & interfaces

New module `src/marketgnn/volspill.py` (torch-free, numpy/pandas only):
- `trailing_vol(returns, *, lookback) -> pd.DataFrame` — rolling std with
  `min_periods = lookback` (a short history yields NaN, never a 2-observation "vol").
- `neighbour_innovation(innov_row: pd.Series, graph) -> pd.Series` — edge-weighted mean of
  neighbours' vol innovations, aligned to `graph.nodes`. **NaN policy:** non-finite neighbours
  are excluded and the weights renormalized over the finite ones; NaN only when NO finite
  neighbour remains (isolated node, or all neighbours missing). This matters on the real panel:
  one cached name has no prices before 2015-07, and the naive `leadlag_signal`-style
  accumulation would silently NaN-poison its ~10 graph neighbours for ~6 months. (Co-holding
  cosine weights are ≥ 0 by construction, so no |·| is needed.)
- `_residualize_multi(y, X) -> np.ndarray` — cross-sectional OLS residual of y on multiple
  regressors (demeaned, least-squares); `leadlag._residualize` is single-regressor and stays
  untouched.
- `run_volspill(prices, *, graph_provider, lookback=20, level_lookback=250, horizon=20,
  warmup=260, rewire_seeds=(0,1,2)) -> pd.DataFrame` — the experiment table (rows above:
  mean_ic, hac_t, p, mde_80, n_dates, fdr_sig, q), MDE with `ic_sd`/`phi` re-estimated from
  each IC series.
- `make_synthetic_spatial_arch(*, n_assets=60, n_blocks=6, n_days=1500, alpha=0.08, beta=0.5,
  gamma=0.35, block_omega_spread=0.0, seed=0) -> (prices, graph)` — the planted spatial-ARCH
  market on a block graph (`graph.sector_graph` over synthetic blocks); asserts
  `alpha + beta + gamma < 1`; `block_omega_spread > 0` draws per-BLOCK base variances
  `ω_b = ω̄ · exp(N(0, spread²))` (log-normal, shared within a block; test 4 uses spread=1.0)
  to create the clustered-level confound for test 4; γ-term is 0 for edgeless nodes.
- `main()` — `--synthetic-planted` (power proof) and the real-data run (co-holding + rewire).

Reuses: `evaluate.per_date_ic/ic_summary/two_sided_p/benjamini_hochberg`,
`power.min_detectable_effect`, `leadlag._estimate_phi`, `coholding.make_provider`,
`graph.degree_preserving_rewire`, `data.download.load_market` (raises rather than substituting
synthetic when real data is unavailable — no silent fallback).

## Tests (`tests/test_volspill.py`, TDD)

1. `test_trailing_vol_is_pit` — perturbing rows after t leaves `trailing_vol` at t unchanged
   (window strictly `(t−lookback, t]`); short history (< lookback obs) yields NaN.
2. `test_neighbour_innovation_excludes_self_and_handles_nan` — a node's own innovation never
   enters its signal; weights applied; a NaN neighbour is **excluded with weights renormalized**
   (the remaining finite neighbours still produce a value); isolated node (or all-NaN
   neighbours) → NaN (dropped from IC, not zero-filled).
3. `test_planted_spatial_arch_is_recovered` — the positive control: on the planted market
   (stationary params α=0.08, β=0.5, γ=0.35) the **residualized** IC over the true graph is
   significant (thresholds calibrated against the corrected, non-explosive process with
   comfortable margin), and the multi-seed rewire residualized IC is well below it — asserting
   the *contrast*, not just "signal exists" (the blindfold lesson from Run 9's test).
4. `test_level_confound_is_killed_by_innovation_design` — the novel guard, validating the
   *control*, not the signal: on a **γ=0** market with per-block base variances
   (`block_omega_spread > 0` — neighbours share persistent vol levels, the exact real-world
   confound), the *level-based* naive signal residualized on own σ20 only shows a large spurious
   IC (~+0.36 in the design simulation — asserting the confound is real and present), while the
   **innovation signal with the two-regressor control is ~null**. Without this test, a broken
   residualization or a level-leaking signal would pass test 3 (the plant is strong) and
   silently declare false "spillover" on real data. (A homogeneous-ω γ=0 market exhibits NO
   confound — neighbour vol is pure noise there — which is why the clustered-ω construction is
   load-bearing; an earlier draft of this test would have failed for that reason.)
5. `test_fdr_family_excludes_controls` — rewire and own_vol rows have `fdr_sig=False`, `q` NaN.

## Deliverable

`volspill.py` + `test_volspill.py`; **Run 11** in RESULTS.md (planted table + real table +
honest read), README ("How it works" bullet + findings/roadmap updates), paper/note.md bullet;
`python -m marketgnn.volspill` prints both tables; personal-site card updated only after real
numbers exist and validation passes.

## Out of scope (YAGNI)

Full Diebold–Yilmaz VAR/FEVD connectedness tables; realized-kernel/intraday vol estimators;
options-implied vol; multi-horizon grids; any return-channel change; the correlation-kNN graph
(near-circular for a vol question — vol similarity drives correlation clustering; co-holding
only, with its rewire).
