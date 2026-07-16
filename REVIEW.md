# Adversarial review & corrections

This repo argues that most "graph alpha" is an artifact you can only catch by trying hard to
break your own result. So it gets held to the same standard: below is a round of **adversarial
review** (four independent hostile passes — statistical rigor, 13F data integrity, code
correctness, and claims-vs-evidence) and every fix it produced. Findings were verified against
the code/data before acting; a few reviewer claims were **rejected on verification** and are
noted as such.

## Fixed — data & correctness

- **CUSIP map silently isolated 5 names.** `BAC`, `SCHW`, `DIS`, `AMT`, `SPG` carried
  bond / depositary-share / placeholder / lowercase-typo identifiers, so the exact `isin` match
  found **0–6 filing rows** for each and those names became **isolated nodes** in the co-holding
  graph. Corrected to the equity common-stock CUSIPs (now 1,455–4,731 holder rows each). Guarded
  by a new **CUSIP check-digit test** (CI) and a **data-gated full-coverage test** (all universe
  names must have ≥1 holder in the latest snapshot).
  *Reviewer also flagged `RTX` → `75513E101`; verification showed 13F filers still report
  Raytheon common under the legacy `755111507` (`75513E101` has **0** rows), so the original was
  correct and was kept. Verify before trusting.*
- **Silent synthetic fallback under a "REAL prices" header.** `load_market(synthetic=False)` caught
  every exception and returned a 60-name synthetic panel with only a one-line notice. It now
  **raises** by default (opt-in `allow_synthetic_fallback` emits a loud banner and matches the
  requested universe).
- **Doubled mutual edges.** `edges_from_pairs(symmetric=True)` re-added a reverse for every row,
  so a mutual kNN pair got 4 edges (double weight, inflated `avg_degree`). Now de-duplicated
  (max weight per directed pair). New regression test.
- **Temporal graph node-ordering was assumed, not enforced.** The GConvGRU indexed provider-graph
  edges positionally against `prices.columns`; a divergent order would silently scramble every
  edge. Added `_align` to remap (or no-op fast-path) the graph onto the fixed universe.

## Fixed — PIT correctness

- **13F amendments and late filers leaked in.** The loader used `startswith("13F-HR")` (matched
  `/A` amendments) and never checked filing date. Now an **exact `13F-HR`** match plus a
  **`FILING_DATE <= public`** filter — only filings actually public at the as-of date.
- **`pick_asof` returned a not-yet-public graph** before the first snapshot's public date. Now
  returns an **empty graph** at that boundary.

## Fixed — statistics

- **The pre-registered primary endpoint (H1) was never computed.** The study claimed a
  GNN-vs-matched-MLP comparison and `power.py` was powered for the *IC-difference* series, but the
  code only tested each cell's IC-vs-zero. Added `train.primary_endpoint`: the **paired per-date
  IC-difference** (GNN − MLP) with a HAC t, block-bootstrap CI, and MDE, reported outside the FDR
  family. (Synthetic: ΔIC +0.005, t 0.33, p 0.75.) Honest framing (tightened by a later pass): H1
  isolates the incremental value of *message-passing* over an already-graph-informed MLP (both see
  the neighbour feature), and its MDE (~0.047) sits above a plausible message-passing edge
  (~0.005–0.02), so H1 is **underpowered for a small edge** — consistent with, but not ruling out,
  one. The load-bearing power comes from the planted-recovery runs, not H1.
- **BH-FDR pooled heterogeneous endpoints (ret + vol) and null controls into one family.** Now
  applied **within each target family** and over the **discovery family only** (excluding the
  rewire/none/own-mom controls, which are diagnostics). *Honest correction to the first draft of
  this note:* excluding the controls is the discovery-family argument, **not** a way to remove a
  "discovery bias" — pooling placebos would only enlarge `m` and make the real endpoints *harder*
  to call significant (more conservative). It flips no conclusion here (every real endpoint has
  p ≫ 0.05); a later adversarial pass caught the inverted reasoning and it is corrected here.
- **MDE was mis-scaled.** `power.py` fed the *marginal* IC sd as the AR(1) *innovation* sd
  (over-dispersing the series by up to 2.3×) and used a larger HAC lag than the real test. Now
  scales the innovation to `sd·√(1−φ²)`, starts from the stationary distribution, and uses the
  same Newey–West lag as evaluation. (MDEs shifted modestly, e.g. co-holding real 0.023 → 0.025.)
- **Purge could decouple from the label horizon.** Train now derives the minimum purge (in
  rebalance steps) from `label_horizon` and the actual rebalance spacing and never trusts a
  smaller knob; the temporal runner's train/test gap is likewise derived from real date spacing,
  not a hardcoded weekly cadence.

## Fixed — honesty / claims

- **"~4,370 institutions" was prose no code emitted.** The runner now prints the real per-snapshot
  range (**~2,500–5,800**, median ~3,600) and universe coverage; docs cite the range.
- **"byte-identical" reproducibility was unachievable** — Yahoo retroactively re-adjusts historical
  closes. Docs and `snapshot.py` now state real-data figures reproduce to **~2 sig figs**, and that
  the **seeded synthetic recoveries** carry the load-bearing power claims.
- **The reversal positive control is survivorship-exposed** (a positive return claim on current
  membership). Now labelled as a harness control, not a tradable finding.
- Corrected downstream numbers after the CUSIP/graph fixes: co-holding real IC +0.010 → **+0.012**
  (still a powered null), avg degree 18.6 → **16.9**, temporal real graph +0.013 → **+0.006** (≈
  its no-graph twin). The qualitative conclusion — **the graph adds no return signal that
  survives** — is unchanged and, if anything, slightly stronger.
- Synced the stale `paper/note.md` (was "45 tests", Runs 1–5 only) to the full nine-run state;
  test count **59 → 62**.

## Rejected on verification

- *"RTX CUSIP is wrong"* — kept `755111507` (what filers actually use; see above).
- *"The temporal blindfold isn't blind (no-graph baseline is significant)"* — true only at the
  smaller **test** config (n=40, 60 epochs), where own-momentum proxies the block spillover; at the
  default run (n=60, 300 epochs) the blindfold is null (−0.012). The test's docstring/assertions
  were corrected to claim only what holds in both configs (the graph adds **incremental** IC).
