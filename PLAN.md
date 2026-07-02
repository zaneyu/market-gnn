# market-gnn — project plan (v2, post-review)

GNN-forward framing: this is a GNN modeling project with an unusually rigorous,
leak-controlled evaluation. The hygiene harness and the random-graph control are
surfaced prominently because they are what make the model results trustworthy.

## Research question (falsifiable)
> Does modelling stocks as a graph (trailing-correlation + sector edges) and passing
> node features through a GNN produce higher **out-of-sample cross-sectional rank-IC**
> than an **identical feature set fed to a matched non-graph model** — after
> point-in-time construction and purged walk-forward evaluation?

The claim is "graph structure adds *incremental* signal over a matched non-graph
baseline," not "I predict returns." A null (esp. for returns) is an acceptable result.

## Hypotheses
- **H1 (PRIMARY, pre-registered):** GNN rank-IC > matched-MLP rank-IC on the **returns**
  target. This is the *one* primary endpoint; everything else is secondary + FDR-controlled.
- **H2 (control):** GNN on the real graph > GNN on a **degree-sequence-preserving** random
  graph. If H2 fails, H1 is an artifact of parameters/degree, not structure.
- **H3 (anchor):** all models beat naive baselines on the **volatility** target — a sanity
  gate, not a finding.
- **H4 (secondary):** sector vs correlation vs both edges — reported with FDR.

## Decisions from review
- **Full-rigor data:** point-in-time index membership + delisting returns; **weekly**
  cadence; **Russell 1000** universe; **SPY total-return** as the beta benchmark (never a
  survivor equal-weight mean). Power calc up front (min detectable IC gap).
- **GNN-forward framing** (chosen), with hygiene results above the README fold anyway.

## Accepted fixes (from the four-reviewer panel)
**Statistics.** Per-date ICs are autocorrelated (overlapping labels) → i.i.d. t-stat and
bootstrap overstate significance. Use **Newey–West/HAC** SEs (lag ≥ horizon) and **block
bootstrap**; report naive alongside to show the inflation. Pre-register H1; **Benjamini–
Hochberg FDR** across the ablation grid. Add a **power calculation**.

**Survivorship / labels.** SPY-benchmarked beta (kills the survivor-mean leak). **Delisting
returns** filled (−100% bankruptcy / deal price acquisition) so the label window is complete
and the bottom decile isn't under-punished. PIT membership so historical universes are real.

**Model fairness (controlled A/B).** GNN ≡ MLP **+ graph-conv layers**, sharing the *exact*
head, loss, training loop, and purged val early-stopping; zeroing the graph must recover the
MLP. Use a **ranking loss** (soft-Spearman / pairwise) since the metric is rank-IC. Ridge/GBM
are the "any-other-method" bar and may differ.

**Lead-lag channel.** A static GNN cannot express neighbor-return→own-return spillover, so a
returns null would be uninterpretable. Fix: add a **PIT-safe neighbor-aggregated trailing-
return feature** both models consume, and scope the MVP returns test as *contemporaneous
relative-strength*; the temporal spillover channel is **deferred, not tested**.

**Graph.** Keep directed top-k selection, but the model pipeline applies **symmetrize +
self-loops**; edge **weight = |corr|**, **sign = edge attribute** (never a signed attention
logit). Degree-cap the sector graph. Primary model **GraphSAGE-mean**; GATv2 as one ablation.
The random null is a **degree-preserving rewire**, not average-degree matched.

**Engineering.** `pyproject.toml` (src layout, `pythonpath=src`, torch-free test path via
extras). `forward_return` uses `get_indexer` (dup/absent-date safe). `mom_12_1 = P_{t-21}/
P_{t-252}−1`. Deterministic `argsort` (stable + node-index tiebreak). `groupby(...,
include_groups=False)`. Degenerate cross-sections → NaN not silent 0. Strengthen the split
tests (rolling bound + purge/embargo boundary). Add the missing normalization-scope, label-
shuffle, canary, and evaluate tests.

## Data
- Prices/volume: yfinance (Stooq fallback), daily OHLCV, parquet-cached; resample to weekly.
- Universe: **Russell 1000** with **point-in-time membership** + delisting handling.
- Beta benchmark: **SPY** total return.
- MVP features price-derived only (PIT fundamentals deferred).

## Targets
- **Volatility** (next-week realized vol, log) — predictable anchor (H3).
- **Excess-return rank** (next-week) — the hard, honest-null-friendly target (H1).

## Node features (PIT; cross-sectionally normalized per date)
mom_1m, mom_3m, mom_12_1, rev_1w, vol_20d, turnover, beta(vs SPY), size, **nbr_ret**
(neighbor-aggregated trailing return — the lead-lag channel). Missing-history handled with an
indicator, not silent median-fill.

## Evaluation protocol
Purged, embargoed walk-forward (embargo ≥ max(feature_window, corr_window) in steps).
Per-date rank-IC → **HAC** t-stat + IC-IR, decile long-short spread, turnover; vol: QLIKE vs
naive. **Block-bootstrap** CIs, multi-seed variance, **BH-FDR** across the grid, power calc.

## Leakage / hygiene harness (pytest)
1. Label-shuffle → IC ~0. 2. Future-injection canary → IC spikes (detector has teeth).
3. Graph PIT purity → byte-identical under future absent/corrupted. 4. Purge/embargo
correctness → no train feature/label window touches test. 5. Normalization scope → per-date
only, shift/scale-invariant. 6. Neighbor-feature PIT purity.

## Repo layout
```
market-gnn/ README.md PLAN.md pyproject.toml
  configs/default.yaml
  data/ download.py universe.py            # PIT membership + delistings
  src/marketgnn/ splits.py graph.py features.py evaluate.py dataset.py train.py power.py
                 models/ ridge.py gbm.py mlp.py gnn.py temporal.py(stub)
  tests/  (hygiene tests 1–6 + evaluate/stats)
  paper/note.md                            # incl. "the result I almost believed"
  .github/workflows/ci.yml
```

## Status
- Rebuilt to pass CI: `splits.py`, `graph.py`, `features.py`, `evaluate.py` + tests.
- TODO: `dataset.py`, `power.py`, models, `train.py`, PIT `universe.py`/`download.py`,
  configs, README, note, CI.

## Scope (full-rigor track, ~4–6 weekends)
PIT membership + delistings, weekly Russell 1000, SPY beta, corr+sector graph + degree-
preserving null, static GraphSAGE ≡ MLP+conv with ranking loss, ridge/GBM baselines, purged
WF, HAC+block-bootstrap+FDR+power, hygiene tests 1–6, one ablation table, both targets.
Temporal GNN: two-line footnote, not built.

## Expected honest outcome
Graph adds small-but-real IC for **volatility**, marginal-to-none for **returns** after
controls; **sector edges** > correlation edges; **GBM is a tough baseline**. Reporting that
cleanly — with the controls that dissolve any naive positive — is the win.
