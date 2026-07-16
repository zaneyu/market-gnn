# market-gnn

**A graph neural network for cross-sectional equity signals — and a leak-controlled
test of whether the graph actually helps.**

Stocks co-move in groups (sector, supply chain, shared factors). This project models
that structure as a graph, runs a GNN over it, and asks one clean, falsifiable
question:

> Does graph structure add **incremental out-of-sample rank-IC** over an **identical
> feature set fed to a matched non-graph model**, after point-in-time construction and
> purged walk-forward evaluation?

## Findings (real data: 90 large-caps, 2014–2024, weekly)

Tested every configuration by which a graph could plausibly add cross-sectional **return**
signal, and decomposed where the apparent "graph alpha" actually comes from:

1. **Contemporaneous graph is a red herring.** Frozen ≈ dynamic ≈ *no graph* for a linear
   model; the GNN ties the matched MLP (and is slightly worse — over-smoothing). Same-date
   message passing can't carry return signal, and doesn't.
2. **Lead-lag / spillover is a *powered* null on large-caps** — even over a *real* economic
   link. Neighbours' past returns → own future return (the channel Cohen–Frazzini /
   industry-momentum predicts). We first **plant** the effect and show the pipeline
   **recovers it** (IC 0.089, HAC t 9.6, rewire-null clean) — proving power — then on real
   data find ~0.006 with **80% power to detect ~0.03**. And it is not a proxy artefact: a
   graph of genuine **13F institutional co-holding** (Anton–Polk "Connected Stocks", built
   from the actual SEC filings, PIT) is *also* a powered null (IC +0.012, HAC t 1.4, 80% power
   to detect ~0.025, rewire-clean) — the warmest linkage tested, still not significant. Absent,
   not undetectable; consistent with the effect living in small/illiquid names this universe
   excludes.
3. **The vol "predictability" is ~90% persistence** — a naive trailing-vol forecast gets 0.435
   of the model's 0.479, and the model is *worse* on QLIKE (level calibration).
4. **Survivorship inflates it:** point-in-time membership correction shrinks the already-null
   return IC.
5. **What DOES carry signal (positive control):** the same harness finds **short-term (1-day)
   reversal strongly significant on real data** — IC +0.015, HAC t 3.4, p<0.001, FDR-sig. So
   the graph nulls are *real absences*, bracketed by positive controls on both real data
   (reversal) and synthetic data (planted lead-lag) — not a pipeline that can't find anything.
   (Gross rank-IC; reversal is high-turnover and cost-fragile — reported as a statistical
   control, not a tradable strategy.) **Survivorship caveat:** unlike the graph *nulls* (where
   survivorship bias runs *toward* finding signal, so a null is conservative), this reversal
   result is a *positive* return claim run on a current-membership universe — survivorship-exposed.
   Treat it as a suggestive control that the harness can find a real effect, not a clean tradable
   finding; a fully PIT reversal run is future work (needs delisted-name prices).

**Net:** the graph adds no return signal that survives leak-free, powered, control-checked
evaluation; the return signal that *is* present (short-horizon reversal) is non-graph and
cost-fragile — and this repo is the harness that separates a real effect from the survivorship /
overlap / over-smoothing artifacts that make graph "alpha" look real.
Full decomposition and numbers in [RESULTS.md](RESULTS.md).

The reversal effect is real gross but dies at ~0.9 bp of cost, and (on a 194-name universe) is
significant across liquidity terciles without a significant illiquidity gradient:

<p align="center">
  <img src="figures/reversal_cost_decay.png" width="46%" alt="Reversal Sharpe vs transaction cost">
  &nbsp;
  <img src="figures/reversal_by_liquidity.png" width="46%" alt="Reversal rank-IC by liquidity tercile">
</p>

## Why trust the null? The harness has teeth.

A null is only worth reporting if the pipeline could have found signal. Two guarantees:
- **It detects signal when present** — the planted lead-lag recovery above, and a
  **future-injection canary** (feed a feature = tomorrow's return → IC spikes to ~1.0).
- **It rejects spurious signal** — **label-shuffle** collapses IC to ~0; the **degree-preserving
  rewire** null stays flat; **purged walk-forward** removes label-overlap leakage; point-in-time
  purity of graph/features/labels is asserted, not assumed.

This repo is held to its own standard: [REVIEW.md](REVIEW.md) logs an adversarial review round
(four hostile passes) and every bug it found and fixed — a CUSIP map that silently isolated 5
names, a silent synthetic-data fallback, an FDR that pooled null controls, and a primary endpoint
that was never actually computed — plus the reviewer claims that were rejected on verification.

```bash
pytest -q     # 77 tests (3 GNN/temporal skip without torch, 1 skips without the fetched 13F data)
```

## Quickstart

```bash
pip install -e ".[dev]"          # core + tests (no torch needed)
pytest -q                        # 77 tests (73 in torch-free CI), ~40s
python -m marketgnn.train --synthetic          # offline factor-market demo
```

Example output (synthetic market — shape of the real ablation):

```
      graph model target mean_ic   hac_t naive_t   ci_lo  ci_hi  n_dates  fdr_sig
correlation ridge    ret  -0.007  -0.321  -0.269  -0.053 +0.036       52    False
correlation ridge    vol  +0.172  +9.828  +9.529  +0.139 +0.207       52     True
       none ridge    vol  +0.177 +10.573  +9.907  +0.147 +0.213       52     True
```
Returns → honest null; volatility → real, FDR-significant signal. Note `hac_t < naive_t`
on every row: the Newey–West correction deflating the autocorrelation-inflated t-stat.

To train the GNN vs the matched MLP (the primary H1 test): `pip install -e ".[gnn]"` then
`python -m marketgnn.train --synthetic --models mlp,gnn`.

## How it works

- **Graph** (`graph.py`) — per date, a point-in-time graph: trailing-correlation kNN
  and/or sector edges. Fair null for "does topology matter" is a **degree-sequence-
  preserving rewire** (`rewire`), not just an average-degree random graph.
- **Features** (`features.py`) — momentum / reversal / vol / turnover / beta (vs **SPY**,
  never a survivor mean) / size, plus a **neighbour-return** feature that both the GNN and
  the MLP consume — so the GNN must beat an MLP that already has the graph-derived feature.
  Normalized **cross-sectionally per date**.
- **Models** (`models/`) — ridge and LightGBM baselines; the GNN and MLP are the **same
  network** toggling whether real edges are visible (zero the graph → recover the MLP),
  sharing head, a differentiable **ranking loss**, and training loop, so the A/B is clean.
- **Primary endpoint** (`train.primary_endpoint`) — the pre-registered H1 is tested as an
  actual **paired** comparison: the per-date IC-*difference* series (GNN minus MLP, same dates
  and universe) with a HAC t-stat, block-bootstrap CI, and an MDE — the exact estimand
  `power.py` is built for, reported *outside* the FDR family, not eyeballed from two separate
  IC-vs-zero rows. H1 isolates the incremental value of **message-passing** over an already
  graph-informed MLP (both see the neighbour-return feature). (Synthetic: ΔIC +0.005, t 0.33,
  p 0.75; MDE ~0.047 — above a plausible message-passing edge of ~0.005–0.02, so this endpoint
  is *underpowered for a small edge*: consistent with, not ruling out, one. The broader "graph
  adds signal" claim rests on the zero-parameter lead-lag test and the planted-recovery power.)
- **Evaluation** (`evaluate.py`) — per-date rank-IC aggregated with **Newey–West/HAC**
  t-stats and a **block bootstrap** (labels are autocorrelated; i.i.d. inference overstates
  significance). One pre-registered primary endpoint (H1); the grid is **BH-FDR** controlled.
- **Splits** (`splits.py`) — **purged, embargoed walk-forward**: training labels whose
  horizon overlaps the test window are removed.
- **Power** (`power.py`) — minimum detectable IC gap given cadence and autocorrelation, so
  a null is distinguishable from "underpowered."
- **Lead-lag** (`leadlag.py`) — the strictly-lagged neighbour-momentum signal (neighbours'
  past → own future), a **planted-signal recovery** proving the pipeline detects spillover,
  and the own-momentum control. This is the experiment that makes the null a *result*.
- **Temporal GNN** (`models/temporal.py`) — a **GConvGRU** (graph-conv spatial layer → GRU
  over dates): the *learned* counterpart to the zero-parameter lead-lag signal, with the same
  `use_graph` blindfold A/B. It recovers a planted lead-lag (IC +0.067, t 3.9; blindfold −0.012,
  n.s.) then nulls on real data (graph +0.006 vs no-graph +0.007, both n.s. — the graph if
  anything hurts) — the null isn't a modelling limitation.
- **Robustness** (`robustness.py`) — vol model vs the naive random-walk forecast (rank-IC and
  QLIKE), showing how much of the vol IC is mere persistence.
- **Signals** (`signals.py`) — real-data positive controls: textbook anomalies (short-term
  reversal, 12-1 momentum) run through the same harness, so a graph null is provably a real
  absence rather than a pipeline that finds nothing.
- **Costs** (`costs.py`) — the decile long-short a signal implies, net of transaction costs:
  breakeven bps and net Sharpe. Turns "cost-fragile" into a number (reversal breaks even at
  ~0.9 bp → arbitraged net). A gross IC without this is how backtests lie.
- **Conditioning** (`conditioning.py`) — liquidity-tercile analysis: is reversal an
  illiquidity effect? A HAC test on the low-minus-high IC spread, over a widened ~180-name
  universe (`data/universe.extended_universe`) for real liquidity range and power.
- **Figures** (`figures.py`) — the reversal-by-liquidity gradient and the cost-decay curve
  as PNGs (`figures/`), because visual evidence lands harder than tables.
- **Real economic-link graph** (`coholding.py`) — not a proxy: an **institutional co-holding**
  graph (Anton–Polk "Connected Stocks") built from the **SEC's structured 13F filings** — 11
  year-end PIT snapshots, **~2,500–5,800 filing institutions per snapshot** (median ~3,600, and
  printed at runtime, not asserted in prose), edges = cosine of shared-holder vectors, avg degree
  ~16.9. Two PIT guards live in the loader: an **exact** `13F-HR` match (drops restated
  amendments) and a **`FILING_DATE <= public`** filter (drops late filers not yet public). Runs
  the identical lead-lag harness (with a planted-recovery power proof over the real graph) on a
  genuine economic link. `graph.edges_from_pairs` is the generic hook; `data/cusip_map.csv` is
  the committed, auditable ticker→CUSIP map, **check-digit-validated and coverage-tested** so a
  bond/depositary/placeholder identifier can't silently isolate a name (the bug this review
  caught for BAC/SCHW/DIS/AMT/SPG).
- **Risk, not alpha** (`risk.py`) — the honest reframe: graphs add no *return* signal, but does a
  graph-structured **covariance** beat standard shrinkage? Builds a graph-informed covariance (a
  PSD-projected conditional-independence shrinkage target and a per-edge **graphical lasso** on the
  precision matrix), forms the **global minimum-variance portfolio**, and scores out-of-sample
  realized vol and QLIKE against Ledoit–Wolf, with a degree-preserving rewire null and a
  block-structure positive control. Finding (Run 10): the null extends from alpha to risk — every
  graph estimator (graphical lasso 0.98× LW's vol, masked ~1.02×) is *statistically indistinguishable*
  from Ledoit–Wolf (all |t| < 1.96), while plain sample/diagonal are significantly worse; the real
  graph edges its own rewire but not significantly. No graph structure meaningfully improves
  covariance estimation here.
- **Reproducibility** (`data/snapshot.py` + `data_manifest.json`) — pinned universe/dates and a
  content hash of the fetched prices (Yahoo ToS prevents shipping the data itself). Honest
  caveat: Yahoo **retroactively re-adjusts** historical closes for every later split/dividend,
  so a clone re-run months later reproduces the numbers to ~2 significant figures, **not**
  byte-identically — the hash flags *that your pull differs*, it is not a promise the study is
  bit-reproducible. The synthetic planted-recovery results, by contrast, are seeded and exactly
  reproducible, which is why they carry the load-bearing power claims.

## Data & survivorship (read before trusting return claims)

Using *today's* index membership for a historical study is a first-order confound that can
manufacture predictability. `data/universe.py` builds **point-in-time membership** and
patches **delisting returns** so the label window is complete. The synthetic demo and any
current-membership run restrict claims accordingly — **volatility results are robust;
return claims require the PIT path.**

## Layout

```
src/marketgnn/  splits · graph · features · evaluate · dataset · power · leadlag · robustness · signals · costs · conditioning · coholding[13F] · risk[covariance/GMVP] · figures · train
                models/ (ridge · gbm · losses · gnn[MLP≡GNN] · temporal[GConvGRU])
                data/   (download · universe[PIT membership] · cusip_map.csv)
tests/          77 tests — leak/PIT/stat/power/lead-lag/coholding[+CUSIP validation]/temporal/risk[covariance]/control/cost/conditioning harness
configs/        default.yaml
paper/note.md   writeup
```

## Roadmap

Both graph channels are tested — contemporaneous (Runs 1–2) and strictly-lagged spillover
(Run 5) — over the correlation/industry proxy *and* over a **real 13F co-holding graph**
(Run 8), with both a zero-parameter read and a **learned GConvGRU** (Run 9), all powered
nulls; and the null is shown to extend from **alpha to risk** (Run 10: graph-structured
covariance vs Ledoit–Wolf for the min-variance portfolio). Remaining extensions: (1) a
small/illiquid universe where the lead-lag effect is documented to survive; (2) delisted-price
data (CRSP) for a fully survivorship-free PIT run; (3) supplier–customer links from 10-K
segments as a second real economic graph.

## License

MIT.
