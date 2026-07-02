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
2. **Lead-lag / spillover is a *powered* null on large-caps.** Neighbours' past returns →
   own future return (the channel Cohen–Frazzini / industry-momentum predicts). We first
   **plant** the effect in a synthetic market and show the pipeline **recovers it** (IC 0.089,
   HAC t 9.6, rewire-null clean) — proving power — then on real data find ~0.006 with **80%
   power to detect 0.036**. Absent, not undetectable; consistent with the effect living in
   small/illiquid names this universe excludes.
3. **The vol "predictability" is ~90% persistence** — a naive trailing-vol forecast gets 0.435
   of the model's 0.479, and the model is *worse* on QLIKE (level calibration).
4. **Survivorship inflates it:** point-in-time membership correction shrinks the already-null
   return IC.
5. **What DOES carry signal (positive control):** the same harness finds **short-term (1-day)
   reversal strongly significant on real data** — IC +0.015, HAC t 3.4, p<0.001, FDR-sig. So
   the graph nulls are *real absences*, bracketed by positive controls on both real data
   (reversal) and synthetic data (planted lead-lag) — not a pipeline that can't find anything.
   (Gross rank-IC; reversal is high-turnover and cost-fragile — reported as a statistical
   control, not a tradable strategy.)

**Net:** the graph adds no return signal that survives leak-free, powered, control-checked
evaluation; the return signal that *is* present (short-horizon reversal) is non-graph and
cost-fragile — and this repo is the harness that separates a real effect from the survivorship /
overlap / over-smoothing artifacts that make graph "alpha" look real.
Full decomposition and numbers in [RESULTS.md](RESULTS.md).

## Why trust the null? The harness has teeth.

A null is only worth reporting if the pipeline could have found signal. Two guarantees:
- **It detects signal when present** — the planted lead-lag recovery above, and a
  **future-injection canary** (feed a feature = tomorrow's return → IC spikes to ~1.0).
- **It rejects spurious signal** — **label-shuffle** collapses IC to ~0; the **degree-preserving
  rewire** null stays flat; **purged walk-forward** removes label-overlap leakage; point-in-time
  purity of graph/features/labels is asserted, not assumed.

```bash
pytest -q     # 47 tests (2 GNN A/B tests skip without torch); core needs no GNN stack
```

## Quickstart

```bash
pip install -e ".[dev]"          # core + tests (no torch needed)
pytest -q                        # 47 tests (45 without torch), ~26s
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
- **Robustness** (`robustness.py`) — vol model vs the naive random-walk forecast (rank-IC and
  QLIKE), showing how much of the vol IC is mere persistence.

## Data & survivorship (read before trusting return claims)

Using *today's* index membership for a historical study is a first-order confound that can
manufacture predictability. `data/universe.py` builds **point-in-time membership** and
patches **delisting returns** so the label window is complete. The synthetic demo and any
current-membership run restrict claims accordingly — **volatility results are robust;
return claims require the PIT path.**

## Layout

```
src/marketgnn/  splits · graph · features · evaluate · dataset · power · leadlag · robustness · train
                models/ (ridge · gbm · losses · gnn[MLP≡GNN] · temporal[stub])
                data/   (download · universe[PIT membership])
tests/          47 tests — leak/PIT/purge/stat/power/lead-lag/positive-control harness
configs/        default.yaml
paper/note.md   writeup
```

## Roadmap

Both graph channels are now tested: contemporaneous (Runs 1–2) and strictly-lagged spillover
(Run 5, zero-parameter so nothing to overfit). Natural extensions: (1) a true economic-link
graph (supplier/customer from 10-K segments, shared-analyst coverage) in place of the
correlation/industry proxy — the lead-lag literature's effect is strongest there; (2) a
small/illiquid universe where that effect is documented to survive; (3) delisted-price data
(CRSP) for a fully survivorship-free PIT run; (4) a GConvGRU spatiotemporal model — though the
zero-parameter lead-lag signal is the stronger test of *whether linkage carries information*.

## License

MIT.
