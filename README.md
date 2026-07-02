# market-gnn

**A graph neural network for cross-sectional equity signals — and a leak-controlled
test of whether the graph actually helps.**

Stocks co-move in groups (sector, supply chain, shared factors). This project models
that structure as a graph, runs a GNN over it, and asks one clean, falsifiable
question:

> Does graph structure add **incremental out-of-sample rank-IC** over an **identical
> feature set fed to a matched non-graph model**, after point-in-time construction and
> purged walk-forward evaluation?

The honest answer may be "only for volatility, not returns." That's fine — the point is
to measure it without fooling ourselves. Everything here is built so a skeptic can clone
it, run it, and watch the leak-detectors fire.

## Does the evaluation actually catch leaks? (run this first)

The whole result is only worth trusting if the harness has teeth. Two tests prove it:

```bash
pytest tests/test_leakage_canary.py -q
```
- **Future-injection canary** — feed the model a feature equal to *tomorrow's return*;
  IC must spike to ~1.0 (the detector fires).
- **Label-shuffle** — scramble the targets within each date; IC must collapse to ~0
  (a real score can't survive it).

Plus point-in-time purity of the graph, features, and neighbour signal
(`tests/test_graph_pit.py`, `test_features_pit.py`, `test_neighbor_feature.py`), exact
purge/embargo correctness (`test_splits.py`), and per-date-only normalization
(`test_normalize_scope.py`). **34 tests, no GNN stack required.**

## Quickstart

```bash
pip install -e ".[dev]"          # core + tests (no torch needed)
pytest -q                        # 34 tests, ~5s
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

## Data & survivorship (read before trusting return claims)

Using *today's* index membership for a historical study is a first-order confound that can
manufacture predictability. `data/universe.py` builds **point-in-time membership** and
patches **delisting returns** so the label window is complete. The synthetic demo and any
current-membership run restrict claims accordingly — **volatility results are robust;
return claims require the PIT path.**

## Layout

```
src/marketgnn/  splits · graph · features · evaluate · dataset · power · train
                models/ (ridge · gbm · losses · gnn[MLP≡GNN] · temporal[stub])
                data/   (download · universe)
tests/          34 tests — the leak/PIT/purge/stat harness
configs/        default.yaml
paper/note.md   writeup (incl. "the result I almost believed")
```

## Roadmap

Static, contemporaneous graph only (this repo). The lead-lag / momentum-spillover channel
is inherently temporal (neighbour's return today → mine tomorrow) and is the documented
**v2**: a GConvGRU-style spatiotemporal model. Deferred deliberately — a finished, tight
static study beats a half-built temporal one.

## License

MIT.
