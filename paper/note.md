# Does graph structure add cross-sectional equity return signal? A leak-controlled decomposition

*Working note. Status: complete on a liquid large-cap universe; economic-link edges and a
survivorship-free universe are noted extensions.*

## Abstract
We ask whether modelling stocks as a graph adds out-of-sample cross-sectional rank-IC over a
matched non-graph model, under point-in-time construction and purged walk-forward evaluation.
We test **both** channels by which a graph could carry return signal — contemporaneous
(same-date message passing) and strictly-lagged spillover (neighbours' past → own future) —
and validate detection power by recovering a *planted* lead-lag effect on synthetic data
before interpreting the real-data result. On 90 liquid US large-caps (2014–2024, weekly), no
configuration adds return signal that survives leak-free, powered, control-checked evaluation.
The contribution is the decomposition and the harness that separates a real effect from the
survivorship / label-overlap / over-smoothing artifacts that make graph "alpha" look real.

## 1. Data
Daily adjusted OHLCV (yfinance), 90 large-caps, weekly rebalance, 5-day-ahead labels. Beta vs
SPY (exogenous). Point-in-time S&P 500 membership from the public change log corrects
inclusion timing. *Limitation:* delisted-name prices are unavailable via yfinance, so the
real-data run is inclusion-corrected but not fully survivorship-free (needs CRSP); the primary
*mechanistic* evidence is therefore the synthetic planted-signal recovery.

## 2. Method
- **Graphs** (point-in-time): trailing-correlation kNN (raw, shrinkage, frozen), sector
  (industry), degree-preserving rewire null.
- **Contemporaneous test:** ridge/LightGBM with a neighbour-return feature, and the GNN≡MLP
  A/B (same net, edges on/off; shared ranking loss + purged-val early stopping).
- **Lead-lag test:** zero-parameter signal = edge-weighted mean of neighbours' return over
  (t−h, t], predicting own return over (t, t+h]; own-momentum residual control; rewire null.
- **Evaluation:** per-date rank-IC → Newey–West HAC t-stats, block bootstrap, BH-FDR;
  minimum-detectable-effect reported next to every null.

## 3. Results
See RESULTS.md for full tables. Summary:

- **Run 1–2 — contemporaneous graph is a red herring.** Frozen ≈ dynamic ≈ no-graph (ridge);
  GNN == MLP with no edges (to the digit), and slightly worse with the correlation graph
  (over-smoothing). Returns null; volatility significant but see Run 3.
- **Run 3 — volatility is ~90% persistence.** Naive trailing-vol forecast gets rank-IC 0.435
  of the model's 0.479, and the model is worse on QLIKE (level). Not model skill.
- **Run 4 — survivorship inflates returns.** PIT inclusion correction shrinks the already-null
  return IC in the expected direction.
- **Run 5 — lead-lag is a powered null.** Planted-signal recovery: over the true graph the
  pipeline recovers IC 0.089 (HAC t 9.6), survives the own-momentum control, and the rewire
  null is flat — proving power. On real data: IC ~0.006 with 80% power to detect 0.036.

## 4. Interpretation
On liquid large-caps, no graph configuration adds cross-sectional return signal. This is
consistent with the lead-lag literature, where spillover predictability concentrates in
small/illiquid names (excluded here by construction) and has decayed post-2000. The
contemporaneous "graph effect" people report is, in this decomposition, attributable to
survivorship, label overlap, and the graph-derived feature — not topology.

## 5. What would change the conclusion
A true economic-link graph (supplier/customer, shared-analyst) on a small/illiquid,
survivorship-free universe is the setting where the lead-lag channel is documented to survive.
The harness is built to test exactly that; the binding constraints are edge and price data.

## 6. Threats to validity
Large-cap-only (by design); yfinance adjustment quality and non-reproducibility of the live
pull (synthetic evidence is deterministic); transaction costs excluded (predictability study);
weekly cadence power (mitigated by reporting MDE).

## 7. Reproducibility
`pip install -e ".[dev,gnn]"`; `pytest -q` (45 tests); `python -m marketgnn.leadlag
--synthetic-planted` (planted recovery); `python -m marketgnn.train --synthetic`. Every
leak/PIT/power control is a runnable test; the planted-signal recovery is the load-bearing
demonstration that the null is powered, not empty.
