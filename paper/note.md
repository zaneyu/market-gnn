# Does graph structure add cross-sectional equity return signal? A leak-controlled decomposition

*Working note. Status: complete on a liquid large-cap universe over both a correlation proxy and
a real 13F economic-link graph, with zero-parameter and learned-temporal reads; a survivorship-
free universe and supplier–customer edges are noted extensions.*

## Abstract
We ask whether modelling stocks as a graph adds out-of-sample cross-sectional rank-IC over a
matched non-graph model, under point-in-time construction and purged walk-forward evaluation.
We test **both** channels by which a graph could carry return signal — contemporaneous
(same-date message passing) and strictly-lagged spillover (neighbours' past → own future) —
over both a **correlation/industry proxy** graph and a **real economic link** (SEC 13F
institutional co-holding), with both a **zero-parameter** read and a **learned temporal model**
(GConvGRU), validating detection power by recovering a *planted* lead-lag effect over each exact
graph before interpreting the real-data result. On 90 liquid US large-caps (2014–2024, weekly),
no configuration adds return signal that survives leak-free, powered, control-checked evaluation.
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
- **Real economic-link graph:** SEC 13F institutional co-holding (cosine of shared-holder
  vectors), the identical harness run on a genuine link rather than the correlation/sector proxy.
- **Learned temporal model:** a GConvGRU (graph-conv per date → GRU over dates) with the same
  `use_graph` blindfold A/B, the learned counterpart to the zero-parameter lead-lag read.
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
- **Run 5 — lead-lag is a powered null (proxy graph).** Planted-signal recovery: over the true
  graph the pipeline recovers IC 0.089 (HAC t 9.6), survives the own-momentum control, and the
  rewire null is flat — proving power. On real data: IC ~0.006 with 80% power to detect ~0.03.
- **Run 6–7 — the positive control.** Short-term (1-day) reversal is strongly significant on
  real data (IC +0.015, HAC t 3.4, FDR-sig) — so the nulls are real absences, not a dead
  pipeline — but it breaks even at ~0.9 bp (arbitraged net) and is *survivorship-exposed* (a
  positive return claim on current membership; treat as a harness control, not a tradable
  finding).
- **Run 8 — the null holds over a REAL economic link.** An institutional co-holding graph
  (Anton–Polk "Connected Stocks") built from the actual SEC 13F filings (11 PIT snapshots, exact
  `13F-HR` + filing-date PIT guards, check-digit-validated CUSIP map) is *also* a powered null:
  planted recovery +0.027 (t 4.05, rewire destroyed), then real data +0.012 (t 1.4, MDE ~0.025),
  not significant. Closes the "the proxy graph was hiding a real link" escape hatch.
- **Run 9 — the null holds for a LEARNED temporal model.** A GConvGRU (graph-conv per date → GRU
  over dates) recovers the planted lead-lag (+0.067, t 3.9; blindfold −0.012, n.s.) then nulls on
  the real co-holding graph (+0.006 vs no-graph +0.007, both n.s.). Closes the "you just needed a
  bigger model" escape hatch.
- **Run 10 — the null extends to RISK.** Covariance estimation is where structural priors are
  *known* to help (Ledoit–Wolf), so we ask whether a graph-structured covariance improves the
  out-of-sample **minimum-variance portfolio**. Every graph estimator — a PSD-projected
  conditional-independence shrinkage target (masked, ~1.02× LW's vol) and a precision-space
  **graphical lasso** (0.98×) — is statistically indistinguishable from Ledoit–Wolf (all |t| < 1.96),
  while plain sample/diagonal are significantly worse. The real graph edges its own degree-preserving
  rewire but not significantly. No graph structure meaningfully improves covariance estimation on
  large-caps — bracketed by a block-structure positive control that proves the harness has power.
- **Run 11 — the one positive, with its identification ceiling.** Neighbours' volatility
  *innovations* (short vol vs own long-run level — a vol-*level* signal is structurally
  confounded by graph-clustered levels) carry FDR-significant incremental information about own
  forward volatility along the real co-holding topology (+0.038, HAC t 2.65, rewire-clean,
  exactly at the design's MDE — borderline-powered; planted spatial-ARCH recovery proves power,
  a control-validation test proves the level confound is killed). Pre-registered limit: heterogeneous factor-vol *exposure* is
  observationally equivalent to transmission in this design (simulated exposure alone produces
  +0.10–0.22), so the claim is "volatility-relevant information travels with the topology,"
  not causal spillover. Own persistence dominates (+0.510, t 28).
- **Run 12 — the positives get the López-de-Prado battery.** CSCV/PBO over the reversal
  signal's pre-registered 3×3 grid (daily Jegadeesh–Titman tranche series; 12,870 splits;
  strictly-below-median convention, noise ⇒ 4/9) and deflated Sharpe at N ∈ {9,25,100}:
  **PBO 0.437 ≈ noise, DSR 0.74 (N=9) → 0.62 (N=100)** — the reversal control is demoted from
  "the one real return signal" to a genuine cross-sectional **association that does not
  validate as a strategy even gross**. Every grid config has positive gross Sharpe (+0.16 to
  +0.52), so the verdict is "no reliable selection," not "no effect." Run 11's IC gets the
  same PSR treatment (HAC T_eff = 85 of 125; N=1, pre-registration is its protection) and
  survives at **0.9946** — conclusion unchanged.
- **Primary endpoint (H1), tested directly.** The pre-registered GNN-vs-matched-MLP comparison is
  a *paired* per-date IC-difference test (HAC t + block-bootstrap CI + MDE), reported outside the
  FDR family — not two eyeballed IC-vs-zero rows. It isolates the incremental value of message-
  passing over an already-graph-informed MLP. Synthetic: ΔIC +0.005, t 0.33, p 0.75 — no gap; but
  the MDE (~0.047) exceeds a plausible message-passing edge (~0.005–0.02), so H1 is underpowered
  for a small edge (consistent with, not ruling out, one). Power rests on the planted recoveries.

## 4. Interpretation
On liquid large-caps, no graph configuration adds cross-sectional return signal — not the
correlation proxy, not a real 13F economic link, not a zero-parameter read, not a learned
GConvGRU. This is consistent with the lead-lag literature, where spillover predictability
concentrates in small/illiquid names (excluded here by construction) and has decayed post-2000.
The contemporaneous "graph effect" people report is, in this decomposition, attributable to
survivorship, label overlap, and the graph-derived feature — not topology.

## 5. What would change the conclusion
A true economic-link graph on a small/illiquid, survivorship-free universe is the setting where
the lead-lag channel is documented to survive. The 13F co-holding graph (Run 8) removes the
"proxy" objection but not the universe one; supplier–customer links from 10-K segments and a
CRSP-backed survivorship-free universe are the remaining extensions. The harness is built to
test exactly that; the binding constraints are edge and price data.

## 6. Threats to validity
Large-cap-only (by design); yfinance adjustment quality and non-reproducibility of the live
pull (synthetic evidence is deterministic); transaction costs excluded (predictability study);
weekly cadence power (mitigated by reporting MDE).

## 7. Reproducibility
`pip install -e ".[dev,gnn]"`; `pytest -q` (89 tests; 3 GNN/temporal skip without torch, 1 skips
without the fetched 13F data); `python -m marketgnn.leadlag --synthetic-planted` (planted
recovery); `python -m marketgnn.coholding` (real 13F graph); `python -m marketgnn.models.temporal
--synthetic-planted` (GConvGRU power); `python -m marketgnn.train --synthetic --models
ridge,mlp,gnn` (grid + H1 primary endpoint). Every leak/PIT/power control is a runnable test; the
planted-signal recoveries are the load-bearing, **exactly reproducible** demonstrations that the
nulls are powered, not empty. Real-data figures reproduce to ~2 significant figures only — Yahoo
retroactively re-adjusts historical closes — so the synthetic recoveries carry the power claims.
