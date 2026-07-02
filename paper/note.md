# Does graph structure add cross-sectional equity signal? A leak-controlled test

*Working note. Status: pipeline complete on synthetic data; real-data run pending.*

## Abstract
We test whether modelling stocks as a graph and applying a GNN yields higher
out-of-sample cross-sectional rank-IC than a matched non-graph model, under
point-in-time construction and purged walk-forward evaluation. Primary endpoint
(pre-registered): GNN vs matched MLP on next-period return rank. Secondary:
volatility target, edge-type ablations, and a degree-preserving random-graph
control. All significance uses HAC t-stats and a block bootstrap; the grid is
FDR-controlled.

## 1. Data & universe
- Russell 1000, **point-in-time membership** (survivorship defense); delisting
  returns patched so the label window is complete.
- Daily OHLCV (yfinance, adjusted), weekly rebalance. Beta vs **SPY** (exogenous),
  never a survivor equal-weight mean.
- *Limitation to state plainly:* data-vendor adjustment quality; current-membership
  fallback (if used) restricts return claims to the volatility target.

## 2. Graph construction
Point-in-time per date: trailing-correlation kNN (window W, top-k |corr|) and sector
co-membership. Model applies symmetrize + self-loops. **Null:** degree-sequence-
preserving rewire (matches in/out-degree exactly) — isolates topology from degree.

## 3. Features
Momentum (1m/3m/12-1), reversal (1w), realized vol, turnover, beta(SPY), size, and a
neighbour-aggregated trailing-return feature (the contemporaneous relative-strength
channel, consumed by both models). Cross-sectional normalization per date.

## 4. Models
Ridge, LightGBM (tough baseline), and the GNN≡MLP pair (same net, edges on/off),
shared ranking loss + head + purged-val early stopping.

## 5. Evaluation
Purged, embargoed walk-forward. Per-date rank-IC → HAC t-stat + IC-IR, decile
long-short spread, turnover; volatility via QLIKE vs naive. Block-bootstrap CIs,
multi-seed variance, BH-FDR across the grid. Power/MDE reported.

## 6. Results
*(populate from the real-data run)*
- H3 anchor (volatility): all models beat naive — pipeline validated.
- H1 (returns): GNN vs MLP — report effect, HAC t, CI, and whether it clears FDR.
- H2 (control): real graph vs degree-preserving rewire.
- H4 (edge type): correlation vs sector vs both (FDR-controlled).

## 7. The result I almost believed
*(the centerpiece — fill in as it happens)* The first number that looked great, the
specific test/control that revealed it was leakage or survivorship, and the corrected
number. This is the honest core of the note: skepticism → test → kill → report.

## 8. Limitations & threats to validity
Power on weekly cadence; residual look-ahead in sector labels; transaction costs
excluded (predictability study, not a strategy); vendor data quality.

## 9. Reproducibility
`pip install -e ".[dev,gnn]"`; `pytest -q`; `python -m marketgnn.train --config
configs/default.yaml`. Seeds fixed; data cached; every leakage control is a runnable test.

## Appendix: expected honest outcome
Graph adds small-but-real IC for volatility, marginal-to-none for returns after
controls; sector edges ≥ correlation edges; GBM is a hard baseline. Reporting that —
with the control that dissolves any naive positive — is the contribution.
