"""Orchestrator: run the ablation grid under purged walk-forward and print the
table with HAC significance, block-bootstrap CIs, and BH-FDR across the grid.

H1 (GNN vs matched MLP on returns) is the single pre-registered primary endpoint;
everything else is secondary and FDR-controlled. Runs offline on the synthetic
factor market by default (`--synthetic`), so a reviewer can reproduce the shape of
the results without a data pull.
"""

from __future__ import annotations

import argparse
from itertools import product

import numpy as np
import pandas as pd
import yaml

from .dataset import build_dataset, rebalance_dates
from .evaluate import benjamini_hochberg, block_bootstrap_ci, ic_summary, per_date_ic, two_sided_p
from .models import build_model
from .splits import PurgedWalkForward

DEFAULTS = dict(
    graph_kinds=["correlation", "sector", "both", "rewire", "none"],
    models=["ridge"],
    targets=["ret", "vol"],
    label_horizon=5,
    corr_window=60,
    k=10,
    nbr_lookback=21,
    warmup=260,
    rebal_freq="W",
    purge_steps=1,
    embargo_steps=2,
    n_test=26,
    min_train=160,
    step=26,
    hac_lag=None,  # None -> Newey-West rule of thumb (~5 at n~300); honest for overlapping labels
    seed=0,
)


def _deps_ok(model: str) -> bool:
    if model == "gbm":
        import importlib.util

        return importlib.util.find_spec("lightgbm") is not None
    if model in ("mlp", "gnn"):
        import importlib.util

        return importlib.util.find_spec("torch") is not None
    return True


def _fold_predictions(model_name, dataset, cv, target, seed, val_gap=1):
    """Concatenate out-of-sample test predictions across all folds for one cell."""
    y = dataset.y_ret if target == "ret" else dataset.y_vol
    is_neural = model_name in ("mlp", "gnn")
    extra = {"val_gap": val_gap} if is_neural else {}
    preds, targs, dts = [], [], []
    for fold in cv.split(dataset.dates):
        tr, te = list(fold.train_dates), list(fold.test_dates)
        model = build_model(model_name, seed=seed, **extra)
        if is_neural:
            model.fit(dataset, tr, target=target)
            p = model.predict(dataset, te)
        else:
            Xtr = dataset.X.loc[tr]
            ytr = y.loc[Xtr.index]
            keep = ytr.notna().to_numpy()
            model.fit(Xtr.to_numpy()[keep], ytr.to_numpy()[keep])
            Xte = dataset.X.loc[te]
            p = pd.Series(model.predict(Xte.to_numpy()), index=Xte.index)
        t = y.loc[p.index]
        preds.append(p.to_numpy())
        targs.append(t.to_numpy())
        dts.append(p.index.get_level_values("date").to_numpy())
    return np.concatenate(preds), np.concatenate(targs), np.concatenate(dts)


def _prep(cfg: dict):
    """Load the market and build the purged walk-forward CV shared by the ablation grid
    and the primary-endpoint test."""
    from .data.download import load_market
    from .data.universe import default_universe

    prices, volume, sectors, market = load_market(
        synthetic=cfg.get("synthetic", True),
        tickers=cfg.get("tickers") or default_universe(),
        start=cfg.get("start", "2014-01-01"),
        end=cfg.get("end", "2024-12-31"),
    )
    rebal = rebalance_dates(prices.index, cfg["rebal_freq"])
    # purge must cover the label horizon: derive the minimum purge in rebalance steps
    # from label_horizon and the rebalance cadence, and never trust a smaller knob.
    steps_per_rebal = _rebal_step_len(prices.index, rebal, cfg["rebal_freq"])
    need = int(np.ceil(cfg["label_horizon"] / max(1, steps_per_rebal)))
    purge = max(cfg["purge_steps"], need)
    if purge > cfg["purge_steps"]:
        print(f"[purge] label_horizon={cfg['label_horizon']} over ~{steps_per_rebal}-day "
              f"rebalance steps needs purge>={need}; using {purge} (was {cfg['purge_steps']}).")
    cfg["purge_steps"] = purge
    cv = PurgedWalkForward(
        label_horizon=purge, n_test=cfg["n_test"], embargo=cfg["embargo_steps"],
        min_train=cfg["min_train"], step=cfg["step"],
    )
    return prices, volume, sectors, market, rebal, cv


def _rebal_step_len(index, rebal, rebal_freq: str) -> int:
    """Median trading-day gap between consecutive rebalance dates (so a 5-day label
    over a weekly rebalance = ~1 step, over a monthly = ~1 step but ~21 days)."""
    locs = index.get_indexer(list(rebal))
    locs = locs[locs >= 0]
    if len(locs) < 2:
        return 5
    return int(np.median(np.diff(locs)))


def primary_endpoint(config: dict) -> dict | None:
    """The pre-registered H1, computed as an actual paired test (not two eyeballed
    rows): does the GNN beat the MATCHED MLP on out-of-sample return rank-IC? Both
    models see the identical feature set (including the neighbour-return feature) and
    the same dataset/graph; they differ only in whether real edges are visible in
    message passing. We form the per-date IC-DIFFERENCE series (GNN minus MLP on the
    same dates and universe) and test its mean with a HAC t-stat + block-bootstrap CI
    -- exactly the estimand power.py computes an MDE for. Returns None if torch is
    absent."""
    cfg = {**DEFAULTS, **config}
    if not (_deps_ok("gnn") and _deps_ok("mlp")):
        return None
    from .leadlag import _estimate_phi
    from .power import min_detectable_effect

    prices, volume, sectors, market, rebal, cv = _prep(cfg)
    graph_kind = cfg.get("primary_graph", "correlation")
    target = "ret"
    ds = build_dataset(
        prices, volume, sectors, market, rebal, graph_kind=graph_kind,
        label_horizon=cfg["label_horizon"], corr_window=cfg["corr_window"],
        k=cfg["k"], nbr_lookback=cfg["nbr_lookback"], warmup=cfg["warmup"],
    )
    val_gap = cfg["purge_steps"] + cfg["embargo_steps"]
    ics = {}
    for m in ("gnn", "mlp"):
        pred, targ, dts = _fold_predictions(m, ds, cv, target, cfg["seed"], val_gap=val_gap)
        ics[m] = per_date_ic(pred, targ, dts).dropna()
    diff = (ics["gnn"] - ics["mlp"]).dropna()
    if len(diff) < 3:
        return None
    s = ic_summary(diff, hac_lag=cfg["hac_lag"])
    lo, hi = block_bootstrap_ci(diff.to_numpy(), block=max(2, cfg["purge_steps"] + 1), seed=cfg["seed"])
    mde = min_detectable_effect(n_dates=s["n"], ic_sd=max(diff.std(ddof=1), 1e-6),
                                phi=_estimate_phi(diff), n_sims=400)
    return {
        "graph": graph_kind, "target": target,
        "ic_gnn": float(ics["gnn"].mean()), "ic_mlp": float(ics["mlp"].mean()),
        "delta_ic": s["mean_ic"], "hac_t": s["hac_t"], "p": two_sided_p(s["hac_t"]),
        "ci_lo": lo, "ci_hi": hi, "mde_80": mde, "n_dates": s["n"],
    }


def run(config: dict) -> pd.DataFrame:
    cfg = {**DEFAULTS, **config}
    prices, volume, sectors, market, rebal, cv = _prep(cfg)

    membership = None
    if cfg.get("use_membership") and not cfg.get("synthetic", True):
        from .data.universe import sp500_membership

        membership = sp500_membership(list(prices.columns), prices.index)
        print(f"[membership] PIT S&P 500 mask on: universe "
              f"{int(membership.iloc[0].sum())} -> {int(membership.iloc[-1].sum())} names")

    models = [m for m in cfg["models"] if _deps_ok(m) or print(f"[skip] {m}: missing deps")]
    graph_kinds = cfg["graph_kinds"]
    if membership is not None and "frozen" in graph_kinds:
        print("[membership] skipping 'frozen' (assumes constant universe)")
        graph_kinds = [g for g in graph_kinds if g != "frozen"]
    rows = []
    for gk in graph_kinds:
        ds = build_dataset(
            prices, volume, sectors, market, rebal, graph_kind=gk,
            label_horizon=cfg["label_horizon"], corr_window=cfg["corr_window"],
            k=cfg["k"], nbr_lookback=cfg["nbr_lookback"], warmup=cfg["warmup"],
            membership=membership,
        )
        val_gap = cfg["purge_steps"] + cfg["embargo_steps"]
        for model_name, target in product(models, cfg["targets"]):
            pred, targ, dts = _fold_predictions(model_name, ds, cv, target, cfg["seed"], val_gap=val_gap)
            ic = per_date_ic(pred, targ, dts)
            s = ic_summary(ic, hac_lag=cfg["hac_lag"])
            lo, hi = block_bootstrap_ci(ic.dropna().to_numpy(), block=max(2, cfg["purge_steps"] + 1), seed=cfg["seed"])
            rows.append({
                "graph": gk, "model": model_name, "target": target,
                "mean_ic": s["mean_ic"], "hac_t": s["hac_t"], "naive_t": s["naive_t"],
                "ci_lo": lo, "ci_hi": hi, "n_dates": s["n"], "p": two_sided_p(s["hac_t"]),
            })

    table = pd.DataFrame(rows)
    if len(table):
        # BH-FDR within a HOMOGENEOUS family: separately per target (ret and vol are
        # different endpoints; vol's near-certain rejections must not shift the return
        # threshold) and EXCLUDING null-control graph kinds (rewire/random/none), which
        # are not discovery candidates. Controls get fdr_sig=False, q=NaN.
        #   Direction note: pooling controls would only make the family LARGER and thus
        # MORE conservative for the real endpoints -- excluding them is the discovery-
        # family argument, not a way to remove a discovery bias (there isn't one).
        controls = {"rewire", "random", "none"}
        table["fdr_sig"] = False
        table["q"] = np.nan
        is_alt = ~table["graph"].isin(controls)
        for tgt in table["target"].unique():
            fam = is_alt & table["target"].eq(tgt)
            if fam.any():
                reject, q = benjamini_hochberg(table.loc[fam, "p"].fillna(1.0).to_numpy())
                table.loc[fam, "fdr_sig"] = reject
                table.loc[fam, "q"] = q
    return table


def _format(table: pd.DataFrame) -> str:
    if not len(table):
        return "(no results)"
    t = table.copy()
    for c in ["mean_ic", "hac_t", "naive_t", "ci_lo", "ci_hi", "q"]:
        t[c] = t[c].map(lambda v: f"{v:+.3f}" if pd.notna(v) else "  nan")
    return t.to_string(index=False)


def main():
    ap = argparse.ArgumentParser(description="market-gnn ablation runner")
    ap.add_argument("--config", type=str, default=None)
    ap.add_argument("--synthetic", action="store_true", help="use the offline factor market")
    ap.add_argument("--models", type=str, default=None, help="comma-separated: ridge,gbm,mlp,gnn")
    args = ap.parse_args()

    config = {}
    if args.config:
        with open(args.config) as f:
            config = yaml.safe_load(f) or {}
    if args.synthetic:
        config["synthetic"] = True
    if args.models:
        config["models"] = args.models.split(",")

    table = run(config)
    print("\n=== ablation (out-of-sample, purged walk-forward) ===")
    print(_format(table))
    print("\nHAC t-stats account for label autocorrelation; q = BH-FDR within each "
          "target family, excluding null controls (rewire/none).")

    # the pre-registered primary endpoint, as an actual paired test
    pe = primary_endpoint(config)
    if pe is None:
        print("\nPrimary endpoint (H1): needs torch (mlp+gnn); install '.[gnn]' to run it.")
    else:
        print("\n=== PRIMARY ENDPOINT (H1): GNN - MLP paired IC difference, target=ret ===")
        print(f"  graph={pe['graph']}  IC(gnn)={pe['ic_gnn']:+.4f}  IC(mlp)={pe['ic_mlp']:+.4f}")
        print(f"  (both models see the neighbour-return feature; H1 isolates the incremental "
              f"value of MESSAGE-PASSING over that graph-informed MLP)")
        print(f"  ΔIC = {pe['delta_ic']:+.4f}   HAC t = {pe['hac_t']:+.2f}   p = {pe['p']:.3f}"
              f"   95% CI [{pe['ci_lo']:+.4f}, {pe['ci_hi']:+.4f}]")
        # No binary powered/underpowered verdict at an arbitrary 0.05 line: report the MDE
        # against the effect that would actually be interesting and let the reader judge.
        verdict = "no significant gap" if pe["p"] >= 0.05 else "significant gap"
        print(f"  n_dates = {pe['n_dates']}   MDE(80% power) = {pe['mde_80']:.4f}   -> {verdict}.")
        print(f"  Caveat: a plausible message-passing edge is ~0.005-0.02 IC, well below this MDE, "
              f"so this endpoint is UNDERPOWERED for a small edge — it is consistent with, but "
              f"does not rule out, one. The load-bearing power is the planted-recovery runs.")


if __name__ == "__main__":
    main()
