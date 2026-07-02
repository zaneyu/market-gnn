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


def _fold_predictions(model_name, dataset, cv, target, seed):
    """Concatenate out-of-sample test predictions across all folds for one cell."""
    y = dataset.y_ret if target == "ret" else dataset.y_vol
    is_neural = model_name in ("mlp", "gnn")
    preds, targs, dts = [], [], []
    for fold in cv.split(dataset.dates):
        tr, te = list(fold.train_dates), list(fold.test_dates)
        model = build_model(model_name, seed=seed)
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


def run(config: dict) -> pd.DataFrame:
    cfg = {**DEFAULTS, **config}
    from .data.download import load_market
    from .data.universe import default_universe

    prices, volume, sectors, market = load_market(
        synthetic=cfg.get("synthetic", True),
        tickers=cfg.get("tickers") or default_universe(),
        start=cfg.get("start", "2014-01-01"),
        end=cfg.get("end", "2024-12-31"),
    )
    rebal = rebalance_dates(prices.index, cfg["rebal_freq"])
    cv = PurgedWalkForward(
        label_horizon=cfg["purge_steps"], n_test=cfg["n_test"], embargo=cfg["embargo_steps"],
        min_train=cfg["min_train"], step=cfg["step"],
    )

    models = [m for m in cfg["models"] if _deps_ok(m) or print(f"[skip] {m}: missing deps")]
    rows = []
    for gk in cfg["graph_kinds"]:
        ds = build_dataset(
            prices, volume, sectors, market, rebal, graph_kind=gk,
            label_horizon=cfg["label_horizon"], corr_window=cfg["corr_window"],
            k=cfg["k"], nbr_lookback=cfg["nbr_lookback"], warmup=cfg["warmup"],
        )
        for model_name, target in product(models, cfg["targets"]):
            pred, targ, dts = _fold_predictions(model_name, ds, cv, target, cfg["seed"])
            ic = per_date_ic(pred, targ, dts)
            s = ic_summary(ic, hac_lag=cfg["purge_steps"])
            lo, hi = block_bootstrap_ci(ic.dropna().to_numpy(), block=max(2, cfg["purge_steps"] + 1), seed=cfg["seed"])
            rows.append({
                "graph": gk, "model": model_name, "target": target,
                "mean_ic": s["mean_ic"], "hac_t": s["hac_t"], "naive_t": s["naive_t"],
                "ci_lo": lo, "ci_hi": hi, "n_dates": s["n"], "p": two_sided_p(s["hac_t"]),
            })

    table = pd.DataFrame(rows)
    if len(table):
        reject, q = benjamini_hochberg(table["p"].fillna(1.0).to_numpy())
        table["fdr_sig"], table["q"] = reject, q
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
    print("\nHAC t-stats account for label autocorrelation; q = BH-FDR across the grid.")
    print("Primary endpoint (H1): compare model=gnn vs model=mlp at target=ret.")


if __name__ == "__main__":
    main()
