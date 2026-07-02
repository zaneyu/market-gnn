"""Volatility robustness: is the ~0.48 vol IC real model skill, or just volatility
persistence? The honest test is whether the model beats the naive random-walk
forecast (forward vol = trailing vol) on BOTH rank-IC and QLIKE. If the naive
forecast already gets ~0.48, the model isn't adding anything -- which is the honest
thing to report, not a failure.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .dataset import build_dataset, rebalance_dates
from .evaluate import per_date_ic, qlike
from .models import build_model
from .splits import PurgedWalkForward


def run_vol_robustness(config: dict | None = None) -> dict:
    from .data.download import load_market
    from .data.universe import default_universe

    cfg = {
        "synthetic": False, "start": "2014-01-01", "end": "2024-12-31",
        "label_horizon": 5, "corr_window": 60, "k": 10, "warmup": 260,
        "rebal_freq": "W", "purge_steps": 1, "embargo_steps": 2,
        "n_test": 52, "min_train": 200, "step": 52, "naive_window": 20, "seed": 0,
        **(config or {}),
    }
    prices, volume, sectors, market = load_market(
        synthetic=cfg["synthetic"], tickers=cfg.get("tickers") or default_universe(),
        start=cfg["start"], end=cfg["end"],
    )
    rebal = rebalance_dates(prices.index, cfg["rebal_freq"])
    ds = build_dataset(prices, volume, sectors, market, rebal, graph_kind="correlation",
                       label_horizon=cfg["label_horizon"], corr_window=cfg["corr_window"],
                       k=cfg["k"], warmup=cfg["warmup"])
    cv = PurgedWalkForward(cfg["purge_steps"], cfg["n_test"], embargo=cfg["embargo_steps"],
                           min_train=cfg["min_train"], step=cfg["step"])

    # naive forecast: log trailing annualized vol (random-walk-in-vol), PIT by construction
    rets = prices.pct_change()
    rv = rets.rolling(cfg["naive_window"]).std() * np.sqrt(252)
    naive_log = np.log(rv)

    model_p, naive_p, realized, dts = [], [], [], []
    for fold in cv.split(ds.dates):
        m = build_model("ridge", seed=cfg["seed"])
        Xtr = ds.X.loc[list(fold.train_dates)]
        ytr = ds.y_vol.loc[Xtr.index]
        keep = ytr.notna().to_numpy()
        m.fit(Xtr.to_numpy()[keep], ytr.to_numpy()[keep])
        Xte = ds.X.loc[list(fold.test_dates)]
        pred = pd.Series(m.predict(Xte.to_numpy()), index=Xte.index)
        nv = pd.Series([naive_log.at[d, a] for d, a in Xte.index], index=Xte.index)
        y = ds.y_vol.loc[Xte.index]
        model_p.append(pred); naive_p.append(nv); realized.append(y)
        dts.append(pred.index.get_level_values("date").to_numpy())

    model_p = pd.concat(model_p); naive_p = pd.concat(naive_p); realized = pd.concat(realized)
    dates = np.concatenate(dts)

    ic_model = per_date_ic(model_p.to_numpy(), realized.to_numpy(), dates).dropna()
    ic_naive = per_date_ic(naive_p.to_numpy(), realized.to_numpy(), dates).dropna()
    # QLIKE on variances (vol = exp(log-vol) -> var = exp(2*log-vol))
    r_var = np.exp(2 * realized.to_numpy())
    qlike_model = qlike(r_var, np.exp(2 * model_p.to_numpy()))
    qlike_naive = qlike(r_var, np.exp(2 * naive_p.to_numpy()))
    return {
        "ic_model": float(ic_model.mean()), "ic_naive": float(ic_naive.mean()),
        "qlike_model": qlike_model, "qlike_naive": qlike_naive,
        "n_obs": int(len(realized)), "n_dates": int(ic_model.shape[0]),
    }


def main():
    r = run_vol_robustness()
    print("=== volatility robustness: model vs naive random-walk forecast ===")
    print(f"rank-IC   model {r['ic_model']:+.3f}   naive {r['ic_naive']:+.3f}   "
          f"(delta {r['ic_model'] - r['ic_naive']:+.3f})")
    print(f"QLIKE     model {r['qlike_model']:.4f}   naive {r['qlike_naive']:.4f}   "
          f"(lower is better; delta {r['qlike_model'] - r['qlike_naive']:+.4f})")
    print(f"n_obs {r['n_obs']}  n_dates {r['n_dates']}")


if __name__ == "__main__":
    main()
