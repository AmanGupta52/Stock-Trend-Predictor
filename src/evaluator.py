"""Metrics, uncertainty intervals, model selection and backtesting.

Everything is computed from out-of-sample predictions only (validation block for selection and interval
calibration, untouched test block for reporting).
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from .model_trainer import FAMILY


def horizon_metrics(pred: np.ndarray, actual: np.ndarray, close_t: np.ndarray, horizons: List[int]) -> Dict[int, dict]:
    """pred/actual: (n, H) cumulative log returns; close_t: (n,) base price. Price metrics in price units."""
    out = {}
    for h in horizons:
        p, a = pred[:, h - 1], actual[:, h - 1]
        pp, ap = close_t * np.exp(p), close_t * np.exp(a)
        nz = a != 0
        out[h] = {
            "mae": float(np.mean(np.abs(pp - ap))),
            "rmse": float(np.sqrt(np.mean((pp - ap) ** 2))),
            "mape": float(np.mean(np.abs(pp - ap) / ap) * 100),
            "mae_ret": float(np.mean(np.abs(p - a))),
            "dir_acc": float(np.mean(np.sign(p[nz]) == np.sign(a[nz])) * 100) if nz.any() and np.any(p != 0) else None,
            "up_share": float(np.mean(a[nz] > 0) * 100) if nz.any() else None,
            "n": int(len(a)),
        }
    return out


def relative_mae(metrics: Dict[int, dict], naive: Dict[int, dict]) -> float:
    """Mean over horizons of MAE(model) / MAE(naive). <1 means better than 'no change'."""
    return float(np.mean([metrics[h]["mae_ret"] / naive[h]["mae_ret"] for h in metrics]))


def residual_quantiles(pred_val: np.ndarray, actual_val: np.ndarray, qs) -> dict:
    """Spread of out-of-sample errors, centred on their median so the range always brackets the point forecast.

    (Uncentred residual quantiles would bake validation-period bias into the band while leaving the point
    forecast unchanged, which can put the forecast outside its own range.)"""
    res = actual_val - pred_val
    res = res - np.median(res, axis=0)
    return {"lo": np.quantile(res, qs[0], axis=0), "hi": np.quantile(res, qs[1], axis=0), "resid": res}


def backtest_strategy(pred1: np.ndarray, actual_logret1: np.ndarray, cost: float) -> dict:
    """Long/flat strategy on the 1-day forecast: hold when predicted return > 0. Costs per position change."""
    pos = (pred1 > 0).astype(float)
    simple = np.exp(actual_logret1) - 1
    turn = np.abs(np.diff(np.concatenate([[0.0], pos])))
    strat = pos * simple - turn * cost
    eq = np.cumprod(1 + strat)
    dd = eq / np.maximum.accumulate(eq) - 1
    return {"equity": eq, "total_return": float(eq[-1] - 1), "max_drawdown": float(dd.min()),
            "trades": int(turn.sum()), "days_long": int(pos.sum()),
            "hit_rate": float(np.mean(simple[pos == 1] > 0) * 100) if pos.sum() else None}


def evaluate(train_res, fs, cfg: dict) -> dict:
    """Build the full evaluation block (metrics, selection, intervals, backtest)."""
    plan, H = train_res.plan, train_res.plan.horizon
    horizons = [h for h in cfg.get("eval_horizons", [1, 5, 10, 21]) if h <= H]
    Y = fs.Y.values
    close = fs.close.values
    qs = cfg.get("interval_quantiles", [0.10, 0.90])
    cost = cfg.get("transaction_cost", 0.001)

    yv, yt = Y[plan.val], Y[plan.test]
    cv, ct = close[plan.val], close[plan.test]
    naive_v = horizon_metrics(train_res.pred_val["naive"], yv, cv, horizons)
    naive_t = horizon_metrics(train_res.pred_test["naive"], yt, ct, horizons)

    models, intervals = {}, {}
    for name in train_res.models:
        mv = horizon_metrics(train_res.pred_val[name], yv, cv, horizons)
        mt = horizon_metrics(train_res.pred_test[name], yt, ct, horizons)
        rel_v, rel_t = relative_mae(mv, naive_v), relative_mae(mt, naive_t)
        iv = residual_quantiles(train_res.pred_val[name], yv, qs)
        intervals[name] = iv
        # empirical coverage of the validation-calibrated interval on the untouched test block
        lo, hi = train_res.pred_test[name] + iv["lo"], train_res.pred_test[name] + iv["hi"]
        cover = {h: float(np.mean((yt[:, h - 1] >= lo[:, h - 1]) & (yt[:, h - 1] <= hi[:, h - 1])) * 100)
                 for h in horizons}
        models[name] = {"name": name, "family": FAMILY[name], "val": mv, "test": mt,
                        "val_rel_mae": rel_v, "test_rel_mae": rel_t, "coverage_test": cover}

    candidates = [m for m in train_res.models if FAMILY[m] != "Baseline"]
    best = min(candidates, key=lambda m: models[m]["val_rel_mae"])
    best_ml = min([m for m in candidates if FAMILY[m] == "ML"], key=lambda m: models[m]["val_rel_mae"])
    dl = [m for m in candidates if FAMILY[m] == "DL"]
    best_dl = min(dl, key=lambda m: models[m]["val_rel_mae"]) if dl else None
    ens = "ensemble" if "ensemble" in candidates else None

    b = models[best]
    beats_val = b["val_rel_mae"] < 1.0
    beats_test = b["test_rel_mae"] < 1.0
    dir_edge = (b["test"][1]["dir_acc"] or 0) - max(b["test"][1]["up_share"] or 50, 100 - (b["test"][1]["up_share"] or 50))
    if beats_val and beats_test and b["test_rel_mae"] < 0.97 and dir_edge > 2:
        reliability = "Moderate"
        why = ("The selected model beat the naive no-change baseline on both validation and untouched test data "
               "and its direction hit-rate exceeded the majority-class rate.")
    elif beats_test:
        reliability = "Low"
        why = ("The selected model only marginally beat the naive baseline on held-out data; treat the forecast "
               "as barely better than 'no change'.")
    else:
        reliability = "None demonstrated"
        why = ("The selected model did NOT beat the naive no-change baseline on held-out test data. "
               "Its forecast has no demonstrated skill over simply assuming the price stays the same.")

    # ---- backtest on the held-out test block
    test_dates = [str(fs.X.index[i + 1].date()) for i in plan.test]
    actual_next = np.exp(yt[:, 0]) * ct
    bt_models = {}
    for name in candidates:
        p1 = train_res.pred_test[name][:, 0]
        strat = backtest_strategy(p1, yt[:, 0], cost)
        bt_models[name] = {"pred_close": (ct * np.exp(p1)).tolist(), "equity": strat["equity"].tolist(),
                           "total_return": strat["total_return"], "max_drawdown": strat["max_drawdown"],
                           "trades": strat["trades"], "days_long": strat["days_long"], "hit_rate": strat["hit_rate"]}
    bh = np.cumprod(np.exp(yt[:, 0]))
    bh_dd = bh / np.maximum.accumulate(bh) - 1
    backtest = {
        "dates": test_dates, "actual_close": actual_next.tolist(),
        "naive_close": ct.tolist(), "models": bt_models,
        "buy_hold": {"equity": bh.tolist(), "total_return": float(bh[-1] - 1), "max_drawdown": float(bh_dd.min())},
        "cost": cost, "n": int(len(test_dates)),
        "note": ("Long/flat strategy on the model's 1-day forecast over data never used for fitting. Includes a "
                 f"{cost * 100:.2f}% cost per position change; ignores slippage, taxes and liquidity. "
                 "Past performance does not indicate future results."),
    }
    return {"models": models, "baselines": {"naive": {"val": naive_v, "test": naive_t},
                                              "drift": {"val": horizon_metrics(train_res.pred_val["drift"], yv, cv, horizons),
                                                        "test": horizon_metrics(train_res.pred_test["drift"], yt, ct, horizons)}},
            "horizons": horizons, "selected": best, "best_ml": best_ml, "best_dl": best_dl, "ensemble": ens,
            "reliability": reliability, "reliability_note": why, "beats_naive_val": beats_val,
            "beats_naive_test": beats_test, "backtest": backtest, "intervals": intervals}
