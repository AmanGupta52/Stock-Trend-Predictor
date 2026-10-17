"""Volatility forecasting: how much the stock is expected to swing, not which way.

Target: annualised realised volatility (std of daily log returns x sqrt(252)) over the NEXT h sessions,
for h in 5, 10, 21, modelled on a log scale. Same chronological split/embargo as the price models.
Baselines: persistence (trailing 20-session realised vol) and RiskMetrics EWMA (lambda = 0.94).
"""
from __future__ import annotations

import logging
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from .model_trainer import LABEL, Scaler, SplitPlan, build_model, HAS_XGB

log = logging.getLogger(__name__)
ANN = np.sqrt(252.0)
VOL_H = (5, 10, 21)
VOL_MODELS = ("ridge", "random_forest", "gradient_boosting", "xgboost")
Z80 = 1.2816  # two-sided 80% normal quantile
LABEL = {**LABEL, "persistence": "Persistence (last 20-day vol)", "ewma": "EWMA (RiskMetrics, λ=0.94)"}


def realised_targets(close: pd.Series, horizons=VOL_H) -> pd.DataFrame:
    """log of annualised realised vol over t+1..t+h (NaN where the future is unknown)."""
    lr = np.log(close).diff()
    return pd.DataFrame({f"v{h}": np.log(np.maximum(lr.rolling(h).std().shift(-h) * ANN, 1e-4))
                         for h in horizons}, index=close.index)


def ewma_vol(close: pd.Series, lam: float = 0.94) -> pd.Series:
    lr = np.log(close).diff().fillna(0.0)
    return np.sqrt(lr.pow(2).ewm(alpha=1 - lam, adjust=False).mean()) * ANN


def _metrics(pred_vol: np.ndarray, actual_vol: np.ndarray, base: Dict[str, np.ndarray]) -> Dict[int, dict]:
    out = {}
    for j, h in enumerate(VOL_H):
        p, a = pred_vol[:, j], actual_vol[:, j]
        mae = float(np.mean(np.abs(p - a)))
        out[h] = {"mae": mae * 100, "mape": float(np.mean(np.abs(p - a) / a) * 100),
                  "corr": float(np.corrcoef(p, a)[0, 1]) if np.std(p) > 0 and np.std(a) > 0 else None,
                  **{f"rel_{k}": mae / max(float(np.mean(np.abs(v[:, j] - a))), 1e-12) for k, v in base.items()}}
    return out


def _mean_rel(m: Dict[int, dict], key: str) -> float:
    return float(np.mean([m[h][key] for h in m]))


def run_volatility(X: pd.DataFrame, close: pd.Series, plan: SplitPlan, cfg: dict, last_close: float,
                   progress: Optional[Callable[[float, str], None]] = None) -> dict:
    """Train, evaluate and forecast volatility. X/close are aligned (feature rows incl. the latest session)."""
    progress = progress or (lambda p, m: None)
    F_raw = X.values
    n_lab = int(plan.all.max()) + 1
    T = realised_targets(close).reindex(X.index).values[:n_lab]          # log-vol targets, (n_lab, 3)
    pers = np.tile((X["vol_20"].values * ANN)[:, None], (1, len(VOL_H)))  # persistence baseline, all rows
    ew = np.tile(ewma_vol(close).reindex(X.index).values[:, None], (1, len(VOL_H)))
    act = lambda idx: np.exp(T[idx])
    names = [m for m in cfg.get("vol_models", VOL_MODELS) if m != "xgboost" or HAS_XGB]
    out = {"horizons": list(VOL_H), "notes": []}
    pv, pt, pf = {}, {}, {}
    for k, name in enumerate(names):
        try:
            progress(0.90 + 0.06 * k / len(names), f"Training volatility model: {LABEL[name]}")
            F = Scaler().fit(F_raw[plan.train]).transform(F_raw)
            m = build_model(name, cfg, params={"alpha": 10.0} if name == "ridge" else None).fit(F, plan.train, T[plan.train])
            pv[name] = np.exp(m.predict(F, plan.val))
            F = Scaler().fit(F_raw[plan.fit_tv]).transform(F_raw)
            m = build_model(name, cfg, params={"alpha": 10.0} if name == "ridge" else None).fit(F, plan.fit_tv, T[plan.fit_tv])
            pt[name] = np.exp(m.predict(F, plan.test))
            F = Scaler().fit(F_raw[plan.all]).transform(F_raw)
            m = build_model(name, cfg, params={"alpha": 10.0} if name == "ridge" else None).fit(F, plan.all, T[plan.all])
            pf[name] = np.exp(m.predict(F, np.array([len(F_raw) - 1]))[0])
        except Exception as exc:
            log.warning("volatility model %s failed: %s", name, exc)
            out["notes"].append(f"Volatility model {LABEL[name]} skipped: {type(exc).__name__}: {exc}")
    if not pv:
        return {"error": "No volatility model could be trained.", **out}

    yv, yt = act(plan.val), act(plan.test)
    bv = {"persistence": pers[plan.val], "ewma": ew[plan.val]}
    bt = {"persistence": pers[plan.test], "ewma": ew[plan.test]}
    val = {n: _metrics(pv[n], yv, bv) for n in pv}
    test = {n: _metrics(pt[n], yt, bt) for n in pv}
    base_val = {"persistence": _metrics(bv["persistence"], yv, bv), "ewma": _metrics(bv["ewma"], yv, bv)}
    base_test = {"persistence": _metrics(bt["persistence"], yt, bt), "ewma": _metrics(bt["ewma"], yt, bt)}
    selected = min(pv, key=lambda n: _mean_rel(val[n], "rel_persistence"))
    beats_test = _mean_rel(test[selected], "rel_persistence") < 1.0

    def rows(split_metrics, base_metrics):
        r = [{"model": n, "label": LABEL[n], "family": "Baseline", "rel_persistence": 1.0 if n == "persistence" else None,
              "rel_ewma": _mean_rel(base_metrics[n], "rel_ewma"), "horizons": {str(h): v for h, v in base_metrics[n].items()}}
             for n in ("persistence", "ewma")]
        r[0]["rel_ewma"] = _mean_rel(base_metrics["persistence"], "rel_ewma")
        r[1]["rel_persistence"] = _mean_rel(base_metrics["ewma"], "rel_persistence")
        for n, m in split_metrics.items():
            r.append({"model": n, "label": LABEL[n], "family": "ML", "rel_persistence": _mean_rel(m, "rel_persistence"),
                      "rel_ewma": _mean_rel(m, "rel_ewma"), "horizons": {str(h): v for h, v in m.items()}})
        return r

    sel_f = pf[selected]
    sigma_d = sel_f[-1] / ANN
    out.update({
        "current": {"realised_20d": float(X["vol_20"].iloc[-1] * ANN * 100), "realised_5d": float(X["vol_5"].iloc[-1] * ANN * 100),
                    "ewma": float(ewma_vol(close).iloc[-1] * 100)},
        "forecast": {n: {str(h): float(v[j] * 100) for j, h in enumerate(VOL_H)} for n, v in pf.items()},
        "selected": selected,
        "expected_daily_move_pct": float(sigma_d * 100),
        "month_range": {"low": last_close * float(np.exp(-Z80 * sigma_d * np.sqrt(21))),
                        "high": last_close * float(np.exp(Z80 * sigma_d * np.sqrt(21))),
                        "pct": float(Z80 * sigma_d * np.sqrt(21) * 100)},
        "val": rows(val, base_val), "test": rows(test, base_test),
        "beats_persistence_test": bool(beats_test),
        "verdict": ("The selected volatility model beat the 'volatility stays the same' baseline on unseen test data."
                    if beats_test else
                    "The selected volatility model did NOT beat the simple 'volatility stays the same' baseline on "
                    "unseen test data; treat its forecast as no better than the current 20-day volatility."),
        "history": {"dates": [str(d.date()) for d in X.index[-260:]],
                    "realised_20d": [round(float(v) * ANN * 100, 2) for v in X["vol_20"].iloc[-260:]]},
        "note": ("Volatility is annualised, in %. The 80% range uses the selected model's 21-session volatility and a "
                 "normal approximation (real returns have fatter tails). Estimates, not guarantees."),
    })
    return out
