"""Forecast tracking: log every forecast, resolve it once the target date has passed, and report real accuracy."""
from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from .model_trainer import FAMILY, LABEL
from .store import Store

NAIVE = "naive"


def forecast_rows(result: dict, now_iso: str) -> List[Dict[str, Any]]:
    """Flatten an analysis result into log rows (all models + a naive 'no change' reference)."""
    f, meta = result["forecast"], result["meta"]
    base, base_date = f["base_close"], f["as_of"]
    rows = []
    for name, m in f["models"].items():
        for kind in ("next_day", "month_end"):
            b = m[kind]
            rows.append({"symbol": meta["symbol"], "model": name, "kind": kind, "made_at": now_iso,
                         "base_date": base_date, "base_close": base, "target_date": b["date"],
                         "pred_close": b["pred_close"], "low": b["low"], "high": b["high"],
                         "selected": int(name == f["selected"]), "currency": meta.get("currency")})
    first = next(iter(f["models"].values()))
    for kind in ("next_day", "month_end"):
        rows.append({"symbol": meta["symbol"], "model": NAIVE, "kind": kind, "made_at": now_iso,
                     "base_date": base_date, "base_close": base, "target_date": first[kind]["date"],
                     "pred_close": base, "low": None, "high": None, "selected": 0, "currency": meta.get("currency")})
    return rows


def resolve_pending(store: Store, symbol: str, df: pd.DataFrame, now_iso: str) -> int:
    """Fill in actual closes for forecasts whose target session has completed (``df`` = completed bars only)."""
    if df is None or df.empty:
        return 0
    n = 0
    idx, closes = df.index, df["Close"].values
    for r in store.pending(symbol):
        target = pd.Timestamp(r["target_date"])
        if idx[-1] < target:
            continue
        pos = idx.searchsorted(target)            # first completed session on/after the target date
        if pos >= len(idx) or (idx[pos] - target).days > 7:
            continue
        store.resolve(r["id"], float(closes[pos]), str(idx[pos].date()), now_iso)
        n += 1
    return n


def summarize(store: Store, symbol: Optional[str] = None, recent: int = 60) -> dict:
    rows = store.forecasts(symbol)
    done = [r for r in rows if r["actual_close"] is not None]
    pending = [r for r in rows if r["actual_close"] is None and r["model"] != NAIVE]
    naive_err = {(r["symbol"], r["kind"], r["base_date"]): abs(r["actual_close"] - r["pred_close"])
                 for r in done if r["model"] == NAIVE}
    groups = defaultdict(list)
    for r in done:
        groups[(r["kind"], r["model"])].append(r)
    table = []
    for (kind, model), rs in groups.items():
        err = np.array([x["pred_close"] - x["actual_close"] for x in rs])
        act = np.array([x["actual_close"] for x in rs])
        sgn_p = np.sign(np.array([x["pred_close"] - x["base_close"] for x in rs]))
        sgn_a = np.sign(np.array([x["actual_close"] - x["base_close"] for x in rs]))
        nz = (sgn_a != 0) & (sgn_p != 0)
        banded = [x for x in rs if x["low"] is not None]
        paired = [(abs(x["pred_close"] - x["actual_close"]), naive_err[(x["symbol"], x["kind"], x["base_date"])])
                  for x in rs if (x["symbol"], x["kind"], x["base_date"]) in naive_err]
        table.append({
            "kind": kind, "model": model, "label": LABEL.get(model, model), "family": FAMILY.get(model, ""),
            "n": len(rs), "mae": float(np.mean(np.abs(err))), "mape": float(np.mean(np.abs(err) / act) * 100),
            "bias_pct": float(np.mean(err / act) * 100),
            "dir_acc": float(np.mean(sgn_p[nz] == sgn_a[nz]) * 100) if nz.any() else None,
            "interval_hit": float(np.mean([x["low"] <= x["actual_close"] <= x["high"] for x in banded]) * 100) if banded else None,
            "vs_naive": float(1 - np.sum([p for p, _ in paired]) / max(np.sum([q for _, q in paired]), 1e-12)) * 100
            if paired and model != NAIVE else None,
        })
    table.sort(key=lambda t: (t["kind"], t["mae"] / max(1e-9, 1)))
    shown = [r for r in rows if r["model"] != NAIVE and (r["selected"] or r["actual_close"] is not None)]
    return {"symbol": symbol, "symbols": store.tracked_symbols(), "resolved": len(done) - len(naive_err),
            "pending": len(pending), "table": table,
            "recent": [{**r, "error_pct": (None if r["actual_close"] is None else
                                           (r["pred_close"] / r["actual_close"] - 1) * 100)} for r in shown[:recent]],
            "note": ("Every forecast the app makes is saved and compared with the real close once that date passes. "
                     "'vs naive' > 0 means the model was closer than assuming no change. Small samples are noisy.")}
