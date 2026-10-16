"""Turns model outputs (cumulative log returns) into price forecasts with uncertainty ranges."""
from __future__ import annotations

import logging
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from .model_trainer import FAMILY, LABEL

log = logging.getLogger(__name__)

US_EXCHANGES = {"NMS", "NYQ", "NGM", "NCM", "ASE", "PCX", "BTS", "NasdaqGS", "NYSE", "NasdaqCM", "NasdaqGM"}


def _holiday_calendar(symbol: str, meta: dict, years: List[int]):
    """Return (set of holiday dates, modelled?) using the optional `holidays` package."""
    try:
        import holidays
    except ImportError:
        return set(), False
    exch = meta.get("exchangeName", "")
    try:
        if symbol.endswith((".NS", ".BO")):
            cal = holidays.financial_holidays("XNSE", years=years)
        elif exch in US_EXCHANGES or "." not in symbol:
            cal = holidays.financial_holidays("NYSE", years=years)
        else:
            return set(), False
        return {pd.Timestamp(d) for d in cal.keys()}, True
    except Exception as exc:
        log.info("holiday calendar unavailable: %s", exc)
        return set(), False


def future_sessions(last_date: pd.Timestamp, n: int, symbol: str = "", meta: dict = None) -> Tuple[List[pd.Timestamp], bool]:
    """Next n weekday sessions after last_date, skipping exchange holidays when a calendar is available."""
    years = list(range(last_date.year, last_date.year + 2))
    hol, modelled = _holiday_calendar(symbol, meta or {}, years)
    out, d = [], last_date
    while len(out) < n:
        d = d + pd.Timedelta(days=1)
        if d.weekday() < 5 and d not in hol:
            out.append(d)
    return out, modelled


def _direction(pct: float, flat_band: float = 0.05) -> str:
    return "Up" if pct > flat_band else "Down" if pct < -flat_band else "Flat"


def build_forecasts(pred_final: Dict[str, np.ndarray], intervals: dict, last_close: float,
                    future_dates: List[pd.Timestamp], include: List[str]) -> Dict[str, dict]:
    """Per-model next-day and 21-session forecasts with empirical 80% ranges."""
    H = len(future_dates)
    out = {}
    for name in include:
        p = pred_final[name]
        iv = intervals.get(name)
        if iv is None:      # baselines have no residual calibration here
            continue
        lo_c, hi_c = p + iv["lo"], p + iv["hi"]
        res = iv["resid"]

        def block(h):
            i = h - 1
            price = last_close * float(np.exp(p[i]))
            pct = (price / last_close - 1) * 100
            return {
                "date": str(future_dates[i].date()), "horizon": h,
                "pred_close": price, "change_pct": pct, "direction": _direction(pct),
                "low": last_close * float(np.exp(lo_c[i])), "high": last_close * float(np.exp(hi_c[i])),
                "prob_up": float(np.mean(p[i] + res[:, i] > 0) * 100),
            }

        out[name] = {
            "label": LABEL[name], "family": FAMILY[name],
            "next_day": block(1),
            "month_end": block(H),
            "path": {
                "dates": [str(d.date()) for d in future_dates],
                "pred": (last_close * np.exp(p[:H])).tolist(),
                "low": (last_close * np.exp(lo_c[:H])).tolist(),
                "high": (last_close * np.exp(hi_c[:H])).tolist(),
            },
        }
    return out
