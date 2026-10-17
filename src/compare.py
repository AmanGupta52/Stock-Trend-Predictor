"""Side-by-side comparison of 2-4 stocks: rebased prices, returns, volatility, drawdown, correlation."""
from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd

from .data_fetcher import StockDataFetcher
from .data_processor import DataError

ANN = 252


def compare_stocks(fetcher: StockDataFetcher, queries: List[str], period: str, force: bool = False) -> dict:
    qs = [q.strip() for q in queries if q and q.strip()]
    if not 2 <= len(qs) <= 4:
        raise DataError("Choose between 2 and 4 stocks to compare.", "bad_request")
    loaded, seen = [], set()
    for q in qs:
        d = fetcher.load(q, period, force=force, min_rows=60)
        if d.symbol in seen:
            raise DataError(f"{d.symbol} was selected twice.", "bad_request")
        seen.add(d.symbol)
        loaded.append(d)
    px = pd.concat({d.symbol: d.df["Adj Close"] for d in loaded}, axis=1).dropna()
    if len(px) < 60:
        raise DataError(f"These stocks share only {len(px)} trading days in this period; pick a longer period "
                        "or stocks listed for the same time span.", "insufficient_history")
    rebased = px / px.iloc[0] * 100
    rets = np.log(px).diff().dropna()
    dd = px / px.cummax() - 1
    stats = []
    for d in loaded:
        s, r = px[d.symbol], rets[d.symbol]
        n = len(r)
        total = float(s.iloc[-1] / s.iloc[0] - 1)
        vol = float(r.std() * np.sqrt(ANN))
        ann = float((s.iloc[-1] / s.iloc[0]) ** (ANN / n) - 1)

        def ret(k):
            return float(s.iloc[-1] / s.iloc[-1 - k] - 1) if n > k else None
        stats.append({
            "symbol": d.symbol, "name": d.meta.get("longName") or d.meta.get("shortName") or d.symbol,
            "exchange": d.meta.get("fullExchangeName") or d.meta.get("exchangeName") or "", "currency": d.meta.get("currency") or "",
            "last_close": float(d.df["Close"].iloc[-1]), "last_date": str(d.df.index[-1].date()),
            "total_return": total, "annual_return": ann, "volatility": vol,
            "sharpe": float(r.mean() * ANN / vol) if vol > 0 else None,
            "max_drawdown": float(dd[d.symbol].min()), "current_drawdown": float(dd[d.symbol].iloc[-1]),
            "best_day": float(np.exp(r.max()) - 1), "worst_day": float(np.exp(r.min()) - 1),
            "ret_1m": ret(21), "ret_3m": ret(63), "ret_6m": ret(126), "ret_1y": ret(252),
        })
    best = max(stats, key=lambda x: x["total_return"])["symbol"]
    lowest_vol = min(stats, key=lambda x: x["volatility"])["symbol"]
    notes = ["Prices use dividend/split-adjusted closes, rebased to 100 at the first common date; "
             "different currencies are compared as percentage changes only.",
             "Past performance does not predict future returns."]
    for d in loaded:
        notes.extend(w for w in d.warnings if "Interpreted" in w or "using" in w)
    return {"period": period, "start": str(px.index[0].date()), "end": str(px.index[-1].date()), "days": int(len(px)),
            "dates": [str(x.date()) for x in px.index],
            "rebased": {k: [round(float(v), 3) for v in rebased[k]] for k in rebased},
            "drawdown": {k: [round(float(v) * 100, 3) for v in dd[k]] for k in dd},
            "stats": stats, "best_return": best, "lowest_volatility": lowest_vol,
            "correlation": {"symbols": list(rets.columns), "matrix": rets.corr().round(3).values.tolist()},
            "notes": notes}
