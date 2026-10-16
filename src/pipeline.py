"""End-to-end orchestration: data -> features -> models -> evaluation -> forecast -> JSON payload."""
from __future__ import annotations

import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd

from . import __version__
from .data_fetcher import StockData, StockDataFetcher
from .data_processor import DataError
from .evaluator import evaluate, relative_mae
from .feature_engineer import make_feature_set
from .model_trainer import FAMILY, LABEL, make_splits, train_all
from .predictor import build_forecasts, future_sessions
from .technical_indicators import compute_indicators, trend_summary
from .utils import ConfigLoader

log = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parent.parent
DISCLAIMER = ("Forecasts are statistical estimates produced by models trained on past prices. They are not "
              "guaranteed outcomes, not investment advice, and markets can move far outside the shown ranges. "
              "Do not rely on them for trading decisions.")


def load_config(path: Optional[str] = None) -> dict:
    cfg = ConfigLoader(str(ROOT / "config")).load_yaml_config("config.yaml")
    if not cfg:
        raise RuntimeError("config/config.yaml could not be loaded")
    return cfg


def clean(o):
    """Recursively convert numpy/pandas values to JSON-safe Python (NaN/inf -> None)."""
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, np.ndarray):
        return clean(o.tolist())
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, (pd.Timestamp,)):
        return o.isoformat()
    return o


def _round(a, nd=4):
    return [None if (v is None or not np.isfinite(v)) else round(float(v), nd) for v in a]


class Analyzer:
    def __init__(self, cfg: Optional[dict] = None, base_dir: Optional[Path] = None):
        self.cfg = cfg or load_config()
        d, self.m = self.cfg["data"], self.cfg["modeling"]
        self.base = Path(base_dir) if base_dir else ROOT
        cache = self.base / d.get("cache_dir", ".cache")
        self.fetcher = StockDataFetcher(str(cache), d.get("data_cache_ttl_minutes", 60), d.get("min_rows", 300),
                                        d.get("stale_after_days", 10))
        self.result_dir = cache / "results"
        self.result_dir.mkdir(parents=True, exist_ok=True)

    # ---- result cache -----------------------------------------------------------
    def _key(self, data: StockData, period: str) -> str:
        blob = json.dumps([__version__, data.symbol, period, str(data.df.index[-1].date()), len(data.df),
                           round(float(data.df["Close"].iloc[-1]), 4), self.m, self.cfg["backtest"]], sort_keys=True)
        return hashlib.sha1(blob.encode()).hexdigest()

    def _result_path(self, symbol, period):
        return self.result_dir / f"{symbol.replace('/', '_')}__{period}.json"

    # ---- main entry -------------------------------------------------------------
    def analyze(self, query: str, period: str = "5y", force_refresh: bool = False, force_retrain: bool = False,
                progress: Optional[Callable[[float, str], None]] = None, data: Optional[StockData] = None) -> dict:
        progress = progress or (lambda p, m: None)
        t0 = time.time()
        progress(0.02, "Resolving symbol and downloading price history from Yahoo Finance")
        data = data or self.fetcher.load(query, period, force=force_refresh)
        progress(0.10, f"Loaded {len(data.df)} daily sessions for {data.symbol}")

        key = self._key(data, period)
        rp = self._result_path(data.symbol, period)
        if rp.exists() and not force_retrain:
            try:
                cached = json.loads(rp.read_text())
                if cached.get("cache_key") == key:
                    cached["run"] = {**cached.get("run", {}), "models_from_cache": True,
                                     "note": "Data unchanged since the last training run; reusing trained results. "
                                             "Use 'Force retrain' to train again."}
                    # data-level fields reflect this request
                    cached["data"].update(self._data_block(data))
                    cached["warnings"] = self._warnings(data, cached.get("model_warnings", []))
                    progress(1.0, "Done (cached models)")
                    return cached
            except Exception as exc:
                log.warning("ignoring bad result cache: %s", exc)

        result = self._compute(data, period, progress)
        result["cache_key"] = key
        result["run"]["seconds"] = round(time.time() - t0, 1)
        result["run"]["models_from_cache"] = False
        try:
            rp.write_text(json.dumps(clean(result)))
        except Exception as exc:
            log.warning("could not write result cache: %s", exc)
        progress(1.0, "Done")
        return clean(result)

    # ---- blocks -----------------------------------------------------------------
    def _warnings(self, data: StockData, model_warnings):
        return list(data.warnings) + list(model_warnings)

    def _data_block(self, data: StockData) -> dict:
        df, meta = data.df, data.meta
        last_close = float(df["Close"].iloc[-1])
        prev = float(df["Close"].iloc[-2])
        quote, tz = None, meta.get("exchangeTimezoneName", "UTC")
        qt_iso = None
        try:
            if meta.get("regularMarketTime") is not None:
                qt_iso = pd.Timestamp(int(meta["regularMarketTime"]), unit="s", tz="UTC").tz_convert(tz).isoformat()
        except Exception:
            qt_iso = None
        if data.partial_bar:
            price = meta.get("regularMarketPrice")
            price = float(price) if price is not None else data.partial_bar["price"]
            quote = {"price": price, "time": qt_iso or data.partial_bar["date"],
                     "kind": "session in progress / just closed"}
            day_change, change_label = (price / last_close - 1) * 100, "vs last completed close"
        else:
            if meta.get("regularMarketPrice") is not None and qt_iso:
                quote = {"price": float(meta["regularMarketPrice"]), "time": qt_iso, "kind": "last regular-market price"}
            day_change, change_label = (last_close / prev - 1) * 100, "vs prior close"
        return {
            "provider": "Yahoo Finance (via yfinance)",
            "rows": int(len(df)), "start": str(df.index[0].date()), "end": str(df.index[-1].date()),
            "last_close": last_close, "last_close_date": str(df.index[-1].date()),
            "day_change_pct": day_change, "day_change_label": change_label, "latest_quote": quote,
            "fetched_at": data.fetched_at, "from_data_cache": data.from_cache, "stale_cache": data.stale_cache,
            "freshness": ("Delayed / latest-available data from Yahoo Finance (end-of-day bars; quotes may be "
                          "delayed by up to ~15 min or more depending on the exchange). Not confirmed real-time."),
            "interpreted_from": data.interpreted_from, "alternatives": data.alternatives,
        }

    def _compute(self, data: StockData, period: str, progress) -> dict:
        df, meta, m = data.df, data.meta, self.m
        H, L = m["horizon"], m["seq_len"]
        progress(0.12, "Computing indicators and leak-free features")
        ohlcv = df[["Open", "High", "Low", "Close", "Volume"]]
        ind = compute_indicators(ohlcv)
        analysis = trend_summary(df, ind)
        fs = make_feature_set(df, H, L)
        plan = make_splits(fs.n_labeled, L, H, m["split"]["train"], m["split"]["val"])

        tr = train_all(fs.X.values, fs.Y.values, fs.feature_names, plan, m, progress)
        progress(0.88, "Evaluating on held-out data and building forecasts")
        ev = evaluate(tr, fs, {**m, **self.cfg["backtest"]})

        last_close = float(df["Close"].iloc[-1])
        fdates, hol_ok = future_sessions(df.index[-1], H, data.symbol, meta)
        include = [n for n in tr.models if FAMILY[n] != "Baseline"]
        fc = build_forecasts(tr.pred_final, ev["intervals"], last_close, fdates, include)

        mw = list(tr.notes)
        if not hol_ok:
            mw.append("Exchange holidays could not be modelled (install the 'holidays' package); forecast dates "
                      "assume Monday-Friday sessions and may shift around market holidays.")
        if len(plan.val) < 100 or len(plan.test) < 100:
            mw.append(f"Small evaluation blocks (validation {len(plan.val)}, test {len(plan.test)} samples): metrics are "
                      "noisy. Use 5y or longer history for more reliable evaluation.")
        mw.append("Forecast prices are applied to the latest completed close; models are trained on dividend/"
                  "split-adjusted prices.")
        if ev["reliability"] == "None demonstrated":
            mw.append("The best model did not beat the naive 'no change' baseline on held-out test data.")

        def metrics_rows(split):
            rows = []
            for name in ["naive", "drift"] + [n for n in tr.models if FAMILY[n] != "Baseline"]:
                if name in ("naive", "drift"):
                    hm = ev["baselines"][name][split]
                    rel = 1.0 if name == "naive" else relative_mae(hm, ev["baselines"]["naive"][split])
                else:
                    hm = ev["models"][name][split]
                    rel = ev["models"][name][f"{split}_rel_mae"]
                rows.append({"model": name, "label": LABEL[name], "family": FAMILY[name], "rel_mae": rel,
                             "horizons": {str(h): v for h, v in hm.items()}})
            return rows

        hist = {
            "dates": [str(d.date()) for d in df.index],
            "open": _round(df["Open"], 2), "high": _round(df["High"], 2), "low": _round(df["Low"], 2),
            "close": _round(df["Close"], 2), "adj_close": _round(df["Adj Close"], 2),
            "volume": [int(v) for v in df["Volume"]],
        }
        for col in ["sma_20", "sma_50", "sma_200", "ema_12", "ema_26", "rsi_14", "macd", "macd_signal", "macd_hist",
                    "bb_up", "bb_low", "volatility_20d"]:
            hist[col] = _round(ind[col], 4)
        cut = max(0, len(df) - 3000)       # cap payload size
        hist = {k: v[cut:] for k, v in hist.items()}

        return {
            "meta": {
                "symbol": data.symbol,
                "name": meta.get("longName") or meta.get("shortName") or data.symbol,
                "exchange": meta.get("fullExchangeName") or meta.get("exchangeName") or "Unknown",
                "exchange_code": meta.get("exchangeName"), "currency": meta.get("currency") or "",
                "instrument_type": meta.get("instrumentType"), "timezone": meta.get("exchangeTimezoneName"),
            },
            "period": period,
            "data": self._data_block(data),
            "history": hist,
            "analysis": analysis,
            "forecast": {
                "as_of": str(df.index[-1].date()), "base_close": last_close, "horizon": H, "models": fc,
                "selected": ev["selected"], "best_ml": ev["best_ml"], "best_dl": ev["best_dl"],
                "reliability": ev["reliability"], "reliability_note": ev["reliability_note"],
                "interval_label": f"{int((m['interval_quantiles'][1] - m['interval_quantiles'][0]) * 100)}% empirical range",
                "holidays_modelled": hol_ok,
                "next_session_in_progress": bool(data.partial_bar and data.partial_bar["date"] == str(fdates[0].date())),
            },
            "evaluation": {
                "horizons": ev["horizons"], "val": metrics_rows("val"), "test": metrics_rows("test"),
                "coverage_test": {n: ev["models"][n]["coverage_test"] for n in include},
                "split": {"train": int(len(plan.train)), "val": int(len(plan.val)), "test": int(len(plan.test)),
                          "embargo": H,
                          "train_range": [str(fs.X.index[plan.train[0]].date()), str(fs.X.index[plan.train[-1]].date())],
                          "val_range": [str(fs.X.index[plan.val[0]].date()), str(fs.X.index[plan.val[-1]].date())],
                          "test_range": [str(fs.X.index[plan.test[0]].date()), str(fs.X.index[plan.test[-1]].date())]},
                "selection_rule": ("Best model = lowest mean over horizons of MAE(model)/MAE(naive) on the "
                                   "VALIDATION block. Test metrics are computed on later data never used for "
                                   "fitting or selection."),
                "beats_naive_val": ev["beats_naive_val"], "beats_naive_test": ev["beats_naive_test"],
                "feature_importance": tr.feature_importance,
                "lstm": {"backend": tr.lstm_backend, "epochs": tr.lstm_epochs},
            },
            "backtest": ev["backtest"],
            "model_warnings": mw,
            "warnings": self._warnings(data, mw),
            "disclaimer": DISCLAIMER,
            "run": {"features": len(fs.feature_names), "samples": int(len(plan.all)), "version": __version__},
        }
