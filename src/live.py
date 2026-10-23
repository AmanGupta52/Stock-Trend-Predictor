"""Live price + fast next-session forecast.

Design (so a 5-second refresh stays cheap and honest):

* The quick model is trained ONCE per symbol on a chronological 80 % / 20 % split of completed daily bars
  (a further 15 % of the 80 % is used to choose between Ridge and a small gradient-boosting model, and to
  calibrate the 80 % range). The final 20 % is never used for fitting, selection or calibration.
* On every refresh only the newest price is added: the latest ~300 daily bars are kept in memory, the current
  price becomes a provisional bar for today, features are recomputed for the tail only and the already-trained
  model predicts. Nothing is re-downloaded or re-trained. If the price has not changed, the cached forecast is
  returned without any computation.
* Only features derived from the close are used, so an unfinished intraday bar cannot distort them
  (no volume / high-low features).
* Price feed = Yahoo Finance's latest 1-minute bar (with its own timestamp). It may be delayed and may not
  change every 5 seconds; the app shows the quote time and never calls it real-time.
"""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge

from .data_fetcher import StockData
from .data_processor import DataError
from .feature_engineer import build_features
from .model_trainer import Scaler
from .predictor import future_sessions

log = logging.getLogger(__name__)

LIVE_FEATURES = ["ret_1", "ret_2", "ret_3", "ret_5", "ret_10", "ret_20", "ret_60",
                 "close_sma5", "close_sma10", "close_sma20", "close_sma50", "sma20_sma50", "ema12_ema26",
                 "rsi", "macd", "macd_hist", "bb_pctb", "bb_width",
                 "vol_5", "vol_10", "vol_20", "vol_60", "vol_ratio", "dd_252", "zscore_20", "dow_sin", "dow_cos"]
MIN_LABELED = 200


def _estimators() -> Dict[str, Any]:
    return {
        "ridge": lambda: Ridge(alpha=30.0),
        "gradient_boosting": lambda: HistGradientBoostingRegressor(
            max_iter=120, learning_rate=0.05, max_depth=3, min_samples_leaf=20, l2_regularization=1.0, random_state=0),
    }


LABELS = {"ridge": "Ridge Regression", "gradient_boosting": "Gradient Boosting"}


def _dir(p: float, band: float = 0.02) -> str:
    return "Up" if p > band else "Down" if p < -band else "Flat"


@dataclass
class LiveState:
    symbol: str
    meta: dict
    name: str
    currency: str
    tail: pd.DataFrame                       # completed daily bars (raw + adjusted), newest last
    scaler: Scaler
    est: Any
    model_name: str
    q_lo: float
    q_hi: float
    metrics: dict
    trained_through: str
    trained_at: float
    n_train: int
    n_test: int
    train_seconds: float
    lock: threading.Lock = field(default_factory=threading.Lock)
    last_key: Optional[tuple] = None
    last_pred: Optional[dict] = None
    ticks: int = 0
    tail_checked: float = field(default_factory=time.time)
    refreshing: bool = False


def _provisional(tail: pd.DataFrame, date: pd.Timestamp, price: float) -> pd.DataFrame:
    """Append today's unfinished session as a one-price bar (only the close is used by the live features)."""
    ratio = float(tail["Adj Close"].iloc[-1] / tail["Close"].iloc[-1])
    row = pd.DataFrame({"Open": price, "High": price, "Low": price, "Close": price,
                        "Adj Close": price * ratio, "Volume": float(tail["Volume"].iloc[-1])}, index=[date])
    return pd.concat([tail, row])


def live_features(frame: pd.DataFrame) -> pd.Series:
    return build_features(frame)[LIVE_FEATURES].iloc[-1]


def _metrics(pred_ret: np.ndarray, y: np.ndarray, close_t: np.ndarray, q_lo: float, q_hi: float) -> dict:
    pred_px, act_px = close_t * np.exp(pred_ret), close_t * np.exp(y)
    err = pred_px - act_px
    mae, rmse = float(np.mean(np.abs(err))), float(np.sqrt(np.mean(err ** 2)))
    naive_mae = float(np.mean(np.abs(close_t - act_px)))
    nz = y != 0
    dir_acc = float(np.mean(np.sign(pred_ret[nz]) == np.sign(y[nz]))) if nz.any() else None
    up_rate = float(np.mean(y[nz] > 0)) if nz.any() else None
    lo, hi = close_t * np.exp(pred_ret + q_lo), close_t * np.exp(pred_ret + q_hi)
    return {"mae": mae, "rmse": rmse, "mape": float(np.mean(np.abs(err) / act_px) * 100),
            "naive_mae": naive_mae, "rel_mae": mae / naive_mae if naive_mae > 0 else None,
            "directional_accuracy": dir_acc, "up_day_rate": up_rate,
            "coverage_80": float(np.mean((act_px >= lo) & (act_px <= hi))), "n": int(len(y))}


def train_quick_model(data: StockData, cfg: dict) -> LiveState:
    """Fit the quick next-session model on the first 80 % (chronological) and score it on the last 20 %."""
    t0 = time.time()
    lc = cfg.get("live", {})
    df = data.df
    feats = build_features(df)[LIVE_FEATURES]
    adj = df["Adj Close"]
    y_all = np.log(adj.shift(-1) / adj)
    ok = feats.notna().all(axis=1)
    X, y = feats[ok], y_all[ok]
    labeled = y.notna()
    Xl, yl = X[labeled], y[labeled]
    n = len(Xl)
    if n < MIN_LABELED:
        raise DataError(f"Only {n} usable trading days after warm-up; the quick model needs at least {MIN_LABELED}.",
                        "insufficient_history")
    test_frac = float(lc.get("test_fraction", 0.20))
    n_train = int(n * (1 - test_frac))
    test_start = n_train + 1                                # 1-row embargo: a training target never touches test rows
    val_start = int(n_train * 0.85)
    A, v = Xl.values, yl.values
    fit_i, val_i = np.arange(0, val_start - 1), np.arange(val_start, n_train)
    all_train, test_i = np.arange(0, n_train), np.arange(test_start, n)
    # model choice + calibration use the validation slice of the training block only
    sc_in = Scaler().fit(A[fit_i])
    best, best_mae, best_res = None, np.inf, None
    for name, mk in _estimators().items():
        est = mk().fit(sc_in.transform(A[fit_i]), v[fit_i])
        res = v[val_i] - est.predict(sc_in.transform(A[val_i]))
        mae = float(np.mean(np.abs(res)))
        if mae < best_mae:
            best, best_mae, best_res = name, mae, res
    q_lo, q_hi = (float(np.quantile(best_res, 0.10)), float(np.quantile(best_res, 0.90)))
    # refit the chosen model on the complete 80 % block (scaler too)
    sc = Scaler().fit(A[all_train])
    est = _estimators()[best]().fit(sc.transform(A[all_train]), v[all_train])
    pred_test = est.predict(sc.transform(A[test_i]))
    close_t = df["Close"].reindex(Xl.index[test_i]).values
    metrics = _metrics(pred_test, v[test_i], close_t, q_lo, q_hi)
    metrics.update({"train_start": str(Xl.index[0].date()), "train_end": str(Xl.index[n_train - 1].date()),
                    "test_start": str(Xl.index[test_start].date()), "test_end": str(Xl.index[n - 1].date())})
    rel = metrics["rel_mae"]
    metrics["beats_naive"] = bool(rel is not None and rel < 1.0)
    metrics["verdict"] = (
        "On the held-out 20 % the quick model was closer than the 'tomorrow = today' baseline."
        if metrics["beats_naive"] else
        "On the held-out 20 % the quick model did NOT beat the simple 'tomorrow = today' baseline; "
        "treat its forecast as no better than the current price.")
    tail = df.tail(int(lc.get("tail_rows", 300))).copy()
    return LiveState(symbol=data.symbol, meta=data.meta,
                     name=data.meta.get("longName") or data.meta.get("shortName") or data.symbol,
                     currency=data.meta.get("currency") or "", tail=tail, scaler=sc, est=est, model_name=best,
                     q_lo=q_lo, q_hi=q_hi, metrics=metrics, trained_through=str(df.index[-1].date()),
                     trained_at=time.time(), n_train=int(n_train), n_test=int(len(test_i)),
                     train_seconds=time.time() - t0)


class LiveEngine:
    def __init__(self, fetcher, cfg: dict):
        self.fetcher, self.cfg = fetcher, cfg
        self.lc = cfg.get("live", {})
        self.states: Dict[str, LiveState] = {}
        self._quotes: Dict[str, tuple] = {}        # symbol -> (monotonic ts, quote dict)
        self._qlock = threading.Lock()

    # ---- training ---------------------------------------------------------------
    def start(self, query: str, progress=None) -> dict:
        progress = progress or (lambda p, m: None)
        progress(0.05, "Resolving symbol and loading daily history")
        data = self.fetcher.load(query, "5y", min_rows=MIN_LABELED + 80)
        progress(0.45, "Training the quick model on the first 80 % of history")
        st = train_quick_model(data, self.cfg)
        self.states[st.symbol] = st
        progress(1.0, "Done")
        return self.summary(st)

    def summary(self, st: LiveState) -> dict:
        return {"symbol": st.symbol, "name": st.name, "currency": st.currency, "model": LABELS[st.model_name],
                "metrics": st.metrics, "trained_through": st.trained_through, "n_train": st.n_train,
                "n_test": st.n_test, "train_seconds": round(st.train_seconds, 2),
                "poll_seconds": int(self.lc.get("poll_seconds", 5))}

    # ---- price feed (shared, throttled) --------------------------------------------
    def _quote(self, symbol: str) -> tuple:
        min_gap = float(self.lc.get("min_quote_interval_seconds", 4))
        with self._qlock:
            hit = self._quotes.get(symbol)
            if hit and time.monotonic() - hit[0] < min_gap:
                return hit[1], False
            try:
                q = self.fetcher.fetch_quote(symbol)
            except DataError:
                if hit:
                    return hit[1], True               # keep serving the last good quote, flagged stale
                raise
            self._quotes[symbol] = (time.monotonic(), q)
            return q, False

    def _maybe_refresh_tail(self, st: LiveState) -> None:
        if st.refreshing or time.time() - st.tail_checked < float(self.lc.get("tail_refresh_seconds", 600)):
            return
        st.refreshing = True

        def work():
            try:
                d = self.fetcher.fetch_symbol(st.symbol, "1y", force=True, min_rows=60)
                if d.df.index[-1] > st.tail.index[-1]:
                    with st.lock:
                        st.tail = d.df.tail(int(self.lc.get("tail_rows", 300))).copy()
                        st.last_key = None
            except Exception as exc:                  # a failed top-up must never break the live feed
                log.info("daily tail refresh failed for %s: %s", st.symbol, exc)
            finally:
                st.tail_checked, st.refreshing = time.time(), False
        threading.Thread(target=work, daemon=True).start()

    # ---- one refresh --------------------------------------------------------------
    def tick(self, symbol: str) -> dict:
        st = self.states.get(symbol)
        if st is None:
            raise DataError("The quick model for this stock is not trained yet. Press Start live first.", "not_trained")
        t0 = time.perf_counter()
        q, stale = self._quote(symbol)
        self._maybe_refresh_tail(st)
        price = float(q["price"])
        with st.lock:
            last_bar = st.tail.index[-1]
            qdate = pd.Timestamp(q["date"])
            provisional = qdate > last_bar
            key = (round(price, 6), str(last_bar.date()), str(qdate.date()))
            computed = key != st.last_key
            if computed:                                # only when the price (or the daily tail) changed
                frame = _provisional(st.tail, qdate, price) if provisional else st.tail
                x = live_features(frame)
                if x.isna().any():
                    raise DataError("Not enough recent history to build features for the live forecast.", "insufficient_history")
                r = float(st.est.predict(st.scaler.transform(x.values.reshape(1, -1)))[0])
                base_date = frame.index[-1]
                nxt, _ = future_sessions(base_date, 1, st.symbol, st.meta)
                st.last_pred = {
                    "pred_close": price * float(np.exp(r)), "low": price * float(np.exp(r + st.q_lo)),
                    "high": price * float(np.exp(r + st.q_hi)), "change_pct": (float(np.exp(r)) - 1) * 100,
                    "direction": _dir((float(np.exp(r)) - 1) * 100), "target_date": str(nxt[0].date()),
                    "base_price": price, "model": LABELS[st.model_name]}
                st.last_key = key
            st.ticks += 1
            pred = dict(st.last_pred)
        last_close = float(st.tail["Close"].iloc[-1])
        ref = last_close if provisional else float(st.tail["Close"].iloc[-2])
        age_min = None
        try:
            age_min = max(0.0, (pd.Timestamp.now(tz="UTC") - pd.Timestamp(q["time_utc"])).total_seconds() / 60)
        except Exception:
            pass
        max_age = float(self.lc.get("max_quote_age_minutes", 20))
        return {
            "symbol": symbol, "name": st.name, "currency": st.currency, "price": price,
            "quote_time": q["time"], "quote_age_minutes": age_min, "quote_stale": bool(age_min is not None and age_min > max_age),
            "feed_error": bool(stale), "source": q.get("source"),
            "last_close": last_close, "last_close_date": str(st.tail.index[-1].date()),
            "change_abs": price - ref, "change_pct": (price / ref - 1) * 100,
            "change_label": "vs last completed close" if provisional else "vs prior close",
            "session_in_progress": bool(provisional),
            "prediction": pred, "recomputed": bool(computed),
            "model": {"trained_through": st.trained_through, "trained_at": st.trained_at, "ticks_since_training": st.ticks,
                      "name": LABELS[st.model_name]},
            "compute_ms": round((time.perf_counter() - t0) * 1000, 1),
            "freshness": ("Latest price Yahoo Finance returned (its own timestamp is shown); it may be delayed and "
                          "is not confirmed real-time."),
        }
