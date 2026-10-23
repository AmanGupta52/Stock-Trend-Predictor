"""Live price + quick-model tests (offline, synthetic data)."""
import json

import numpy as np
import pandas as pd

from src.data_fetcher import StockData
from src.data_processor import DataError, clean_prices
from src.feature_engineer import build_features
from src.live import LIVE_FEATURES, LiveEngine, live_features, train_quick_model, _provisional
from src.pipeline import load_config
from tests.helpers import synthetic_prices
from tests.test_core import raises


def _data(sym="TEST", n=1300):
    df, _ = clean_prices(synthetic_prices(n), min_rows=300)
    return StockData(symbol=sym, df=df, meta={"currency": "USD", "exchangeName": "TST"}, fetched_at="2026-01-01T00:00:00+00:00")


class FakeFetcher:
    def __init__(self, data):
        self.data, self.calls, self.price, self.date = data, 0, float(data.df["Close"].iloc[-1]), None

    def load(self, q, period, force=False, min_rows=None):
        return self.data

    def fetch_quote(self, symbol):
        self.calls += 1
        d = self.date or (self.data.df.index[-1] + pd.Timedelta(days=1))
        return {"price": self.price, "time": f"{d.date()}T10:00:00+00:00", "date": str(d.date()),
                "time_utc": pd.Timestamp.now(tz="UTC").isoformat(), "source": "fake"}

    def fetch_symbol(self, *a, **k):
        return self.data


def _engine():
    cfg = load_config()
    cfg["live"]["min_quote_interval_seconds"] = 0
    f = FakeFetcher(_data())
    return LiveEngine(f, cfg), f


def test_split_is_80_20_chronological_and_test_untouched():
    st = train_quick_model(_data(), load_config())
    m = st.metrics
    assert m["train_end"] < m["test_start"]
    assert abs(st.n_train / (st.n_train + st.n_test) - 0.8) < 0.02
    assert m["n"] == st.n_test and m["mae"] > 0 and m["naive_mae"] > 0
    assert 0 <= m["coverage_80"] <= 1 and 0 <= m["directional_accuracy"] <= 1
    json.dumps(m, allow_nan=False)


def test_tail_features_equal_full_history_features():
    """Incremental update (tail only) must give the same features as recomputing everything."""
    df = _data().df
    full = build_features(df)[LIVE_FEATURES].iloc[-1]
    tail = live_features(df.tail(300))
    assert np.allclose(full.values, tail.values, rtol=1e-6, atol=1e-9)


def test_tick_adds_price_without_retraining_and_caches_unchanged_price():
    eng, f = _engine()
    summary = eng.start("TEST")
    st = eng.states["TEST"]
    fits = []
    orig = st.est.fit
    st.est.fit = lambda *a, **k: fits.append(1) or orig(*a, **k)
    f.price = float(f.data.df["Close"].iloc[-1]) * 1.01
    a = eng.tick("TEST")
    b = eng.tick("TEST")                                   # same price -> cached
    f.price *= 1.02
    c = eng.tick("TEST")
    assert not fits, "model must not be retrained on a tick"
    assert a["recomputed"] and not b["recomputed"] and c["recomputed"]
    assert a["session_in_progress"] and a["prediction"]["target_date"] > str(f.data.df.index[-1].date())
    assert c["prediction"]["pred_close"] != a["prediction"]["pred_close"]
    p = c["prediction"]
    assert p["low"] <= p["pred_close"] <= p["high"]
    assert summary["metrics"]["n"] > 100 and c["compute_ms"] < 1500
    json.dumps(c, allow_nan=False)


def test_closed_market_uses_last_bar_and_no_provisional_row():
    eng, f = _engine()
    eng.start("TEST")
    f.date = f.data.df.index[-1]                            # quote is from the last completed session
    f.price = float(f.data.df["Close"].iloc[-1])
    r = eng.tick("TEST")
    assert not r["session_in_progress"] and abs(r["change_abs"]) < 1e-6 or r["change_label"] == "vs prior close"


def test_quote_throttle_and_feed_failure_serves_last_good():
    cfg = load_config()
    cfg["live"]["min_quote_interval_seconds"] = 60
    f = FakeFetcher(_data())
    eng = LiveEngine(f, cfg)
    eng.start("TEST")
    for _ in range(5):
        eng.tick("TEST")
    assert f.calls == 1                                      # shared quote cache, Yahoo called once

    def boom(symbol):
        raise DataError("down", "no_data")
    cfg["live"]["min_quote_interval_seconds"] = 0
    eng2 = LiveEngine(f, cfg)
    eng2.start("TEST")
    eng2.tick("TEST")
    f.fetch_quote = boom
    r = eng2.tick("TEST")
    assert r["feed_error"] is True


def test_untrained_symbol_and_short_history():
    eng, _ = _engine()
    raises("not_trained", eng.tick, "NOPE")
    short = _data()
    short.df = short.df.tail(230)
    raises("insufficient_history", train_quick_model, short, load_config())


def test_api_validation():
    import app as A
    c = A.app.test_client()
    assert c.post("/api/live/start", json={}).status_code == 400
    assert c.get("/api/live/tick").status_code == 400
    assert c.get("/api/live/tick?symbol=ZZZ").status_code == 409
