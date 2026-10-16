import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

import src.data_fetcher as dfm
from src.data_fetcher import StockDataFetcher, StockData
from src.data_processor import DataError, clean_prices
from src.pipeline import Analyzer, load_config
from tests.helpers import synthetic_prices
from tests.test_core import raises


def _analyzer():
    cfg = load_config()
    cfg["modeling"]["lstm"]["n_seeds"] = 1
    cfg["modeling"]["lstm"]["epochs"] = 15
    return Analyzer(cfg, base_dir=Path(tempfile.mkdtemp()))


def _data(sym="TEST"):
    df, _ = clean_prices(synthetic_prices(1300), min_rows=300)
    return StockData(symbol=sym, df=df, meta={"currency": "USD", "exchangeName": "TST"}, fetched_at="2026-01-01T00:00:00+00:00")


def test_end_to_end_payload():
    an = _analyzer()
    seen = []
    r = an.analyze("TEST", "5y", data=_data(), progress=lambda p, m: seen.append(p))
    assert seen == sorted(seen) and seen[-1] == 1.0
    f = r["forecast"]
    assert f["horizon"] == 21 and f["selected"] in f["models"]
    assert "lstm" in f["models"] and "random_forest" in f["models"] and "ridge" in f["models"]
    for name, m in f["models"].items():
        assert len(m["path"]["pred"]) == 21 and len(set(m["path"]["dates"])) == 21
        assert all(lo <= p <= hi for lo, p, hi in zip(m["path"]["low"], m["path"]["pred"], m["path"]["high"]))
        assert m["next_day"]["low"] <= m["next_day"]["pred_close"] <= m["next_day"]["high"]
        assert m["next_day"]["date"] > r["data"]["last_close_date"]
    ev = r["evaluation"]
    naive = next(x for x in ev["test"] if x["model"] == "naive")
    assert naive["rel_mae"] == 1.0 and naive["horizons"]["1"]["dir_acc"] is None
    for row in ev["test"] + ev["val"]:
        for h, v in row["horizons"].items():
            assert v["mae"] >= 0 and v["rmse"] >= v["mae"] - 1e-9
    s = ev["split"]
    assert s["train_range"][1] < s["val_range"][0] < s["val_range"][1] < s["test_range"][0]
    assert r["backtest"]["n"] == s["test"] and f["reliability"] in ("Moderate", "Low", "None demonstrated")
    json.dumps(r, allow_nan=False)                      # strictly JSON-safe
    # second call reuses trained results; force_retrain does not
    r2 = an.analyze("TEST", "5y", data=_data())
    assert r2["run"]["models_from_cache"] is True
    r3 = an.analyze("TEST", "5y", data=_data(), force_retrain=True)
    assert r3["run"]["models_from_cache"] is False


def test_too_little_history_gives_clear_error():
    an = _analyzer()
    df, _ = clean_prices(synthetic_prices(320), min_rows=300)
    sd = StockData(symbol="SHORT", df=df, meta={}, fetched_at="2026-01-01T00:00:00+00:00")
    raises("insufficient_history", an.analyze, "SHORT", "2y", data=sd)


# ---------------------------------------------------------------- fetcher with a fake Yahoo
class FakeYF:
    """Stands in for the yfinance module."""
    calls = []
    good = {"AAPL", "RELIANCE.NS"}
    fail = None

    class Ticker:
        def __init__(self, sym):
            self.sym = sym

        def history(self, **kw):
            FakeYF.calls.append(self.sym)
            if FakeYF.fail:
                raise FakeYF.fail
            if self.sym not in FakeYF.good:
                return pd.DataFrame()
            d = synthetic_prices(700, start="2023-06-01")
            d.index = d.index.tz_localize("Asia/Kolkata" if self.sym.endswith(".NS") else "America/New_York")
            return d

        def get_history_metadata(self):
            return {"currency": "INR" if self.sym.endswith(".NS") else "USD", "exchangeName": "NSI",
                    "exchangeTimezoneName": "Asia/Kolkata"}

    class Search:
        def __init__(self, q, **kw):
            self.quotes = [{"symbol": "RELIANCE.NS", "longname": "Reliance Industries Limited", "exchDisp": "NSE",
                            "quoteType": "EQUITY"}] if "reliance" in q.lower() else []


def _fetcher():
    dfm.yf = FakeYF
    FakeYF.calls, FakeYF.fail = [], None
    return StockDataFetcher(tempfile.mkdtemp(), ttl_minutes=60, min_rows=300, stale_after_days=10 ** 6)


def test_resolution_and_errors():
    f = _fetcher()
    d = f.load("aapl", "5y")
    assert d.symbol == "AAPL" and not d.df.index.tz
    d = f.load("RELIANCE", "5y")                                  # bare Indian symbol -> .NS fallback
    assert d.symbol == "RELIANCE.NS" and "RELIANCE" in FakeYF.calls and any("using RELIANCE.NS" in w for w in d.warnings)
    d = f.load("Reliance Industries", "5y")                       # company name -> search
    assert d.symbol == "RELIANCE.NS"
    raises("not_found", f.load, "zzz unknown company", "5y")
    raises("empty_query", f.load, "   ", "5y")
    raises("bad_period", f.fetch_symbol, "AAPL", "7y")
    raises("invalid_symbol", f.fetch_symbol, "AA PL;drop", "5y")


def test_cache_and_provider_failure():
    f = _fetcher()
    f.load("AAPL", "5y")
    n = len(FakeYF.calls)
    f.load("AAPL", "5y")
    assert len(FakeYF.calls) == n                                  # served from cache
    FakeYF.fail = ConnectionError("network down")
    d = f.load("AAPL", "5y", force=True)                           # refresh fails -> stale cache + warning
    assert d.stale_cache and any("cached data" in w for w in d.warnings)
    g = _fetcher()
    FakeYF.fail = ConnectionError("network down")
    raises("provider_unavailable", g.load, "MSFT", "5y")
    FakeYF.fail = type("YFRateLimitError", (Exception,), {})("429 Too Many Requests")
    raises("rate_limited", _fetcher_with_fail().load, "TSLA", "5y")


def _fetcher_with_fail():
    f = _fetcher()
    FakeYF.fail = type("YFRateLimitError", (Exception,), {})("429 Too Many Requests")
    return f


# ---------------------------------------------------------------- HTTP API
def test_api_validation():
    import app as A
    c = A.app.test_client()
    assert c.get("/api/health").json["status"] == "ok"
    assert c.post("/api/analyze", json={"query": ""}).status_code == 400
    assert c.post("/api/analyze", json={"query": "AAPL", "period": "99y"}).status_code == 400
    assert c.post("/api/analyze", json={"query": "x" * 100}).status_code == 400
    assert c.get("/api/jobs/nope").status_code == 404
    assert c.get("/api/search?q=a").json == {"results": []}
    assert b"Stock Trend Predictor" in c.get("/").data


def test_forecast_always_inside_its_own_range_and_partial_session_quote():
    an = _analyzer()
    sd = _data("PART")
    # make the last bar an unfinished session (as Yahoo returns it during market hours)
    today = pd.Timestamp.now(tz="UTC").tz_convert("Asia/Kolkata").normalize().tz_localize(None)
    df = sd.df.copy()
    df.index = list(df.index[:-1]) + [today]
    sd2 = StockData(symbol="PART.NS", df=df, meta={"exchangeTimezoneName": "Asia/Kolkata", "currency": "INR",
                    "regularMarketPrice": float(df["Close"].iloc[-1]),
                    "regularMarketTime": int(pd.Timestamp.now(tz="UTC").timestamp()),
                    "currentTradingPeriod": {"regular": {"end": int((pd.Timestamp.now(tz="UTC") + pd.Timedelta(hours=3)).timestamp())}}},
                    fetched_at="2026-01-01T00:00:00+00:00")
    from src.data_fetcher import StockDataFetcher
    from src.data_processor import split_incomplete_session
    kept, partial = split_incomplete_session(sd2.df, sd2.meta)
    assert partial is not None
    sd2 = StockData(symbol="PART.NS", df=kept, meta=sd2.meta, fetched_at=sd2.fetched_at, partial_bar=partial)
    r = an.analyze("PART.NS", "5y", data=sd2)
    d = r["data"]
    assert d["day_change_label"] == "vs last completed close"
    assert abs(d["day_change_pct"] - (d["latest_quote"]["price"] / d["last_close"] - 1) * 100) < 1e-9
    assert "T" in d["latest_quote"]["time"]                      # real timestamp, not midnight date
    for m in r["forecast"]["models"].values():
        for blk in (m["next_day"], m["month_end"]):
            assert blk["low"] <= blk["pred_close"] <= blk["high"]
        assert all(lo <= p <= hi for lo, p, hi in zip(m["path"]["low"], m["path"]["pred"], m["path"]["high"]))
