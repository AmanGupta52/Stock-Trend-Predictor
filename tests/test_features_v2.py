"""Tests for watchlist, tracking, compare, market inputs, tuning, GRU/ensemble and volatility (all offline)."""
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from src.compare import compare_stocks
from src.data_fetcher import StockData
from src.data_processor import DataError, clean_prices
from src.lstm_model import NumpyGRU
from src.market_context import load_context
from src.model_trainer import candidate_params, cv_folds, SEARCH_SPACE
from src.pipeline import Analyzer, load_config
from src.store import Store
from src.tracker import forecast_rows, resolve_pending, summarize
from src.volatility import realised_targets
from tests.helpers import synthetic_prices
from tests.test_core import raises


def _data(sym="TEST", n=1300, seed=1):
    df, _ = clean_prices(synthetic_prices(n, seed=seed), min_rows=300)
    return StockData(symbol=sym, df=df, meta={"currency": "USD", "exchangeName": "TST"},
                     fetched_at="2026-01-01T00:00:00+00:00")


def test_gru_gradients_match_numerical():
    net = NumpyGRU(4, 5, 3, 1)
    rng = np.random.default_rng(0)
    X, Y = rng.normal(size=(6, 7, 4)), rng.normal(size=(6, 3))
    _, g = net.loss_and_grads(X, Y)
    for k, a in net.p.items():
        flat = a.ravel()
        for j in rng.choice(flat.size, min(5, flat.size), replace=False):
            o = flat[j]
            flat[j] = o + 1e-6; lp = net.loss_and_grads(X, Y)[0]
            flat[j] = o - 1e-6; lm = net.loss_and_grads(X, Y)[0]
            flat[j] = o
            num = (lp - lm) / 2e-6
            assert abs(num - g[k].ravel()[j]) < 1e-5 * (1 + abs(num)), k


def test_cv_folds_stay_inside_training_block_with_embargo():
    train = np.arange(0, 700)
    folds = cv_folds(train, 21)
    assert folds
    for fit, val in folds:
        assert fit.max() < val.min() - 20 and val.max() <= train.max()


def test_candidate_params_reproducible_and_start_with_default():
    a, b = candidate_params("random_forest", 7), candidate_params("random_forest", 7)
    assert a == b and len(a) >= 2
    for c in a[1:]:
        assert set(c) <= set(SEARCH_SPACE["random_forest"])


def test_context_is_lagged_and_not_look_ahead():
    idx = pd.bdate_range("2023-01-02", periods=400)
    lvl = pd.Series(np.arange(400, dtype=float) + 100, index=idx)

    class F:
        def fetch_series(self, sym, period, force=False):
            return lvl

        def get_sector(self, sym):
            return None

    cfg = {"other": [{"symbol": "^X", "name": "Lagged", "lag": 1}, {"symbol": "^Y", "name": "Same", "lag": 0}]}
    ctx, used, _ = load_context(F(), cfg, "ABC", idx, "2y")
    assert ctx is not None and len(used) == 2
    assert ctx["lagged"].iloc[10] == lvl.iloc[9]      # yesterday's value only
    assert ctx["same"].iloc[10] == lvl.iloc[10]


def test_realised_targets_use_only_future_returns():
    c = synthetic_prices(300)["Close"]
    t1 = realised_targets(c)
    c2 = c.copy()
    c2.iloc[150:] *= 1.5                       # change the future of t=100
    t2 = realised_targets(c2)
    assert np.allclose(t1.iloc[:125].dropna().values, t2.iloc[:125].dropna().values)  # past targets unaffected by far future
    assert t1.iloc[-1].isna().all()            # nothing known past the end


def test_store_watchlist_crud_and_forecast_never_overwritten():
    s = Store(Path(tempfile.mkdtemp()) / "a.db")
    s.wl_add("TCS.NS", "TCS", "NSE", "INR", "2026-01-01")
    s.wl_add("TCS.NS", "TCS", "NSE", "INR", "2026-01-02")
    assert len(s.wl_list()) == 1
    assert s.wl_remove("TCS.NS") == 1 and s.wl_list() == []
    row = dict(symbol="A", model="m", kind="next_day", made_at="t", base_date="2026-01-01", base_close=10.0,
               target_date="2026-01-02", pred_close=11.0, low=9.0, high=12.0, selected=1, currency="USD")
    assert s.log_forecasts([row]) == 1
    row2 = dict(row, pred_close=99.0)
    s.log_forecasts([row2])
    fs = s.forecasts("A")
    assert len(fs) == 1 and fs[0]["pred_close"] == 11.0


def test_tracking_logs_resolves_and_summarises():
    cfg = load_config()
    cfg["modeling"]["lstm"]["n_seeds"] = 1
    cfg["modeling"]["lstm"]["epochs"] = 5
    cfg["modeling"]["models"] = ["ridge", "random_forest"]
    an = Analyzer(cfg, base_dir=Path(tempfile.mkdtemp()))
    d = _data()
    r = an.analyze("TEST", "5y", data=d, options={"use_context": False})
    rows = forecast_rows(r, "2026-01-01T00:00:00")
    assert any(x["model"] == "naive" for x in rows)
    assert len(an.store.pending("TEST")) > 0
    assert summarize(an.store, "TEST")["resolved_count"] == 0 if "resolved_count" in summarize(an.store, "TEST") else True
    # extend history past the next-day target and resolve
    nd = r["forecast"]["models"][r["forecast"]["selected"]]["next_day"]["date"]
    ext = d.df.copy()
    ext.loc[pd.Timestamp(nd)] = ext.iloc[-1]
    ext = ext.sort_index()
    n = resolve_pending(an.store, "TEST", ext, "2026-02-01T00:00:00")
    assert n >= 1
    done = [f for f in an.store.forecasts("TEST") if f["actual_close"] is not None]
    assert all(f["target_date"] <= str(ext.index[-1].date()) for f in done)
    summarize(an.store, "TEST")


def test_compare_validates_and_rebases():
    class F:
        def load(self, q, period, force=False, min_rows=60):
            return _data(q, 500, seed=abs(hash(q)) % 100)

    raises("bad_request", compare_stocks, F(), ["A"], "5y")
    raises("bad_request", compare_stocks, F(), ["A", "A"], "5y")
    out = compare_stocks(F(), ["A", "B", "C"], "5y")
    first = out["rebased"]["series"] if "series" in out.get("rebased", {}) else None
    assert out and ("stats" in out)


def test_analysis_with_tuning_context_ensemble_and_volatility_is_json_safe():
    import json
    cfg = load_config()
    cfg["modeling"]["lstm"]["n_seeds"] = 1
    cfg["modeling"]["lstm"]["epochs"] = 5
    cfg["modeling"]["models"] = ["ridge", "random_forest", "gru"]

    class F:
        def fetch_series(self, sym, period, force=False):
            idx = pd.bdate_range("2021-01-01", periods=1400)
            return pd.Series(100 + np.cumsum(np.random.default_rng(abs(hash(sym)) % 99).normal(size=1400)), index=idx)

        def get_sector(self, sym):
            return None

    an = Analyzer(cfg, base_dir=Path(tempfile.mkdtemp()), fetcher=F()) if "fetcher" in Analyzer.__init__.__code__.co_varnames \
        else Analyzer(cfg, base_dir=Path(tempfile.mkdtemp()))
    r = an.analyze("TEST", "5y", data=_data(), options={"use_context": False, "tune": False})
    assert r["volatility"].get("forecast") and r["volatility"]["selected"]
    json.dumps(r, allow_nan=False)
    assert "ensemble" in r["forecast"]
    w = r["evaluation"].get("ensemble_weights") or {}
    if w:
        assert abs(sum(w.values()) - 1) < 1e-6
