import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from src.data_processor import DataError, clean_prices, normalise_history, split_incomplete_session
from src.feature_engineer import make_feature_set, build_features
from src.lstm_model import NumpyLSTM, LSTMRegressor
from src.model_trainer import make_splits
from src.technical_indicators import compute_indicators, rsi, sma, macd
from tests.helpers import synthetic_prices

SAMPLE = Path(__file__).resolve().parent.parent / "sample_data" / "RELIANCE.NS_20251016.csv"


def raises(code, fn, *a, **k):
    try:
        fn(*a, **k)
    except DataError as e:
        assert e.code == code, (e.code, code)
        return
    raise AssertionError("DataError not raised")


# ---------------------------------------------------------------- indicators
def test_indicators_basic():
    df = synthetic_prices(400)
    ind = compute_indicators(df)
    assert np.allclose(ind["sma_20"].iloc[-1], df["Close"].iloc[-20:].mean())
    r = ind["rsi_14"].dropna()
    assert r.between(0, 100).all()
    line, sig, hist = macd(df["Close"])
    assert np.allclose((line - sig).dropna(), hist.dropna())
    up = pd.Series(np.arange(1.0, 60.0))
    assert rsi(up).iloc[-1] == 100.0          # only gains -> RSI 100


# ---------------------------------------------------------------- leakage
def test_features_do_not_use_future_data():
    df = synthetic_prices(600)
    full = build_features(df)
    cut = 450
    part = build_features(df.iloc[:cut])
    a, b = full.iloc[:cut].dropna(), part.dropna()
    assert len(a) == len(b)
    assert np.allclose(a.values, b.values, equal_nan=True), "feature at t changed when future rows were removed"
    # perturbing the future must not change past features
    df2 = df.copy()
    df2.iloc[500:, df2.columns.get_loc("Close")] *= 3
    df2.iloc[500:, df2.columns.get_loc("Adj Close")] *= 3
    f2 = build_features(df2)
    assert np.allclose(full.iloc[:500].dropna().values, f2.iloc[:500].dropna().values)


def test_targets_are_future_log_returns():
    df = synthetic_prices(500)
    fs = make_feature_set(df, 21, 20)
    t = 100
    d = fs.X.index[t]
    c = df["Adj Close"]
    pos = c.index.get_loc(d)
    for h in (1, 5, 21):
        assert np.isclose(fs.Y.iloc[t][f"h{h}"], np.log(c.iloc[pos + h] / c.iloc[pos]))
    assert len(fs.Y) == len(fs.X) - 21


def test_splits_are_chronological_and_embargoed():
    p = make_splits(1200, 20, 21)
    assert p.train.max() < p.val.min() < p.val.max() < p.test.min()
    # label windows (t .. t+21) of an earlier block never reach the first row of the next block
    assert p.train.max() + 21 < p.val.min()
    assert p.val.max() + 21 < p.test.min()
    assert p.fit_tv.max() + 21 < p.test.min()
    raises("insufficient_history", make_splits, 200, 20, 21)


# ---------------------------------------------------------------- data validation
def test_clean_prices_drops_bad_rows_without_inventing():
    df = synthetic_prices(400)
    df.iloc[10, df.columns.get_loc("Close")] = np.nan
    df.iloc[20, df.columns.get_loc("Open")] = -5
    out, warns = clean_prices(df, min_rows=300, today=df.index[-1])
    assert len(out) == 398 and any("Dropped 2" in w for w in warns)
    assert not out.isna().any().any()
    raises("insufficient_history", clean_prices, df.iloc[:100], 300)


def test_stale_and_gap_warnings():
    df = synthetic_prices(400)
    df = df.drop(df.index[200:215])
    out, warns = clean_prices(df, 300, 10, today=df.index[-1] + pd.Timedelta(days=30))
    assert any("gap" in w for w in warns) and any("old" in w for w in warns)


def test_incomplete_session_is_split_off():
    df = synthetic_prices(400)
    now = pd.Timestamp("2026-10-15 08:00", tz="UTC")                  # 13:30 IST, market open
    df.index = list(df.index[:-1]) + [pd.Timestamp("2026-10-15")]
    meta = {"exchangeTimezoneName": "Asia/Kolkata",
            "currentTradingPeriod": {"regular": {"end": int(pd.Timestamp("2026-10-15 10:00", tz="UTC").timestamp())}}}
    kept, partial = split_incomplete_session(df, meta, now)
    assert partial and len(kept) == len(df) - 1
    kept, partial = split_incomplete_session(df, meta, pd.Timestamp("2026-10-15 12:00", tz="UTC"))  # after close
    assert partial is None and len(kept) == len(df)


def test_real_reliance_history_from_project():
    """Real Yahoo Finance daily data that shipped in the original project (Oct 2024 - Oct 2025)."""
    raw = pd.read_csv(SAMPLE, header=0, skiprows=[1, 2], index_col=0, parse_dates=True)
    df = normalise_history(raw.rename_axis("Date"))
    assert len(df) > 240 and df.index.is_monotonic_increasing
    clean, _ = clean_prices(df, min_rows=200, today=df.index[-1])
    ind = compute_indicators(clean[["Open", "High", "Low", "Close", "Volume"]])
    assert 0 < ind["rsi_14"].iloc[-1] < 100 and ind["sma_50"].iloc[-1] > 0
    feats = build_features(clean)
    assert feats.dropna().shape[0] > 150 and np.isfinite(feats.dropna().values).all()


# ---------------------------------------------------------------- LSTM
def test_numpy_lstm_gradients_match_numerical():
    net = NumpyLSTM(4, 5, 3, 1)
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


def test_lstm_learns_a_signal():
    rng = np.random.default_rng(3)
    F = rng.normal(size=(400, 3))
    Y = np.stack([F[:, 0], F[:, 0] * 0.5], axis=1)            # target depends on the latest row
    idx = np.arange(9, 400)
    m = LSTMRegressor(seq_len=10, hidden=12, epochs=40, patience=40, n_seeds=1, backend="numpy", lr=1e-2)
    m.fit(F, idx[:250], Y[idx[:250]], idx[250:300], Y[idx[250:300]])
    p = m.predict(F, idx[300:])
    corr = np.corrcoef(p[:, 0], Y[idx[300:], 0])[0, 1]
    assert corr > 0.8, corr
