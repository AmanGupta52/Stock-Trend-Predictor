"""Leak-free feature and target construction.

Feature row t uses only data up to and including session t. The target for row t is the vector of
cumulative log returns ln(C[t+h] / C[t]) for h = 1..H, i.e. a *future* quantity that is never used as input.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd

from .data_processor import adjust_ohlc
from .technical_indicators import atr, bollinger, ema, macd, obv, rsi, sma

WARMUP = 60  # rows lost to rolling windows (longest feature window)


@dataclass
class FeatureSet:
    X: pd.DataFrame          # features for every row after warm-up (incl. rows without a target yet)
    Y: pd.DataFrame          # targets for the first n_labeled rows, columns h1..hH
    close: pd.Series         # adjusted close aligned with X
    horizon: int
    seq_len: int
    context_used: Optional[List[str]] = None

    @property
    def n_labeled(self) -> int:
        return len(self.Y)

    @property
    def feature_names(self) -> List[str]:
        return list(self.X.columns)


def context_features(ctx: pd.DataFrame, lr: pd.Series) -> pd.DataFrame:
    """Features from aligned market-context series (index, VIX, FX, commodity, sector).

    ``ctx`` columns are levels already aligned to the stock's sessions and lagged where the asset closes
    after the stock does, so row t only contains information available when the stock's session t ended.
    The FIRST column is the primary market index (used for beta / correlation / relative strength).
    """
    f = pd.DataFrame(index=ctx.index)
    for col in ctx.columns:
        s = ctx[col].where(ctx[col] > 0)
        f[f"ctx_{col}_ret1"] = s.pct_change()
        f[f"ctx_{col}_ret5"] = s.pct_change(5)
        f[f"ctx_{col}_ret20"] = s.pct_change(20)
        f[f"ctx_{col}_z60"] = ((s - s.rolling(60, min_periods=60).mean()) /
                               s.rolling(60, min_periods=60).std().replace(0, np.nan)).clip(-5, 5)
    prim = ctx.iloc[:, 0].where(ctx.iloc[:, 0] > 0)
    mlr = np.log(prim).diff()
    f["beta_60"] = (lr.rolling(60, min_periods=60).cov(mlr) / mlr.rolling(60, min_periods=60).var()).clip(-5, 5)
    f["corr_60"] = lr.rolling(60, min_periods=60).corr(mlr)
    f["rel_ret_20"] = lr.rolling(20, min_periods=20).sum() - mlr.rolling(20, min_periods=20).sum()
    return f


def build_features(ohlcv: pd.DataFrame, context: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Compute model features from a raw (Close/Adj Close) price frame. Uses adjusted prices."""
    a = adjust_ohlc(ohlcv)
    c, h, l, o, v = a["Close"], a["High"], a["Low"], a["Open"], a["Volume"]
    lr = np.log(c).diff()
    f = pd.DataFrame(index=a.index)
    f["ret_1"] = lr
    for k in (2, 3, 5, 10, 20, 60):
        f[f"ret_{k}"] = np.log(c / c.shift(k))
    for n in (5, 10, 20, 50):
        f[f"close_sma{n}"] = c / sma(c, n) - 1
    f["sma20_sma50"] = sma(c, 20) / sma(c, 50) - 1
    f["ema12_ema26"] = ema(c, 12) / ema(c, 26) - 1
    f["rsi"] = rsi(c, 14) / 100
    line, sig, hist = macd(c)
    f["macd"], f["macd_hist"] = line / c, hist / c
    _, _, _, pctb, width = bollinger(c)
    f["bb_pctb"], f["bb_width"] = pctb, width
    f["atr_pct"] = atr(h, l, c) / c
    for n in (5, 10, 20, 60):
        f[f"vol_{n}"] = lr.rolling(n, min_periods=n).std()
    f["vol_ratio"] = f["vol_5"] / f["vol_20"]
    f["hl_range"] = (h - l) / c
    f["oc_ret"] = np.log(c / o)
    f["gap"] = np.log(o / c.shift())
    vm, vs = v.rolling(20, min_periods=20).mean(), v.rolling(20, min_periods=20).std()
    f["vol_z"] = ((v - vm) / vs.replace(0, np.nan)).clip(-5, 5)
    f["vol_chg"] = np.log1p(v) - np.log1p(vm)
    f["obv_slope"] = (obv(c, v).diff(10) / (vm * 10).replace(0, np.nan)).clip(-5, 5)
    f["dd_252"] = c / c.rolling(252, min_periods=WARMUP).max() - 1
    f["zscore_20"] = (c - sma(c, 20)) / c.rolling(20, min_periods=20).std().replace(0, np.nan)
    dow = a.index.dayofweek
    f["dow_sin"], f["dow_cos"] = np.sin(2 * np.pi * dow / 5), np.cos(2 * np.pi * dow / 5)
    if context is not None and context.shape[1]:
        f = f.join(context_features(context.reindex(f.index), lr))
    return f.replace([np.inf, -np.inf], np.nan)


def build_targets(close: pd.Series, horizon: int) -> pd.DataFrame:
    """Cumulative log return for h=1..horizon (NaN where the future is unknown)."""
    lc = np.log(close)
    return pd.DataFrame({f"h{h}": lc.shift(-h) - lc for h in range(1, horizon + 1)}, index=close.index)


def make_feature_set(ohlcv: pd.DataFrame, horizon: int = 21, seq_len: int = 20,
                     context: Optional[pd.DataFrame] = None) -> FeatureSet:
    feats = build_features(ohlcv, context)
    valid = feats.dropna()
    # Rolling features are NaN only in the warm-up prefix; make sure nothing is silently dropped mid-series.
    first = valid.index[0]
    interior_nan = feats.loc[first:].isna().any(axis=1).sum()
    if interior_nan:
        feats = feats.loc[first:].ffill(limit=3).dropna()
    else:
        feats = valid
    close = ohlcv["Adj Close"].reindex(feats.index)
    targets = build_targets(close, horizon)
    n_lab = int(targets.notna().all(axis=1).sum())
    Y = targets.iloc[:n_lab]
    return FeatureSet(X=feats, Y=Y, close=close, horizon=horizon, seq_len=seq_len,
                      context_used=list(context.columns) if context is not None and context.shape[1] else None)
