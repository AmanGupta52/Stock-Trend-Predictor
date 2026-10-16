"""Technical indicators computed from OHLCV data. Every value at row t uses data up to t only."""
from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n, min_periods=n).mean()


def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False, min_periods=n).mean()


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    """Wilder's RSI."""
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    rs = gain / loss.replace(0, np.nan)
    out = 100 - 100 / (1 + rs)
    out[(loss == 0) & (gain > 0)] = 100.0
    out[(loss == 0) & (gain == 0)] = 50.0
    return out


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    line = ema(close, fast) - ema(close, slow)
    sig = line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    return line, sig, line - sig


def bollinger(close: pd.Series, n: int = 20, k: float = 2.0):
    mid = sma(close, n)
    sd = close.rolling(n, min_periods=n).std(ddof=0)
    up, lo = mid + k * sd, mid - k * sd
    pctb = (close - lo) / (up - lo).replace(0, np.nan)
    return mid, up, lo, pctb, (up - lo) / mid


def atr(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14) -> pd.Series:
    prev = close.shift()
    tr = pd.concat([high - low, (high - prev).abs(), (low - prev).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()


def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    return (np.sign(close.diff()).fillna(0) * volume).cumsum()


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Return a frame (same index) of indicators for a frame with Open/High/Low/Close/Volume."""
    c, h, l, v = df["Close"], df["High"], df["Low"], df["Volume"]
    out = pd.DataFrame(index=df.index)
    for n in (20, 50, 200):
        out[f"sma_{n}"] = sma(c, n)
    out["ema_12"], out["ema_26"] = ema(c, 12), ema(c, 26)
    out["rsi_14"] = rsi(c, 14)
    out["macd"], out["macd_signal"], out["macd_hist"] = macd(c)
    out["bb_mid"], out["bb_up"], out["bb_low"], out["bb_pctb"], out["bb_width"] = bollinger(c)
    out["atr_14"] = atr(h, l, c)
    out["obv"] = obv(c, v)
    out["vol_sma_20"] = sma(v, 20)
    lr = np.log(c).diff()
    out["ret_1d"] = c.pct_change()
    out["volatility_20d"] = lr.rolling(20, min_periods=20).std() * np.sqrt(TRADING_DAYS)
    return out


class TechnicalIndicators:
    """Backward-compatible wrapper around :func:`compute_indicators`."""

    def compute_all_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.join(compute_indicators(df))


def trend_summary(df: pd.DataFrame, ind: pd.DataFrame) -> dict:
    """Descriptive (non-predictive) reading of the latest indicator values."""
    c = df["Close"]
    last = len(df) - 1
    price = float(c.iloc[last])
    lr = np.log(df["Adj Close"]).diff().dropna()
    signals = []

    def val(col):
        x = ind[col].iloc[last]
        return None if pd.isna(x) else float(x)

    s20, s50, s200 = val("sma_20"), val("sma_50"), val("sma_200")
    if s50 is not None:
        signals.append({"name": "Price vs 50-day average", "value": f"{(price / s50 - 1) * 100:+.1f}%",
                        "reading": "Bullish" if price > s50 else "Bearish"})
    if s200 is not None:
        signals.append({"name": "Price vs 200-day average", "value": f"{(price / s200 - 1) * 100:+.1f}%",
                        "reading": "Bullish" if price > s200 else "Bearish"})
    if s50 is not None and s200 is not None:
        signals.append({"name": "50-day vs 200-day average", "value": f"{(s50 / s200 - 1) * 100:+.1f}%",
                        "reading": "Bullish (golden-cross regime)" if s50 > s200 else "Bearish (death-cross regime)"})
    r = val("rsi_14")
    if r is not None:
        reading = "Overbought (>70)" if r > 70 else "Oversold (<30)" if r < 30 else "Neutral"
        signals.append({"name": "RSI (14)", "value": f"{r:.1f}", "reading": reading})
    m, ms = val("macd"), val("macd_signal")
    if m is not None and ms is not None:
        signals.append({"name": "MACD vs signal", "value": f"{m - ms:+.3f}",
                        "reading": "Bullish" if m > ms else "Bearish"})
    pb = val("bb_pctb")
    if pb is not None:
        reading = "Near upper band" if pb > 0.8 else "Near lower band" if pb < 0.2 else "Mid-range"
        signals.append({"name": "Bollinger %B", "value": f"{pb:.2f}", "reading": reading})
    vs, vol_avg = float(df["Volume"].iloc[last]), val("vol_sma_20")
    if vol_avg and vol_avg > 0:
        signals.append({"name": "Volume vs 20-day average", "value": f"{vs / vol_avg:.2f}x",
                        "reading": "Above average" if vs > vol_avg * 1.2 else "Below average" if vs < vol_avg * 0.8 else "Normal"})

    bull = sum(1 for s in signals if s["reading"].startswith(("Bullish", "Near lower", "Oversold")))
    bear = sum(1 for s in signals if s["reading"].startswith(("Bearish", "Near upper", "Overbought")))
    trend = "Uptrend" if bull - bear >= 2 else "Downtrend" if bear - bull >= 2 else "Sideways / mixed"

    def ret_over(n):
        return float(c.iloc[last] / c.iloc[last - n] - 1) if last >= n else None

    peak = df["Adj Close"].cummax()
    drawdown = df["Adj Close"] / peak - 1
    stats = {
        "return_1m": ret_over(21), "return_3m": ret_over(63), "return_6m": ret_over(126), "return_1y": ret_over(252),
        "volatility_20d": val("volatility_20d"),
        "volatility_1y": float(lr.iloc[-252:].std() * np.sqrt(TRADING_DAYS)) if len(lr) >= 60 else None,
        "avg_daily_return": float(lr.mean()), "max_drawdown": float(drawdown.min()),
        "current_drawdown": float(drawdown.iloc[-1]),
        "high_52w": float(df["High"].iloc[-252:].max()), "low_52w": float(df["Low"].iloc[-252:].min()),
        "avg_volume_20d": vol_avg,
    }
    return {"trend": trend, "signals": signals, "stats": stats,
            "note": "Indicator readings describe past prices; they are not model predictions."}
