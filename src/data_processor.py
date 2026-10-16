"""Preprocessing of raw price data: normalisation, validation and split of incomplete sessions.

No prices are ever invented: unusable rows are dropped and reported as warnings.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

PRICE_COLS = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]


class DataError(Exception):
    """A user-facing data problem (bad symbol, no data, provider down, ...)."""

    def __init__(self, message: str, code: str = "data_error"):
        super().__init__(message)
        self.code = code
        self.message = message


def adjust_ohlc(df: pd.DataFrame) -> pd.DataFrame:
    """Scale Open/High/Low by Adj Close / Close so the whole bar is dividend/split adjusted."""
    factor = df["Adj Close"] / df["Close"]
    out = pd.DataFrame(index=df.index)
    out["Open"] = df["Open"] * factor
    out["High"] = df["High"] * factor
    out["Low"] = df["Low"] * factor
    out["Close"] = df["Adj Close"]
    out["Volume"] = df["Volume"]
    return out


def normalise_history(raw: pd.DataFrame) -> pd.DataFrame:
    """Normalise a yfinance history frame: flat columns, naive daily index, needed columns."""
    df = raw.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.loc[:, ~df.columns.duplicated()]
    if "Adj Close" not in df.columns and "Close" in df.columns:
        df["Adj Close"] = df["Close"]
    missing = [c for c in PRICE_COLS if c not in df.columns and c != "Volume"]
    if missing:
        raise DataError(f"Provider response is missing columns: {', '.join(missing)}", "bad_response")
    if "Volume" not in df.columns:
        df["Volume"] = 0.0
    idx = pd.DatetimeIndex(df.index)
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    df.index = idx.normalize()
    df.index.name = "Date"
    return df[PRICE_COLS].sort_index()


def clean_prices(df: pd.DataFrame, min_rows: int = 300, stale_after_days: int = 10,
                 today: Optional[pd.Timestamp] = None) -> Tuple[pd.DataFrame, List[str]]:
    """Validate a normalised history frame. Never invents prices; bad rows are dropped.

    Returns (clean frame, warnings). Raises DataError when too little usable data remains.
    """
    warnings: List[str] = []
    n0 = len(df)
    df = df[~df.index.duplicated(keep="last")].sort_index()
    df = df.replace([np.inf, -np.inf], np.nan)

    ohlc = ["Open", "High", "Low", "Close", "Adj Close"]
    bad = df[ohlc].isna().any(axis=1) | (df[ohlc] <= 0).any(axis=1)
    if bad.any():
        warnings.append(f"Dropped {int(bad.sum())} rows with missing or non-positive prices (no values were filled in).")
        df = df[~bad]

    inconsistent = (df["High"] < df[["Open", "Close"]].max(axis=1) * 0.999) | \
                   (df["Low"] > df[["Open", "Close"]].min(axis=1) * 1.001)
    if inconsistent.any():
        # Provider quirk (e.g. rounding on illiquid names) - repair the envelope, do not drop.
        df.loc[inconsistent, "High"] = df.loc[inconsistent, ["Open", "High", "Close"]].max(axis=1)
        df.loc[inconsistent, "Low"] = df.loc[inconsistent, ["Open", "Low", "Close"]].min(axis=1)
        warnings.append(f"Adjusted High/Low on {int(inconsistent.sum())} rows where they contradicted Open/Close.")

    if df["Volume"].isna().any():
        df["Volume"] = df["Volume"].fillna(0.0)
    if (df["Volume"] <= 0).mean() > 0.5:
        warnings.append("Volume is missing/zero for most sessions (typical for indices); volume analysis is limited.")

    if len(df) < min_rows:
        raise DataError(
            f"Only {len(df)} usable daily records were found (need at least {min_rows}, ~1.5 years). "
            "This ticker may be newly listed or have sparse data. Try a longer period or another symbol.",
            "insufficient_history")

    gaps = df.index.to_series().diff().dt.days
    big = gaps[gaps > 7]
    if len(big):
        warnings.append(f"{len(big)} gap(s) longer than 7 calendar days in the history "
                        f"(largest {int(big.max())} days, ending {big.idxmax().date()}); trading halts or missing data.")

    jumps = np.log(df["Adj Close"]).diff().abs()
    if (jumps > 0.4).any():
        warnings.append(f"{int((jumps > 0.4).sum())} single-day move(s) larger than 40% in adjusted prices - "
                        "check for unadjusted corporate actions.")

    today = today if today is not None else pd.Timestamp.now().normalize()
    age = (today - df.index[-1]).days
    if age > stale_after_days:
        warnings.append(f"Latest bar is {age} days old ({df.index[-1].date()}); the stock may be suspended, delisted, "
                        "or the provider data is stale.")
    if n0 - len(df) > 0.1 * n0:
        warnings.append("More than 10% of the downloaded rows were unusable.")
    return df, warnings


def split_incomplete_session(df: pd.DataFrame, meta: Dict[str, Any],
                             now_utc: Optional[pd.Timestamp] = None) -> Tuple[pd.DataFrame, Optional[Dict[str, Any]]]:
    """If the last daily bar belongs to a session that has not finished, split it off.

    Yahoo returns the in-progress session as the last daily bar. Its 'Close' is only the latest
    traded price, so it must not be used for training or as a prediction base.
    """
    if df.empty:
        return df, None
    now_utc = now_utc if now_utc is not None else pd.Timestamp.now(tz="UTC")
    tzname = meta.get("exchangeTimezoneName") or meta.get("timezone") or "UTC"
    try:
        now_local = now_utc.tz_convert(tzname)
    except Exception:
        now_local = now_utc
    last = df.index[-1]
    if last.date() != now_local.date():
        return df, None
    end_ts = None
    try:
        end_ts = pd.Timestamp(int(meta["currentTradingPeriod"]["regular"]["end"]), unit="s", tz="UTC")
    except Exception:
        pass
    if end_ts is not None:
        in_progress = now_utc < end_ts + pd.Timedelta(minutes=20)
    else:
        in_progress = now_local.hour < 17  # conservative generic fallback
    if not in_progress:
        return df, None
    row = df.iloc[-1]
    partial = {"date": str(last.date()), "price": float(row["Close"]), "note":
               "Session still in progress or just closed; excluded from model training."}
    return df.iloc[:-1], partial
