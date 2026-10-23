"""Market-data access (Yahoo Finance via yfinance) with validation and caching.

All errors that a user can act on are raised as :class:`DataError` with a short
machine-readable ``code`` and a human-readable message.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:  # yfinance is a hard requirement at runtime, but keep imports testable
    import yfinance as yf
except ImportError:  # pragma: no cover - exercised only when yfinance is absent
    yf = None

from .data_processor import DataError, PRICE_COLS, clean_prices, normalise_history, split_incomplete_session

log = logging.getLogger(__name__)

SYMBOL_RE = re.compile(r"^[A-Za-z0-9^][A-Za-z0-9.\-=&^]{0,19}$")
VALID_PERIODS = ("1y", "2y", "5y", "10y", "max")
# Exchange suffixes tried (in order) when a bare symbol such as "RELIANCE" has no US listing.
INDIA_SUFFIXES = (".NS", ".BO")


@dataclass
class StockData:
    """Container for a downloaded and validated price history."""

    symbol: str
    df: pd.DataFrame                       # completed daily sessions only
    meta: Dict[str, Any] = field(default_factory=dict)
    fetched_at: str = ""
    from_cache: bool = False
    stale_cache: bool = False
    partial_bar: Optional[Dict[str, Any]] = None   # today's unfinished session, if any
    warnings: List[str] = field(default_factory=list)
    alternatives: List[Dict[str, Any]] = field(default_factory=list)
    interpreted_from: Optional[str] = None


# --------------------------------------------------------------------------- helpers
def validate_symbol(text: str) -> str:
    """Return a normalised symbol or raise DataError."""
    q = (text or "").strip()
    if not q:
        raise DataError("Enter a company name or stock symbol (e.g. RELIANCE.NS, TCS.NS, AAPL, MSFT).",
                        "empty_query")
    if len(q) > 60:
        raise DataError("Search text is too long.", "invalid_symbol")
    return q


def _classify_exception(exc: Exception) -> DataError:
    name = type(exc).__name__.lower()
    text = str(exc).lower()
    if "ratelimit" in name or "too many requests" in text or "429" in text:
        return DataError("Yahoo Finance is rate-limiting requests. Wait a minute and try again.", "rate_limited")
    return DataError(
        "Could not reach Yahoo Finance (network or provider problem). "
        f"Check your internet connection and try again. Details: {type(exc).__name__}",
        "provider_unavailable")


# --------------------------------------------------------------------------- fetcher
class StockDataFetcher:
    """Searches, resolves, downloads and caches daily price history from Yahoo Finance."""

    def __init__(self, cache_dir: str = ".cache", ttl_minutes: int = 60, min_rows: int = 300,
                 stale_after_days: int = 10):
        self.cache_dir = Path(cache_dir) / "data"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.ttl = pd.Timedelta(minutes=ttl_minutes)
        self.min_rows = min_rows
        self.stale_after_days = stale_after_days

    # ---- search -----------------------------------------------------------------
    def search(self, query: str, limit: int = 8) -> List[Dict[str, Any]]:
        """Search companies/symbols. Returns [{symbol, name, exchange, type}]. Never raises."""
        q = (query or "").strip()
        if not q or yf is None:
            return []
        quotes: List[Dict[str, Any]] = []
        try:
            quotes = list(yf.Search(q, max_results=limit, news_count=0).quotes or [])
        except Exception as exc:
            log.warning("yf.Search failed (%s); trying HTTP fallback", exc)
            try:
                import requests
                r = requests.get("https://query2.finance.yahoo.com/v1/finance/search",
                                 params={"q": q, "quotesCount": limit, "newsCount": 0},
                                 headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
                r.raise_for_status()
                quotes = r.json().get("quotes", [])
            except Exception as exc2:
                log.warning("Search fallback failed: %s", exc2)
                return []
        out = []
        for item in quotes:
            if item.get("quoteType") not in ("EQUITY", "ETF", "INDEX"):
                continue
            sym = item.get("symbol")
            if not sym:
                continue
            out.append({"symbol": sym,
                        "name": item.get("longname") or item.get("shortname") or sym,
                        "exchange": item.get("exchDisp") or item.get("exchange") or "",
                        "type": item.get("quoteType")})
        return out[:limit]

    # ---- download ---------------------------------------------------------------
    def _cache_paths(self, symbol: str, period: str) -> Tuple[Path, Path]:
        safe = re.sub(r"[^A-Za-z0-9._-]", "_", symbol)
        return self.cache_dir / f"{safe}__{period}.csv", self.cache_dir / f"{safe}__{period}.json"

    def _download(self, symbol: str, period: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        if yf is None:
            raise DataError("The 'yfinance' package is not installed. Run: pip install -r requirements.txt",
                            "missing_dependency")
        try:
            tk = yf.Ticker(symbol)
            raw = tk.history(period=period, interval="1d", auto_adjust=False, actions=False)
        except Exception as exc:
            raise _classify_exception(exc) from exc
        if raw is None or raw.empty:
            raise DataError(f"No price data returned for '{symbol}'.", "no_data")
        meta: Dict[str, Any] = {}
        try:
            meta = dict(tk.get_history_metadata() or {})
        except Exception as exc:
            log.info("history metadata unavailable for %s: %s", symbol, exc)
        return normalise_history(raw), meta

    def _load_cache(self, symbol: str, period: str):
        csv_p, js_p = self._cache_paths(symbol, period)
        if not (csv_p.exists() and js_p.exists()):
            return None
        try:
            df = pd.read_csv(csv_p, index_col=0, parse_dates=True)
            info = json.loads(js_p.read_text())
            return df, info
        except Exception as exc:
            log.warning("Ignoring unreadable cache for %s: %s", symbol, exc)
            return None

    def _save_cache(self, symbol: str, period: str, df: pd.DataFrame, meta: Dict[str, Any]) -> None:
        csv_p, js_p = self._cache_paths(symbol, period)
        try:
            df.to_csv(csv_p)
            js_p.write_text(json.dumps({"fetched_at": pd.Timestamp.now(tz="UTC").isoformat(),
                                        "meta": meta}, default=str))
        except Exception as exc:  # cache problems must never break an analysis
            log.warning("Could not write cache: %s", exc)

    def _get_raw(self, symbol: str, period: str, force: bool):
        """Cache-aware download. Returns (df, meta, fetched_at, from_cache, stale)."""
        cached = self._load_cache(symbol, period)
        if cached and not force:
            fetched = pd.Timestamp(cached[1].get("fetched_at"))
            if pd.Timestamp.now(tz="UTC") - fetched < self.ttl:
                df, info = cached
                return df, info.get("meta", {}), info["fetched_at"], True, False
        try:
            df, meta = self._download(symbol, period)
            fetched_at = pd.Timestamp.now(tz="UTC").isoformat()
            self._save_cache(symbol, period, df, meta)
            return df, meta, fetched_at, False, False
        except DataError as err:
            if err.code in ("provider_unavailable", "rate_limited") and cached:
                df, info = cached
                return df, info.get("meta", {}), info["fetched_at"], True, True
            raise

    def fetch_symbol(self, symbol: str, period: str = "5y", force: bool = False,
                     min_rows: Optional[int] = None) -> StockData:
        """Download (or load from cache) one exact symbol. Raises DataError."""
        if period not in VALID_PERIODS:
            raise DataError(f"Unsupported period '{period}'. Choose one of {', '.join(VALID_PERIODS)}.", "bad_period")
        symbol = symbol.strip().upper()
        if not SYMBOL_RE.match(symbol):
            raise DataError(f"'{symbol}' is not a valid ticker symbol.", "invalid_symbol")
        df, meta, fetched_at, from_cache, stale = self._get_raw(symbol, period, force)
        return self._finish(symbol, df, meta, fetched_at, from_cache, stale, min_rows)

    def fetch_series(self, symbol: str, period: str = "5y", force: bool = False) -> Optional[pd.Series]:
        """Close-price series for a market-context symbol (index, FX, commodity). None if unavailable."""
        try:
            df, _, _, _, _ = self._get_raw(symbol, period, force)
        except DataError as err:
            log.info("context series %s unavailable: %s", symbol, err.message)
            return None
        s = df["Close"].replace([np.inf, -np.inf], np.nan).dropna()
        return s if len(s) > 60 else None

    def fetch_quote(self, symbol: str) -> Dict[str, Any]:
        """Latest available price: the last 1-minute bar of the most recent session, with its real timestamp.

        Yahoo does not promise real-time data (many exchanges are delayed), so the caller must show ``time``.
        """
        if yf is None:
            raise DataError("The 'yfinance' package is not installed. Run: pip install -r requirements.txt",
                            "missing_dependency")
        try:
            h = yf.Ticker(symbol).history(period="1d", interval="1m", auto_adjust=False, actions=False)
        except Exception as exc:
            raise _classify_exception(exc) from exc
        if h is not None and not h.empty:
            h = h[h["Close"].notna() & (h["Close"] > 0)]
        if h is None or h.empty:
            raise DataError(f"No live quote returned for '{symbol}'.", "no_data")
        ts = h.index[-1]
        return {"price": float(h["Close"].iloc[-1]), "time": ts.isoformat(), "date": str(ts.date()),
                "time_utc": ts.tz_convert("UTC").isoformat() if ts.tzinfo else ts.isoformat(),
                "source": "Yahoo Finance 1-minute bar"}

    def get_sector(self, symbol: str) -> Optional[Dict[str, str]]:
        """Sector/industry of a stock (cached on disk). None if the provider does not say."""
        path = self.cache_dir / "sectors.json"
        try:
            store = json.loads(path.read_text()) if path.exists() else {}
        except Exception:
            store = {}
        if symbol in store:
            return store[symbol] or None
        if yf is None:
            return None
        try:
            info = yf.Ticker(symbol).get_info() or {}
            val = {"sector": info.get("sector") or "", "industry": info.get("industry") or ""}
        except Exception as exc:
            log.info("sector lookup failed for %s: %s", symbol, exc)
            return None
        store[symbol] = val if (val["sector"] or val["industry"]) else {}
        try:
            path.write_text(json.dumps(store))
        except Exception:
            pass
        return store[symbol] or None

    def _finish(self, symbol, df, meta, fetched_at, from_cache, stale, min_rows=None) -> StockData:
        need = min_rows or self.min_rows
        df, warns = clean_prices(df, need, self.stale_after_days)
        df, partial = split_incomplete_session(df, meta)
        if len(df) < need:
            raise DataError(f"Only {len(df)} completed sessions available (need {need}).",
                            "insufficient_history")
        if stale:
            warns.insert(0, "Yahoo Finance could not be reached - showing previously cached data "
                            f"downloaded at {fetched_at[:19].replace('T', ' ')} UTC.")
        if partial:
            warns.append(f"Today's session ({partial['date']}) is not finished; it is shown as the latest quote only "
                         "and excluded from model training.")
        return StockData(symbol=symbol, df=df, meta=meta, fetched_at=fetched_at, from_cache=from_cache,
                         stale_cache=stale, partial_bar=partial, warnings=warns)

    # ---- resolve + load ---------------------------------------------------------
    def load(self, query: str, period: str = "5y", force: bool = False, min_rows: Optional[int] = None) -> StockData:
        """Resolve a company name / symbol to a ticker and load its history."""
        q = validate_symbol(query)
        looks_like_symbol = bool(SYMBOL_RE.match(q))
        tried: List[str] = []
        last_error: Optional[DataError] = None

        if looks_like_symbol:
            sym = q.upper()
            candidates = [sym]
            if "." not in sym and not sym.startswith("^"):
                candidates += [sym + s for s in INDIA_SUFFIXES]
            for cand in candidates:
                tried.append(cand)
                try:
                    data = self.fetch_symbol(cand, period, force, min_rows)
                except DataError as err:
                    last_error = err
                    if err.code in ("provider_unavailable", "rate_limited", "missing_dependency",
                                    "insufficient_history", "bad_period"):
                        raise
                    continue
                if cand != sym:
                    data.interpreted_from = q
                    data.warnings.insert(0, f"'{q}' was not found as a US symbol; using {cand}.")
                if "." not in sym:
                    data.alternatives = [s for s in self.search(q, 6) if s["symbol"] != data.symbol]
                return data

        # Not a plain symbol (company name) or symbol not found: use Yahoo search.
        results = self.search(q, 8)
        if not results:
            raise DataError(
                f"Could not find a stock matching '{q}'. Try the exact ticker with its exchange suffix, "
                "e.g. RELIANCE.NS (NSE), RELIANCE.BO (BSE), AAPL, MSFT.", "not_found") from last_error
        for i, res in enumerate(results):
            if res["symbol"] in tried:
                continue
            try:
                data = self.fetch_symbol(res["symbol"], period, force, min_rows)
            except DataError as err:
                last_error = err
                if err.code in ("provider_unavailable", "rate_limited", "missing_dependency"):
                    raise
                continue
            data.interpreted_from = q
            data.alternatives = [r for r in results if r["symbol"] != data.symbol][:6]
            data.warnings.insert(0, f"Interpreted '{q}' as {res['symbol']} ({res['name']}, {res['exchange']}).")
            return data
        raise last_error or DataError(f"No usable data found for '{q}'.", "no_data")
