"""Market and sector context series (indices, VIX, FX, commodities, sector index/ETF) for the models.

Timing rule (no look-ahead): an asset whose daily close happens AFTER the stock's close (e.g. crude oil or
USD/INR for an Indian stock) is lagged by one session (``lag: 1``); assets that close at or before the stock's
close use same-day values (``lag: 0``).
"""
from __future__ import annotations

import logging
import re
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

INDIA_SECTOR = {"Technology": ("^CNXIT", "Nifty IT"), "Financial Services": ("^NSEBANK", "Nifty Bank"),
                "Healthcare": ("^CNXPHARMA", "Nifty Pharma"), "Consumer Defensive": ("^CNXFMCG", "Nifty FMCG"),
                "Energy": ("^CNXENERGY", "Nifty Energy"), "Basic Materials": ("^CNXMETAL", "Nifty Metal"),
                "Real Estate": ("^CNXREALTY", "Nifty Realty"), "Industrials": ("^CNXINFRA", "Nifty Infra"),
                "Communication Services": ("^CNXMEDIA", "Nifty Media")}
INDIA_INDUSTRY = (("auto", ("^CNXAUTO", "Nifty Auto")),)
US_SECTOR = {"Technology": ("XLK", "Tech sector ETF"), "Financial Services": ("XLF", "Financials ETF"),
             "Healthcare": ("XLV", "Healthcare ETF"), "Energy": ("XLE", "Energy ETF"),
             "Consumer Cyclical": ("XLY", "Consumer Discretionary ETF"), "Consumer Defensive": ("XLP", "Staples ETF"),
             "Industrials": ("XLI", "Industrials ETF"), "Basic Materials": ("XLB", "Materials ETF"),
             "Utilities": ("XLU", "Utilities ETF"), "Real Estate": ("XLRE", "Real estate ETF"),
             "Communication Services": ("XLC", "Communication ETF")}


def region_of(symbol: str) -> str:
    s = symbol.upper()
    if s.endswith((".NS", ".BO")):
        return "india"
    if "." not in s and not s.startswith("^"):
        return "us"
    return "other"


def slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def sector_asset(region: str, sector_info: Optional[dict]) -> Optional[Tuple[str, str]]:
    if not sector_info:
        return None
    sector, industry = sector_info.get("sector", ""), (sector_info.get("industry") or "").lower()
    if region == "india":
        for key, val in INDIA_INDUSTRY:
            if key in industry:
                return val
        return INDIA_SECTOR.get(sector)
    if region == "us":
        return US_SECTOR.get(sector)
    return None


def load_context(fetcher, cfg: dict, symbol: str, index: pd.DatetimeIndex, period: str,
                 force: bool = False) -> Tuple[Optional[pd.DataFrame], List[dict], List[str]]:
    """Return (aligned level frame or None, used-asset descriptions, warnings)."""
    region = region_of(symbol)
    assets = [dict(a) for a in cfg.get(region, cfg.get("other", []))]
    sec = sector_asset(region, fetcher.get_sector(symbol) if region != "other" else None)
    if sec:
        assets.append({"symbol": sec[0], "name": sec[1], "lag": 0, "sector": True})
    cols: Dict[str, pd.Series] = {}
    used, warns = [], []
    for a in assets:
        if a["symbol"].upper() == symbol.upper():
            continue
        s = fetcher.fetch_series(a["symbol"], period, force)
        if s is None:
            warns.append(f"Market input '{a['name']}' ({a['symbol']}) could not be downloaded and was skipped.")
            continue
        s = s[~s.index.duplicated(keep="last")].sort_index()
        full = s.reindex(s.index.union(index)).ffill(limit=5).reindex(index)
        if a.get("lag", 0):
            full = full.shift(int(a["lag"]))
        if full.notna().mean() < 0.97:
            warns.append(f"Market input '{a['name']}' covers too little of this stock's history and was skipped.")
            continue
        key = slug(a["name"])
        cols[key] = full
        used.append({"symbol": a["symbol"], "name": a["name"], "lag": int(a.get("lag", 0)), "key": key,
                     "sector": bool(a.get("sector"))})
    if not cols:
        return None, [], warns
    return pd.DataFrame(cols, index=index), used, warns
