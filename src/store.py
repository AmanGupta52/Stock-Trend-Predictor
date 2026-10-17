"""Small SQLite store for the watchlist and the forecast log."""
from __future__ import annotations

import sqlite3
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

SCHEMA = """
CREATE TABLE IF NOT EXISTS watchlist(
  symbol TEXT PRIMARY KEY, name TEXT, exchange TEXT, currency TEXT, added_at TEXT);
CREATE TABLE IF NOT EXISTS forecasts(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  symbol TEXT NOT NULL, model TEXT NOT NULL, kind TEXT NOT NULL,
  made_at TEXT NOT NULL, base_date TEXT NOT NULL, base_close REAL NOT NULL,
  target_date TEXT NOT NULL, pred_close REAL NOT NULL, low REAL, high REAL,
  selected INTEGER DEFAULT 0, currency TEXT,
  actual_close REAL, actual_date TEXT, resolved_at TEXT,
  UNIQUE(symbol, model, kind, base_date));
CREATE INDEX IF NOT EXISTS ix_fc_symbol ON forecasts(symbol, actual_close);
"""


class Store:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()
        with self._conn() as c:
            c.executescript(SCHEMA)

    def _conn(self):
        c = sqlite3.connect(str(self.path), timeout=15)
        c.row_factory = sqlite3.Row
        return c

    def _run(self, sql: str, args: tuple = (), many: Optional[List[tuple]] = None, fetch: bool = False):
        with self.lock:
            c = self._conn()
            try:
                cur = c.executemany(sql, many) if many is not None else c.execute(sql, args)
                rows = [dict(r) for r in cur.fetchall()] if fetch else None
                c.commit()
                return rows if fetch else cur.rowcount
            finally:
                c.close()

    # ---- watchlist
    def wl_add(self, symbol: str, name: str, exchange: str, currency: str, added_at: str) -> None:
        self._run("INSERT OR REPLACE INTO watchlist(symbol,name,exchange,currency,added_at) VALUES(?,?,?,?,?)",
                  (symbol, name, exchange, currency, added_at))

    def wl_remove(self, symbol: str) -> int:
        return self._run("DELETE FROM watchlist WHERE symbol=?", (symbol,))

    def wl_list(self) -> List[Dict[str, Any]]:
        return self._run("SELECT * FROM watchlist ORDER BY added_at", fetch=True)

    # ---- forecasts
    def log_forecasts(self, rows: List[Dict[str, Any]]) -> int:
        """Insert forecasts; an existing (symbol, model, kind, base_date) forecast is never overwritten."""
        if not rows:
            return 0
        cols = ["symbol", "model", "kind", "made_at", "base_date", "base_close", "target_date", "pred_close",
                "low", "high", "selected", "currency"]
        return self._run(f"INSERT OR IGNORE INTO forecasts({','.join(cols)}) VALUES({','.join('?' * len(cols))})",
                         many=[tuple(r.get(c) for c in cols) for r in rows])

    def pending(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        if symbol:
            return self._run("SELECT * FROM forecasts WHERE actual_close IS NULL AND symbol=?", (symbol,), fetch=True)
        return self._run("SELECT * FROM forecasts WHERE actual_close IS NULL", fetch=True)

    def resolve(self, fid: int, actual_close: float, actual_date: str, resolved_at: str) -> None:
        self._run("UPDATE forecasts SET actual_close=?, actual_date=?, resolved_at=? WHERE id=?",
                  (actual_close, actual_date, resolved_at, fid))

    def forecasts(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        if symbol:
            return self._run("SELECT * FROM forecasts WHERE symbol=? ORDER BY made_at DESC, id DESC", (symbol,), fetch=True)
        return self._run("SELECT * FROM forecasts ORDER BY made_at DESC, id DESC", fetch=True)

    def tracked_symbols(self) -> List[str]:
        return [r["symbol"] for r in self._run("SELECT DISTINCT symbol FROM forecasts ORDER BY symbol", fetch=True)]
