#!/usr/bin/env python3
"""Flask server: JSON API + dashboard for the Stock Trend Predictor.

Run:  python app.py        then open http://127.0.0.1:5000
"""
import logging
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor

from flask import Flask, jsonify, render_template, request

import pandas as pd

from src.compare import compare_stocks
from src.data_fetcher import VALID_PERIODS, DataError
from src.pipeline import Analyzer, clean, load_config
from src.tracker import resolve_pending, summarize

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("app")

app = Flask(__name__, template_folder="web/templates", static_folder="web/static")
cfg = load_config()
analyzer = Analyzer(cfg)
executor = ThreadPoolExecutor(max_workers=2)
jobs, jobs_lock = {}, threading.Lock()


def _prune():
    cutoff = time.time() - 3600
    with jobs_lock:
        for k in [k for k, v in jobs.items() if v["created"] < cutoff]:
            del jobs[k]


def _run(job_id, query, period, refresh, retrain, options):
    def progress(p, msg):
        with jobs_lock:
            jobs[job_id].update(progress=round(float(p), 3), message=msg)
    try:
        result = analyzer.analyze(query, period, force_refresh=refresh, force_retrain=retrain, progress=progress,
                                  options=options)
        with jobs_lock:
            jobs[job_id].update(status="done", progress=1.0, result=result, message="Done")
    except DataError as err:
        log.info("data error for %r: %s", query, err.message)
        with jobs_lock:
            jobs[job_id].update(status="error", error={"code": err.code, "message": err.message})
    except Exception:  # never leak a stack trace to the browser
        log.exception("analysis failed for %r", query)
        with jobs_lock:
            jobs[job_id].update(status="error", error={
                "code": "internal_error",
                "message": "Something went wrong while analysing this stock. Details were written to the server log."})


@app.get("/")
def index():
    return render_template("index.html", default_ticker=cfg["data"]["default_ticker"],
                           default_period=cfg["data"]["default_period"], periods=VALID_PERIODS)


@app.get("/api/health")
def health():
    from src.model_trainer import HAS_XGB
    from src.lstm_model import HAS_TORCH
    return jsonify(status="ok", xgboost=HAS_XGB, torch=HAS_TORCH, version=cfg["app"]["version"])


@app.get("/api/search")
def search():
    q = request.args.get("q", "").strip()
    if len(q) < 2:
        return jsonify(results=[])
    return jsonify(results=analyzer.fetcher.search(q, 8))


@app.post("/api/analyze")
def analyze():
    body = request.get_json(silent=True) or {}
    query = str(body.get("query", "")).strip()
    period = str(body.get("period") or cfg["data"]["default_period"])
    if not query:
        return jsonify(error={"code": "empty_query", "message": "Enter a company name or stock symbol."}), 400
    if period not in VALID_PERIODS:
        return jsonify(error={"code": "bad_period", "message": f"Period must be one of {', '.join(VALID_PERIODS)}."}), 400
    if len(query) > 60:
        return jsonify(error={"code": "invalid_symbol", "message": "Search text is too long."}), 400
    _prune()
    job_id = uuid.uuid4().hex
    with jobs_lock:
        jobs[job_id] = {"status": "running", "progress": 0.0, "message": "Queued", "created": time.time()}
    options = {k: bool(body[k]) for k in ("use_context", "tune") if k in body}
    executor.submit(_run, job_id, query, period, bool(body.get("refresh")), bool(body.get("retrain")), options)
    return jsonify(job_id=job_id)


@app.get("/api/jobs/<job_id>")
def job(job_id):
    with jobs_lock:
        j = jobs.get(job_id)
        if not j:
            return jsonify(error={"code": "unknown_job", "message": "Job not found or expired."}), 404
        return jsonify({k: v for k, v in j.items() if k != "created"})


def _err(err: DataError, status=400):
    return jsonify(error={"code": err.code, "message": err.message}), status


# ------------------------------------------------------------------ watchlist
def _wl_row(item, refresh=False):
    row = dict(item)
    try:
        row["quote"] = analyzer.quote_summary(item["symbol"], force=refresh)
    except DataError as err:
        row["quote"], row["error"] = None, err.message
    except Exception:
        log.exception("watchlist quote failed for %s", item["symbol"])
        row["quote"], row["error"] = None, "Could not load the latest price."
    row["forecast"] = analyzer.cached_summary(item["symbol"])
    return row


@app.get("/api/watchlist")
def watchlist():
    refresh = request.args.get("refresh") == "1"
    items = analyzer.store.wl_list()
    with ThreadPoolExecutor(max_workers=4) as pool:
        rows = list(pool.map(lambda i: _wl_row(i, refresh), items))
    return jsonify(items=clean(rows), disclaimer="Delayed / latest-available data from Yahoo Finance; not real-time. "
                   "Forecasts are estimates, not guarantees.")


@app.post("/api/watchlist")
def watchlist_add():
    query = str((request.get_json(silent=True) or {}).get("query", "")).strip()
    if not query or len(query) > 60:
        return jsonify(error={"code": "empty_query", "message": "Enter a company name or stock symbol."}), 400
    try:
        d = analyzer.fetcher.load(query, "1y", min_rows=30)
    except DataError as err:
        return _err(err)
    analyzer.store.wl_add(d.symbol, d.meta.get("longName") or d.meta.get("shortName") or d.symbol,
                          d.meta.get("fullExchangeName") or d.meta.get("exchangeName") or "",
                          d.meta.get("currency") or "", pd.Timestamp.now(tz="UTC").isoformat())
    item = next(i for i in analyzer.store.wl_list() if i["symbol"] == d.symbol)
    return jsonify(item=clean(_wl_row(item)))


@app.delete("/api/watchlist/<path:symbol>")
def watchlist_remove(symbol):
    return jsonify(removed=analyzer.store.wl_remove(symbol.upper()))


# ------------------------------------------------------------------ compare
@app.post("/api/compare")
def compare():
    body = request.get_json(silent=True) or {}
    symbols, period = body.get("symbols") or [], str(body.get("period") or "5y")
    if not isinstance(symbols, list) or any(not isinstance(x, str) or len(x) > 60 for x in symbols):
        return jsonify(error={"code": "bad_request", "message": "Provide 2 to 4 stock names or symbols."}), 400
    if period not in VALID_PERIODS:
        return jsonify(error={"code": "bad_period", "message": f"Period must be one of {', '.join(VALID_PERIODS)}."}), 400
    try:
        return jsonify(clean(compare_stocks(analyzer.fetcher, symbols, period, bool(body.get("refresh")))))
    except DataError as err:
        return _err(err)
    except Exception:
        log.exception("compare failed")
        return jsonify(error={"code": "internal_error", "message": "Comparison failed; see the server log."}), 500


# ------------------------------------------------------------------ forecast tracking
@app.get("/api/tracking")
def tracking():
    symbol = (request.args.get("symbol") or "").upper() or None
    return jsonify(clean(summarize(analyzer.store, symbol)))


@app.post("/api/tracking/resolve")
def tracking_resolve():
    """Download latest data for every symbol with open forecasts and record outcomes that are now known."""
    symbol = str((request.get_json(silent=True) or {}).get("symbol") or "").upper() or None
    now = pd.Timestamp.now(tz="UTC").isoformat()
    resolved, problems = 0, []
    for sym in ([symbol] if symbol else analyzer.store.tracked_symbols()):
        if not analyzer.store.pending(sym):
            continue
        try:
            d = analyzer.fetcher.fetch_symbol(sym, "1y", force=True, min_rows=30)
            resolved += resolve_pending(analyzer.store, sym, d.df, now)
        except DataError as err:
            problems.append(f"{sym}: {err.message}")
    return jsonify(resolved=resolved, problems=problems, summary=clean(summarize(analyzer.store, symbol)))


if __name__ == "__main__":
    app.run(host=cfg["app"]["host"], port=cfg["app"]["port"], debug=False, threaded=True)
