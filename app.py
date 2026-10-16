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

from src.data_fetcher import VALID_PERIODS, DataError
from src.pipeline import Analyzer, load_config

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


def _run(job_id, query, period, refresh, retrain):
    def progress(p, msg):
        with jobs_lock:
            jobs[job_id].update(progress=round(float(p), 3), message=msg)
    try:
        result = analyzer.analyze(query, period, force_refresh=refresh, force_retrain=retrain, progress=progress)
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
    executor.submit(_run, job_id, query, period, bool(body.get("refresh")), bool(body.get("retrain")))
    return jsonify(job_id=job_id)


@app.get("/api/jobs/<job_id>")
def job(job_id):
    with jobs_lock:
        j = jobs.get(job_id)
        if not j:
            return jsonify(error={"code": "unknown_job", "message": "Job not found or expired."}), 404
        return jsonify({k: v for k, v in j.items() if k != "created"})


if __name__ == "__main__":
    app.run(host=cfg["app"]["host"], port=cfg["app"]["port"], debug=False, threaded=True)
