#!/usr/bin/env python3
"""Command-line entry point.

  python main.py --mode ui                       # start the web dashboard (same as: python app.py)
  python main.py --mode cli --ticker TCS.NS      # analyse one stock in the terminal
"""
import argparse
import json
import logging
import sys

from src.data_processor import DataError
from src.pipeline import Analyzer


def run_cli(query, period, refresh, retrain, as_json):
    an = Analyzer()
    last = {"p": -1}

    def progress(p, msg):
        if int(p * 20) != last["p"]:
            last["p"] = int(p * 20)
            print(f"[{p * 100:3.0f}%] {msg}", file=sys.stderr)

    try:
        r = an.analyze(query, period, force_refresh=refresh, force_retrain=retrain, progress=progress)
    except DataError as e:
        print(f"ERROR ({e.code}): {e.message}", file=sys.stderr)
        return 2
    if as_json:
        print(json.dumps(r))
        return 0
    m, d, f = r["meta"], r["data"], r["forecast"]
    print(f"\n{m['name']} ({m['symbol']}) - {m['exchange']} [{m['currency']}]")
    print(f"Last completed close {d['last_close']:.2f} on {d['last_close_date']}  ({d['freshness']})")
    print(f"Models trained on {d['rows']} sessions {d['start']} -> {d['end']}\n")
    print(f"{'Model':24}{'Next day':>12}{'chg%':>8}{'Month-end':>12}{'chg%':>8}")
    for k, v in f["models"].items():
        star = "*" if k == f["selected"] else " "
        print(f"{star}{v['label']:23}{v['next_day']['pred_close']:12.2f}{v['next_day']['change_pct']:8.2f}"
              f"{v['month_end']['pred_close']:12.2f}{v['month_end']['change_pct']:8.2f}")
    sel = f["models"][f["selected"]]
    print(f"\n* selected: {sel['label']}; {f['interval_label']} for month-end: "
          f"{sel['month_end']['low']:.2f} - {sel['month_end']['high']:.2f}")
    print(f"Reliability on unseen data: {f['reliability']} - {f['reliability_note']}")
    print("\nHeld-out TEST metrics (next day):")
    for row in r["evaluation"]["test"]:
        x = row["horizons"]["1"]
        da = "-" if x["dir_acc"] is None else f"{x['dir_acc']:.1f}%"
        print(f"  {row['label']:24} MAE {x['mae']:.3f}  RMSE {x['rmse']:.3f}  dir.acc {da}")
    for w in r["warnings"]:
        print("note:", w)
    print("\n" + r["disclaimer"])
    return 0


def main():
    ap = argparse.ArgumentParser(description="Stock Trend Predictor")
    ap.add_argument("--mode", choices=["ui", "cli"], default="ui")
    ap.add_argument("--ticker", default="RELIANCE.NS", help="company name or symbol")
    ap.add_argument("--period", default="5y", choices=["2y", "5y", "10y", "max"])
    ap.add_argument("--refresh", action="store_true", help="re-download data")
    ap.add_argument("--retrain", action="store_true", help="retrain even if data is unchanged")
    ap.add_argument("--json", action="store_true", help="print full JSON result (cli mode)")
    a = ap.parse_args()
    logging.basicConfig(level=logging.WARNING)
    if a.mode == "ui":
        import app
        app.app.run(host=app.cfg["app"]["host"], port=app.cfg["app"]["port"], threaded=True)
        return 0
    return run_cli(a.ticker, a.period, a.refresh, a.retrain, a.json)


if __name__ == "__main__":
    sys.exit(main())
