# Stock Trend Predictor 2.0

Stock Trend Predictor is a Python web app that forecasts stock prices. You search any Indian or US stock, the app downloads its real daily price history from Yahoo Finance, trains five models on it, tests them on data they have never seen, and shows next-day and next-month (21 trading days) forecasts with an uncertainty range.

It includes a Flask JSON API, a browser dashboard with charts, a command-line mode, a result cache, and automated tests.

> **Important:** Forecasts are statistical estimates. They are not guaranteed and not investment advice. For most stocks, models do **not** reliably beat the simple "price stays the same" baseline. The app measures this on unseen data and tells you when it happens.

## Demo

- Dashboard: http://127.0.0.1:5000
- Health check: http://127.0.0.1:5000/api/health
- Default stock: `RELIANCE.NS`
- Sample data: `sample_data/RELIANCE.NS_20251016.csv`

No login or account is needed.

## Features

- Search by company name or symbol. Example: `Reliance Industries`, `TCS.NS`, `AAPL`.
- Works with NSE (`.NS`), BSE (`.BO`) and US stocks (no suffix).
- Real daily data from Yahoo Finance (`yfinance`): Open, High, Low, Close, Adj Close, Volume.
- Five models: Ridge, Random Forest, Gradient Boosting, XGBoost and LSTM.
- Next-day and next-month (21 sessions) forecasts from every model.
- 80% forecast range, built from real past errors.
- Honest testing: metrics come from a block of data never used for training, model selection or range calibration.
- Metrics: MAE, RMSE, MAPE, directional accuracy, skill against the naive and drift baselines, and range coverage.
- A reliability label for each stock: Moderate, Low, or None demonstrated.
- Backtest of a long/flat strategy against buy-and-hold, including a 0.1% cost per position change.
- Charts: price with forecast, volume, RSI (14), MACD (12, 26, 9), model comparison, backtest. All charts are plain SVG with no chart library.
- Technical indicators: SMA, EMA, RSI, MACD, Bollinger Bands, ATR, OBV, volatility, trend summary.
- Smart symbol search. A bare symbol like `RELIANCE` is tried as US first, then `.NS`, then `.BO`. The dashboard shows "other matches" so you can switch.
- Data cache (60 minutes). If Yahoo is down, the app falls back to old cached data and warns you.
- Result cache. Models retrain only when the data or the config changes.
- Data warnings for gaps, old data and sudden price jumps. Bad rows are dropped. Missing prices are never filled in or invented.
- Forecast dates skip exchange holidays when the `holidays` package is installed.
- Live progress bar while models train.
- Command-line mode with optional JSON output.
- Clear error messages. Stack traces never reach the browser.

## Project Layout

```
Stock-Trend-Predictor/
├── app.py                  Flask server: JSON API + dashboard
├── main.py                 Entry point for dashboard or command line
├── requirements.txt        Python packages
├── setup.bat               Windows one-time setup
├── quick_launch.bat        Windows one-click start
├── config/
│   └── config.yaml         All settings
├── src/
│   ├── data_fetcher.py         Yahoo Finance download, search, symbol lookup, cache
│   ├── data_processor.py       Checks and cleans data, warnings
│   ├── technical_indicators.py SMA, EMA, RSI, MACD, Bollinger, ATR, OBV
│   ├── feature_engineer.py     Builds features and targets
│   ├── model_trainer.py        Ridge, Random Forest, Gradient Boosting, XGBoost
│   ├── lstm_model.py           LSTM (PyTorch, or built-in NumPy version)
│   ├── evaluator.py            Metrics, baselines, reliability, backtest
│   ├── predictor.py            Forecast paths, ranges, dates
│   ├── pipeline.py             Runs all steps, result cache
│   ├── utils.py                Helper functions
│   └── portfolio_analyzer.py   Extra helper, not used by the dashboard
├── web/
│   ├── templates/index.html    Dashboard page
│   └── static/                 app.js, charts.js, style.css
├── tests/                      Automated tests
├── sample_data/                Example price file
└── legacy/                     Old Tkinter app and notebooks (not maintained)
```

## Stack

- Python
- Flask
- yfinance
- pandas and NumPy
- scikit-learn
- XGBoost
- PyTorch (optional)
- PyYAML
- Plain JavaScript, HTML and CSS
- pytest

## How It Works

| Step | File | What it does |
|---|---|---|
| 1. Collect | `src/data_fetcher.py` | Downloads data, searches names, resolves symbols, caches CSV files |
| 2. Clean | `src/data_processor.py` | Checks data, drops bad rows, adds warnings, removes today's unfinished session |
| 3. Indicators | `src/technical_indicators.py` | Calculates the indicators shown on the dashboard |
| 4. Features | `src/feature_engineer.py` | Builds 34 features from adjusted prices. Targets are the total log-return for 1 to 21 days ahead |
| 5. Train | `src/model_trainer.py`, `src/lstm_model.py` | Trains one multi-day model per algorithm |
| 6. Evaluate | `src/evaluator.py` | Scores every model on unseen data and runs the backtest |
| 7. Forecast | `src/predictor.py` | Builds price paths and 80% ranges |
| 8. Run and cache | `src/pipeline.py` | Runs steps 1 to 7 and saves the result |
| 9. Show | `app.py`, `web/` | Serves the API and the dashboard |

### Validation Design

- Data is split in time order: 70% train, 15% validation, 15% test.
- A 21-session gap sits between blocks, so no target overlaps another block.
- Scalers are fit on training rows only.
- The validation block is used to pick the best model and calibrate the range.
- The test block is used only for the final report and the backtest.
- Final forecasts come from models refit on all data.
- LSTM early stopping uses a small block inside the training period, so the validation block stays clean.

### Reliability Label

After testing, each stock gets one label:

- **Moderate**: models beat the naive baseline by a useful margin.
- **Low**: models beat the baseline only slightly.
- **None demonstrated**: models do not beat the baseline. Treat the forecast as a rough guess.

### Data Honesty

Data is delayed, latest available data from Yahoo Finance. It is never shown as real-time. The dashboard shows the last completed daily close and the latest quote with its time.

## Local Setup

You need Python 3.9 or newer and an internet connection.

**1. Create a virtual environment**

```bash
python -m venv venv
```

**2. Activate it**

```bash
# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

**3. Install packages**

```bash
pip install -r requirements.txt
```

**4. Start the app**

```bash
python app.py
```

**5. Open the dashboard**

http://127.0.0.1:5000

### Windows Shortcut

1. Run `setup.bat` once. It creates the environment and installs packages.
2. Run `quick_launch.bat` every time after. It starts the server and opens the browser.

### Optional Packages

- `torch`: faster LSTM. Without it, the app uses a built-in NumPy LSTM automatically.
- `holidays`: exchange holiday calendars for NSE and NYSE. Without it, forecast dates assume Monday to Friday.

## Using the Dashboard

1. Type a company name or symbol in the search box. Suggestions appear as you type.
2. Pick the history length: `2y`, `5y`, `10y` or `Max`.
3. Click **Analyze**. A progress bar shows each step.
4. Read the results: price chart, forecasts, model comparison, evaluation, backtest and trend statistics.
5. Use the **Forecast model** menu to switch between models on the chart.
6. Click **Refresh data** to download fresh data from Yahoo Finance.
7. Tick **Force retrain** to retrain even if the data has not changed.

### Search Examples

| Type | Example |
|---|---|
| NSE stock | `RELIANCE.NS`, `TCS.NS` |
| BSE stock | `INFY.BO` |
| US stock | `AAPL`, `MSFT` |
| Company name | `Reliance Industries` |

`INFY` is the US ADR on NYSE. `INFY.NS` is the NSE listing.

## Command Line

```bash
python main.py --mode cli --ticker TCS.NS --period 5y
```

| Option | Meaning | Default |
|---|---|---|
| `--mode` | `ui` for the dashboard, `cli` for the terminal | `ui` |
| `--ticker` | Company name or symbol | `RELIANCE.NS` |
| `--period` | `2y`, `5y`, `10y` or `max` | `5y` |
| `--refresh` | Download fresh data | off |
| `--retrain` | Retrain even if data is unchanged | off |
| `--json` | Print the full result as JSON | off |

`python main.py --mode ui` starts the dashboard, same as `python app.py`.

The command line prints: stock name and exchange, last close, a table of next-day and month-end forecasts for every model, the selected model's 80% range, the reliability label, test metrics, warnings and the disclaimer.

## API

The base address is `http://127.0.0.1:5000`.

| Method | Route | Purpose |
|---|---|---|
| GET | `/` | Dashboard page |
| GET | `/api/health` | Server status, version, and whether XGBoost and PyTorch are installed |
| GET | `/api/search?q=...` | Search stocks. Needs at least 2 characters. Returns up to 8 matches |
| POST | `/api/analyze` | Starts an analysis. Returns a `job_id` |
| GET | `/api/jobs/<job_id>` | Shows progress, then the result or an error |

### Start an analysis

```bash
curl -X POST http://127.0.0.1:5000/api/analyze \
  -H "Content-Type: application/json" \
  -d '{"query": "TCS.NS", "period": "5y", "refresh": false, "retrain": false}'
```

Request fields:

| Field | Required | Meaning |
|---|---|---|
| `query` | Yes | Company name or symbol. Maximum 60 characters |
| `period` | No | `1y`, `2y`, `5y`, `10y` or `max`. Default comes from config |
| `refresh` | No | Download fresh data |
| `retrain` | No | Retrain models |

Response:

```json
{ "job_id": "a1b2c3d4..." }
```

### Check a job

```bash
curl http://127.0.0.1:5000/api/jobs/<job_id>
```

- While running: `status` is `running`, with `progress` (0 to 1) and `message`.
- When finished: `status` is `done` and `result` holds the full analysis.
- On failure: `status` is `error` with an `error` object that has `code` and `message`.

Jobs are kept for 1 hour. The server runs up to 2 analyses at the same time.

### Error codes

| Code | Meaning |
|---|---|
| `empty_query` | The search box was empty |
| `bad_period` | Period is not allowed |
| `invalid_symbol` | Search text is too long or not a valid symbol |
| `unknown_job` | Job not found or expired |
| `internal_error` | Unexpected problem. Details are in the server log |

Data problems (symbol not found, too little data, Yahoo Finance down) also return a `code` and a plain `message`.

## Configuration

All settings are in `config/config.yaml`.

| Setting | Default | Meaning |
|---|---|---|
| `app.host` | `127.0.0.1` | Server address |
| `app.port` | `5000` | Server port |
| `data.default_ticker` | `RELIANCE.NS` | Stock shown in the search box |
| `data.default_period` | `5y` | Default history length |
| `data.cache_dir` | `.cache` | Where cache files are saved |
| `data.data_cache_ttl_minutes` | `60` | Download again if data is older than this |
| `data.min_rows` | `300` | Minimum clean daily rows needed |
| `data.stale_after_days` | `10` | Warn if the last row is older than this |
| `modeling.horizon` | `21` | Sessions ahead (about 1 month) |
| `modeling.eval_horizons` | `[1, 5, 10, 21]` | Horizons shown in evaluation |
| `modeling.seq_len` | `20` | LSTM look-back window |
| `modeling.split` | `train 0.70, val 0.15` | Time-order split. The rest is test |
| `modeling.interval_quantiles` | `[0.10, 0.90]` | 80% range |
| `modeling.random_state` | `42` | Fixed seed for repeatable results |
| `modeling.models` | all five | Models to train |
| `modeling.lstm.*` | see file | Hidden size, epochs, patience, batch size, learning rate, seeds, backend (`auto`, `torch` or `numpy`) |
| `backtest.transaction_cost` | `0.001` | 0.1% cost per position change |

## Tests

```bash
python -m tests.run_all
```

Or with pytest:

```bash
python -m pytest
```

Tests cover the core code (`tests/test_core.py`) and the pipeline and API (`tests/test_pipeline_and_api.py`).

## Cache

- Downloaded data and trained results are saved in `.cache/`.
- To clear everything, delete the `.cache/` folder.
- To retrain one stock, use **Force retrain** or `--retrain`.
- To download new data for one stock, use **Refresh data** or `--refresh`.

## Troubleshooting

| Problem | Fix |
|---|---|
| Stock not found | Add `.NS` or `.BO` for Indian stocks. Try the company name instead |
| Not enough data | Choose a longer history. At least 300 clean daily rows are needed |
| Yahoo Finance error or rate limit | Wait a few minutes and try again. Old cached data is used if it exists |
| Training is slow | Install `torch` for a faster LSTM, or remove `lstm` from `modeling.models` |
| `ModuleNotFoundError` | Activate the virtual environment, then run `pip install -r requirements.txt` |
| Port 5000 already in use | Change `app.port` in `config/config.yaml` |
| Old numbers on screen | Click **Refresh data** and tick **Force retrain** |

## Known Limitations

- Only daily price and volume data is used. News, company results and economic data are not included.
- Daily stock returns are very hard to predict.
- Settings are fixed defaults, not tuned for each stock.
- Ranges come from past errors. Big events like results or crashes can fall outside them.
- Month paths come from direct multi-day models. They show the expected path, not a simulated price path.
- Yahoo Finance is an unofficial source. It can rate-limit, change, or have gaps. Small Indian stocks may have thin data.
- Without the `holidays` package, forecast dates assume Monday to Friday.
- The old Tkinter app and notebooks are in `legacy/` and are not maintained.

## Disclaimer

This project is for learning and research. It is not financial advice. Do not make real investment decisions based only on these forecasts.
