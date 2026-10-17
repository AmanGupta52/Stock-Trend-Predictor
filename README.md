# Stock Trend Predictor 2.1

Search any Indian or US stock, download its real daily history from **Yahoo Finance (`yfinance`)**, train
ML (Ridge, Random Forest, Gradient Boosting, XGBoost) and deep-learning (LSTM) models on it, test them on data
they never saw, and view next-day and next-month (21 sessions) forecasts with uncertainty ranges.

> Forecasts are statistical estimates, **not** guaranteed outcomes and not investment advice. For most stocks,
> models do **not** reliably beat the naive "price stays the same" baseline - the app measures this on held-out
> data and tells you when that is the case.

## Run it
```bash
python -m venv venv
venv\Scripts\activate            # Windows   (macOS/Linux: source venv/bin/activate)
pip install -r requirements.txt
python app.py                    # then open http://127.0.0.1:5000
```
Windows shortcut: `setup.bat` once, then `quick_launch.bat`.
CLI: `python main.py --mode cli --ticker TCS.NS --period 5y` (add `--refresh`, `--retrain`, `--json`).
Tests: `python -m tests.run_all` (or `python -m pytest`).

Search examples: `RELIANCE.NS`, `TCS.NS`, `INFY.BO`, `AAPL`, `MSFT`, or a name such as `Reliance Industries`.
A bare symbol like `RELIANCE` is tried as a US symbol, then `.NS`, then `.BO`; the dashboard shows the exchange and
"other matches" so you can switch (e.g. `INFY` is the NYSE ADR, `INFY.NS` the NSE listing).

## How it works
| Stage | Module | Notes |
|---|---|---|
| Data collection | `src/data_fetcher.py` | yfinance (`Open/High/Low/Close/Adj Close/Volume`), name search, symbol resolution, CSV cache (60 min TTL, stale-cache fallback if Yahoo is down), clear errors |
| Preprocessing | `src/data_processor.py` | validation, drops bad rows (never fills/invents prices), gap/staleness/jump warnings, splits off today's unfinished session |
| Indicators | `src/technical_indicators.py` | SMA/EMA, RSI, MACD, Bollinger, ATR, OBV, volatility, trend summary |
| Features | `src/feature_engineer.py` | 34 leak-free features on adjusted prices; targets = cumulative log-return for h = 1..21 |
| Models | `src/model_trainer.py`, `src/lstm_model.py` | one multi-horizon model per algorithm; PyTorch LSTM if installed, otherwise a built-in NumPy LSTM |
| Evaluation | `src/evaluator.py` | MAE, RMSE, MAPE, directional accuracy, skill vs naive & drift baselines, interval coverage, backtest |
| Forecast | `src/predictor.py` | price paths and 80% empirical ranges, exchange-holiday-aware dates (if `holidays` installed) |
| Orchestration | `src/pipeline.py` | result cache keyed by symbol + last bar + config (retrains only when data/config change) |
| UI | `app.py`, `web/` | Flask JSON API + responsive dashboard (dependency-free SVG charts) |

**Validation design.** Chronological 70/15/15 split with a 21-session gap between blocks (targets look 21 days ahead, so
blocks never overlap). Scalers are fit on training rows only. Model selection and interval calibration use the validation
block; the reported test metrics and backtest use a later block that was never used for fitting, selection or calibration.
Final forecasts come from models refit on all data. The LSTM's early stopping uses a small block inside the training
period so the validation block stays clean.

**Data honesty.** Data is *delayed / latest available* from Yahoo Finance and is never labelled real-time.
The dashboard shows the last completed daily close and the provider's latest quote with its timestamp.

## New in 2.1
- **Watchlist** (`/api/watchlist`, SQLite `.cache/app.db`): save stocks; one page shows latest price, change vs prior close and the next-day / month-end forecast from your most recent analysis ("Update forecast" retrains).
- **Compare** (`src/compare.py`): 2-4 stocks, adjusted prices rebased to 100 on the first common date, drawdown chart, and a table of returns, volatility, Sharpe-style ratio, max/current drawdown and return correlations.
- **Forecast tracking** (`src/tracker.py`): every analysis logs its forecasts for all models plus a "no change" reference. Once the target date passes, the real close is recorded and accuracy (error, hit rate, vs naive) is summarised. A logged forecast is never overwritten. Small samples are noisy.
- **Market and sector inputs** (`src/market_context.py`): India (Nifty, Sensex, India VIX, USD/INR, crude) or US (S&P 500, VIX, rates, crude) plus the stock's sector index/ETF. Assets that close after the stock are lagged one session, so no look-ahead. Untick "Market inputs" to turn off.
- **Tuned models** (tick "Tune models"): random search per model scored by expanding-window validation *inside the training block only* (with a 21-session embargo). Slower (1-2+ minutes). The test block is untouched.
- **GRU, Transformer and ensemble**: the Transformer needs PyTorch. The ensemble blends the best 3 models with weights from validation error only.
- **Volatility forecast** (`src/volatility.py`): 5/10/21-session annualised volatility versus "stays the same" and EWMA baselines; the month range uses a normal approximation (real tails are fatter).

## Known limitations
- Features are daily prices/volume plus optional market inputs (no news or fundamentals); daily stock returns are close to unpredictable and models often do not beat the naive baseline - the app says so.
- Tuning is off by default (slower); with few rows it can overfit validation, so check the test metrics.
- The Transformer, XGBoost and live Yahoo sector/index symbols could not be exercised in the build sandbox (no PyTorch/xgboost/Yahoo access); they are optional and skipped with a note if unavailable. Tests use synthetic data, never shown as real.
- Intervals are empirical residual ranges; real moves (earnings, crashes) can fall outside them.
- Month paths come from direct multi-horizon models (not recursive simulation), so they are smooth estimates of the
  expected path, not simulated price paths.
- Yahoo Finance is an unofficial source; it can rate-limit, change, or have gaps. Indian small caps may have sparse data.
- Without the `holidays` package forecast dates assume Mon-Fri sessions.
- Old Tkinter UI and notebooks were moved to `legacy/` (not maintained).
