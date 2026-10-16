"""Shared helpers. NOTE: `synthetic_prices` is random data used ONLY to exercise code paths in tests;
it is never used by the application."""
import numpy as np
import pandas as pd


def synthetic_prices(n=1300, seed=1, start="2021-01-01"):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start, periods=n)
    vol = 0.012 + 0.008 * np.abs(np.sin(np.arange(n) / 90))
    c = 100 * np.exp(np.cumsum(0.0004 + vol * rng.standard_normal(n)))
    o = c * np.exp(rng.normal(0, 0.004, n))
    h = np.maximum(o, c) * (1 + np.abs(rng.normal(0, 0.004, n)))
    l = np.minimum(o, c) * (1 - np.abs(rng.normal(0, 0.004, n)))
    return pd.DataFrame({"Open": o, "High": h, "Low": l, "Close": c, "Adj Close": c,
                         "Volume": rng.integers(1_000_000, 5_000_000, n).astype(float)}, index=idx)
