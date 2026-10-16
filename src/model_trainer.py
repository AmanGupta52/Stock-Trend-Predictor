"""Model zoo and chronological training stages.

Stages (all splits are chronological, with an embargo of ``horizon`` rows between blocks so that
overlapping multi-day targets never leak across a boundary):

* A - fit on TRAIN, predict VALIDATION          -> model selection, residual-based uncertainty
* B - fit on everything before the embargo ahead of TEST, predict TEST -> unbiased held-out evaluation
* C - fit on ALL labelled rows, predict the latest session -> the live forecast

Every model predicts the vector of cumulative log returns for h = 1..H in one go.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import numpy as np

from .data_processor import DataError
from .lstm_model import LSTMRegressor

log = logging.getLogger(__name__)

try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except Exception:  # pragma: no cover
    HAS_XGB = False

from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.multioutput import MultiOutputRegressor

FAMILY = {"ridge": "ML", "random_forest": "ML", "gradient_boosting": "ML", "xgboost": "ML", "lstm": "DL",
          "naive": "Baseline", "drift": "Baseline"}
LABEL = {"ridge": "Ridge Regression", "random_forest": "Random Forest", "gradient_boosting": "Gradient Boosting",
         "xgboost": "XGBoost", "lstm": "LSTM (deep learning)", "naive": "Naive (no change)",
         "drift": "Historical drift"}


class Scaler:
    """Standardiser fit on the fitting rows only (no look-ahead), clipped to +-5 sigma."""

    def fit(self, A):
        self.mu, self.sd = A.mean(0), A.std(0)
        self.sd[self.sd < 1e-12] = 1.0
        return self

    def transform(self, A):
        return np.clip((A - self.mu) / self.sd, -5, 5)


class TabularModel:
    def __init__(self, est):
        self.est = est

    def fit(self, F, idx_fit, Y_fit, idx_val=None, Y_val=None):
        self.est.fit(F[idx_fit], Y_fit)
        return self

    def predict(self, F, idx):
        return np.asarray(self.est.predict(F[idx]), dtype=np.float64).reshape(len(idx), -1)


class ConstantBaseline:
    """naive: predicts zero return; drift: mean cumulative return seen in the fitting rows."""

    def __init__(self, kind):
        self.kind = kind

    def fit(self, F, idx_fit, Y_fit, idx_val=None, Y_val=None):
        self.v = np.zeros(Y_fit.shape[1]) if self.kind == "naive" else Y_fit.mean(0)
        return self

    def predict(self, F, idx):
        return np.tile(self.v, (len(idx), 1))


def build_model(name: str, cfg: dict, fixed_epochs: Optional[int] = None):
    seed = cfg.get("random_state", 42)
    if name == "ridge":
        return TabularModel(Ridge(alpha=100.0))
    if name == "random_forest":
        return TabularModel(RandomForestRegressor(n_estimators=300, max_depth=6, min_samples_leaf=10,
                                                  max_features=0.5, n_jobs=-1, random_state=seed))
    if name == "gradient_boosting":
        return TabularModel(MultiOutputRegressor(HistGradientBoostingRegressor(
            max_iter=150, learning_rate=0.05, max_depth=3, min_samples_leaf=20, l2_regularization=1.0,
            random_state=seed)))
    if name == "xgboost":
        if not HAS_XGB:
            raise RuntimeError("xgboost is not installed (pip install xgboost)")
        return TabularModel(XGBRegressor(n_estimators=200, learning_rate=0.03, max_depth=3, subsample=0.8,
                                         colsample_bytree=0.7, reg_lambda=5.0, min_child_weight=10,
                                         tree_method="hist", n_jobs=-1, random_state=seed))
    if name == "lstm":
        c = cfg.get("lstm", {})
        m = LSTMRegressor(seq_len=cfg.get("seq_len", 20), hidden=c.get("hidden", 32), epochs=c.get("epochs", 60),
                          patience=c.get("patience", 8), batch_size=c.get("batch_size", 32), lr=c.get("lr", 3e-3),
                          weight_decay=c.get("weight_decay", 1e-4), n_seeds=c.get("n_seeds", 3),
                          backend=c.get("backend", "auto"), random_state=seed)
        m.fixed_epochs = fixed_epochs
        return m
    if name in ("naive", "drift"):
        return ConstantBaseline(name)
    raise ValueError(f"unknown model {name}")


@dataclass
class SplitPlan:
    train: np.ndarray
    val: np.ndarray
    test: np.ndarray
    fit_tv: np.ndarray      # everything usable for fitting before the test embargo
    all: np.ndarray
    horizon: int


def make_splits(n_labeled: int, seq_len: int, horizon: int, train_frac: float = 0.70,
                val_frac: float = 0.15) -> SplitPlan:
    idx = np.arange(seq_len - 1, n_labeled)
    m = len(idx)
    a, b = int(train_frac * m), int((train_frac + val_frac) * m)
    plan = SplitPlan(train=idx[:a], val=idx[a + horizon:b], test=idx[b + horizon:], fit_tv=idx[:b], all=idx,
                     horizon=horizon)
    if len(plan.train) < 120 or len(plan.val) < 30 or len(plan.test) < 30:
        raise DataError(
            f"Not enough history to train and validate models ({m} usable samples; train/val/test = "
            f"{len(plan.train)}/{len(plan.val)}/{len(plan.test)}). Choose a longer period (5y or more).",
            "insufficient_history")
    return plan


@dataclass
class TrainingResult:
    plan: SplitPlan
    models: List[str]
    pred_val: Dict[str, np.ndarray] = field(default_factory=dict)      # (n_val, H)
    pred_test: Dict[str, np.ndarray] = field(default_factory=dict)     # (n_test, H)
    pred_final: Dict[str, np.ndarray] = field(default_factory=dict)    # (H,)
    notes: List[str] = field(default_factory=list)
    failed: Dict[str, str] = field(default_factory=dict)
    lstm_backend: Optional[str] = None
    lstm_epochs: Optional[int] = None
    feature_importance: List[dict] = field(default_factory=list)


def train_all(F_raw: np.ndarray, Y: np.ndarray, feature_names: List[str], plan: SplitPlan, cfg: dict,
              progress: Optional[Callable[[float, str], None]] = None) -> TrainingResult:
    """Run stages A, B and C for every configured model plus the two baselines."""
    progress = progress or (lambda p, m: None)
    H = plan.horizon
    names = [m for m in cfg.get("models", []) if m in FAMILY]
    res = TrainingResult(plan=plan, models=[], )
    order = ["naive", "drift"] + names
    total_steps = len(order) * 3
    step = 0
    fixed: Dict[str, int] = {}

    def run(name, stage):
        nonlocal step
        step += 1
        progress(0.15 + 0.7 * step / total_steps, f"Training {LABEL[name]} - stage {stage}")

    for name in order:
        try:
            # ---- stage A: train -> validation
            run(name, "A (validation)")
            sc = Scaler().fit(F_raw[plan.train])
            F = sc.transform(F_raw)
            model = build_model(name, cfg)
            if name == "lstm":
                tr = plan.train
                cut = int(0.85 * len(tr))
                inner_fit, inner_val = tr[:cut], tr[cut + H:]          # embargoed inner early-stopping block
                model.fit(F, inner_fit, Y[inner_fit], inner_val, Y[inner_val])
                fixed[name] = model.best_epoch
                res.lstm_backend, res.lstm_epochs = model.backend, model.best_epoch
            else:
                model.fit(F, plan.train, Y[plan.train])
            pv = model.predict(F, plan.val)
            # ---- stage B: (train+val) -> test
            run(name, "B (held-out test)")
            sc = Scaler().fit(F_raw[plan.fit_tv])
            F = sc.transform(F_raw)
            model = build_model(name, cfg, fixed.get(name))
            model.fit(F, plan.fit_tv, Y[plan.fit_tv])
            pt = model.predict(F, plan.test)
            # ---- stage C: all labelled rows -> latest session
            run(name, "C (final fit)")
            sc = Scaler().fit(F_raw[plan.all])
            F = sc.transform(F_raw)
            model = build_model(name, cfg, fixed.get(name))
            model.fit(F, plan.all, Y[plan.all])
            last = np.array([len(F_raw) - 1])
            pf = model.predict(F, last)[0]
            if name == "random_forest":
                imp = model.est.feature_importances_
                top = np.argsort(imp)[::-1][:10]
                res.feature_importance = [{"feature": feature_names[i], "importance": float(imp[i])} for i in top]
            if not (np.isfinite(pv).all() and np.isfinite(pt).all() and np.isfinite(pf).all()):
                raise FloatingPointError("non-finite predictions")
            res.pred_val[name], res.pred_test[name], res.pred_final[name] = pv, pt, pf
            res.models.append(name)
        except Exception as exc:
            log.warning("model %s failed: %s", name, exc)
            res.failed[name] = f"{type(exc).__name__}: {exc}"
    if "naive" not in res.models:
        raise DataError("Baseline computation failed.", "internal_error")
    if not any(FAMILY[m] == "ML" for m in res.models):
        raise DataError("No machine-learning model could be trained.", "internal_error")
    if "lstm" in names and "lstm" not in res.models:
        res.notes.append(f"LSTM was skipped: {res.failed.get('lstm')}")
    for m, why in res.failed.items():
        if m != "lstm":
            res.notes.append(f"{LABEL[m]} was skipped: {why}")
    return res
