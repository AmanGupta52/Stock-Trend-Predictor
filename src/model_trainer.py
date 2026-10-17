"""Model zoo, hyper-parameter search and chronological training stages.

Stages (all splits are chronological, with an embargo of ``horizon`` rows between blocks so that
overlapping multi-day targets never leak across a boundary):

* A - fit on TRAIN, predict VALIDATION          -> model selection, residual-based uncertainty
* B - fit on everything before the embargo ahead of TEST, predict TEST -> unbiased held-out evaluation
* C - fit on ALL labelled rows, predict the latest session -> the live forecast

Every model predicts the vector of cumulative log returns for h = 1..H in one go.
Optional tuning searches hyper-parameters with expanding-window cross-validation *inside the training block only*.
"""
from __future__ import annotations

import itertools
import logging
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import numpy as np

from .data_processor import DataError
from .lstm_model import HAS_TORCH, LSTMRegressor

log = logging.getLogger(__name__)

try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except Exception:  # pragma: no cover
    HAS_XGB = False

from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.multioutput import MultiOutputRegressor

FAMILY = {"ridge": "ML", "random_forest": "ML", "gradient_boosting": "ML", "xgboost": "ML",
          "lstm": "DL", "gru": "DL", "transformer": "DL", "ensemble": "Ensemble",
          "naive": "Baseline", "drift": "Baseline"}
LABEL = {"ridge": "Ridge Regression", "random_forest": "Random Forest", "gradient_boosting": "Gradient Boosting",
         "xgboost": "XGBoost", "lstm": "LSTM (deep learning)", "gru": "GRU (deep learning)",
         "transformer": "Transformer (deep learning)", "ensemble": "Ensemble (blend of best models)",
         "naive": "Naive (no change)", "drift": "Historical drift"}
DEEP = ("lstm", "gru", "transformer")

DEFAULTS = {
    "ridge": {"alpha": 100.0},
    "random_forest": {"max_depth": 6, "min_samples_leaf": 10, "max_features": 0.5},
    "gradient_boosting": {"learning_rate": 0.05, "max_depth": 3, "max_iter": 150, "min_samples_leaf": 20},
    "xgboost": {"learning_rate": 0.03, "max_depth": 3, "n_estimators": 200, "min_child_weight": 10},
    "lstm": {"hidden": 32, "lr": 0.003},
    "gru": {"hidden": 32, "lr": 0.003},
    "transformer": {"hidden": 32, "lr": 0.001},
}
SEARCH_SPACE = {
    "ridge": {"alpha": [1.0, 10.0, 100.0, 1000.0, 10000.0]},
    "random_forest": {"max_depth": [3, 5, 8], "min_samples_leaf": [5, 10, 20], "max_features": [0.3, 0.5, 0.8]},
    "gradient_boosting": {"learning_rate": [0.03, 0.05, 0.1], "max_depth": [2, 3, 4], "max_iter": [100, 200],
                          "min_samples_leaf": [10, 20, 40]},
    "xgboost": {"learning_rate": [0.02, 0.03, 0.06], "max_depth": [2, 3, 4], "n_estimators": [150, 300],
                "min_child_weight": [5, 10, 20]},
    "lstm": {"hidden": [16, 32, 64], "lr": [0.001, 0.003, 0.006]},
    "gru": {"hidden": [16, 32, 64], "lr": [0.001, 0.003, 0.006]},
    "transformer": {"hidden": [16, 32], "lr": [0.001, 0.003]},
}
N_CANDIDATES = {"ridge": 5, "random_forest": 6, "gradient_boosting": 6, "xgboost": 6, "lstm": 3, "gru": 3,
                "transformer": 2}


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


def build_model(name: str, cfg: dict, fixed_epochs: Optional[int] = None, params: Optional[dict] = None,
                fast: bool = False):
    """Create an unfitted model. ``params`` override the defaults (used by tuning); ``fast`` = cheap deep nets."""
    seed = cfg.get("random_state", 42)
    p = {**DEFAULTS.get(name, {}), **(params or {})}
    if name == "ridge":
        return TabularModel(Ridge(alpha=p["alpha"]))
    if name == "random_forest":
        return TabularModel(RandomForestRegressor(n_estimators=300, max_depth=p["max_depth"],
                                                  min_samples_leaf=p["min_samples_leaf"],
                                                  max_features=p["max_features"], n_jobs=-1, random_state=seed))
    if name == "gradient_boosting":
        return TabularModel(MultiOutputRegressor(HistGradientBoostingRegressor(
            max_iter=p["max_iter"], learning_rate=p["learning_rate"], max_depth=p["max_depth"],
            min_samples_leaf=p["min_samples_leaf"], l2_regularization=1.0, random_state=seed)))
    if name == "xgboost":
        if not HAS_XGB:
            raise RuntimeError("xgboost is not installed (pip install xgboost)")
        return TabularModel(XGBRegressor(n_estimators=p["n_estimators"], learning_rate=p["learning_rate"],
                                         max_depth=p["max_depth"], subsample=0.8, colsample_bytree=0.7,
                                         reg_lambda=5.0, min_child_weight=p["min_child_weight"],
                                         tree_method="hist", n_jobs=-1, random_state=seed))
    if name in DEEP:
        c = cfg.get("lstm", {})
        m = LSTMRegressor(seq_len=cfg.get("seq_len", 20), hidden=p["hidden"], epochs=c.get("epochs", 60),
                          patience=c.get("patience", 8), batch_size=c.get("batch_size", 32), lr=p["lr"],
                          weight_decay=c.get("weight_decay", 1e-4), n_seeds=1 if fast else c.get("n_seeds", 3),
                          backend=c.get("backend", "auto"), random_state=seed, kind=name)
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
    backends: Dict[str, str] = field(default_factory=dict)
    epochs: Dict[str, int] = field(default_factory=dict)
    feature_importance: List[dict] = field(default_factory=list)
    tuning: Dict[str, dict] = field(default_factory=dict)
    ensemble_weights: Dict[str, float] = field(default_factory=dict)

    @property
    def lstm_backend(self):            # kept for backward compatibility
        return self.backends.get("lstm")

    @property
    def lstm_epochs(self):
        return self.epochs.get("lstm")


# --------------------------------------------------------------------------- tuning
def _rel_mae(pred: np.ndarray, actual: np.ndarray, horizons) -> float:
    return float(np.mean([np.mean(np.abs(pred[:, h - 1] - actual[:, h - 1])) /
                          max(np.mean(np.abs(actual[:, h - 1])), 1e-12) for h in horizons]))


def cv_folds(train_idx: np.ndarray, horizon: int):
    """Expanding-window folds inside the training block, each with an embargo of ``horizon`` rows."""
    m = len(train_idx)
    folds = []
    for frac in (0.55, 0.70, 0.85):
        c = int(m * frac)
        fit = train_idx[:c]
        val = train_idx[c + horizon:min(m, c + horizon + int(0.15 * m))]
        if len(fit) >= 80 and len(val) >= 10:
            folds.append((fit, val))
    return folds


def candidate_params(name: str, seed: int) -> List[dict]:
    """The default configuration plus a reproducible random sample of the search space."""
    space = SEARCH_SPACE[name]
    keys = list(space)
    grid = [dict(zip(keys, v)) for v in itertools.product(*[space[k] for k in keys])]
    rng = np.random.default_rng(seed)
    rng.shuffle(grid)
    out = [dict(DEFAULTS[name])]
    for g in grid:
        if g != DEFAULTS[name] and len(out) < N_CANDIDATES[name]:
            out.append(g)
    return out


def tune_model(name: str, F_raw: np.ndarray, Y: np.ndarray, plan: SplitPlan, cfg: dict, horizons) -> dict:
    """Pick hyper-parameters by time-aware CV on the TRAIN block only (validation/test rows are never touched)."""
    folds = cv_folds(plan.train, plan.horizon)
    if not folds:
        return {"params": dict(DEFAULTS[name]), "cv_score": None, "default_cv_score": None, "tried": 0}
    results = []
    for params in candidate_params(name, cfg.get("random_state", 42)):
        scores = []
        for fit_idx, val_idx in folds:
            F = Scaler().fit(F_raw[fit_idx]).transform(F_raw)
            m = build_model(name, cfg, fixed_epochs=20 if name in DEEP else None, params=params, fast=True)
            m.fit(F, fit_idx, Y[fit_idx])
            scores.append(_rel_mae(m.predict(F, val_idx), Y[val_idx], horizons))
        results.append((float(np.mean(scores)), params))
    best = min(results, key=lambda r: r[0])
    return {"params": best[1], "cv_score": best[0], "default_cv_score": results[0][0], "tried": len(results),
            "changed": best[1] != DEFAULTS[name]}


# --------------------------------------------------------------------------- training
def blend_weights(res: TrainingResult, Y: np.ndarray, horizons, top_k: int = 3) -> Dict[str, float]:
    """Inverse-squared-error weights over the top-k base models, using VALIDATION predictions only."""
    yv = Y[res.plan.val]
    base = [m for m in res.models if FAMILY[m] in ("ML", "DL")]
    scored = sorted((_rel_mae(res.pred_val[m], yv, horizons), m) for m in base)[:top_k]
    raw = {m: 1.0 / max(s, 1e-6) ** 2 for s, m in scored}
    tot = sum(raw.values())
    return {m: w / tot for m, w in raw.items()}


def train_all(F_raw: np.ndarray, Y: np.ndarray, feature_names: List[str], plan: SplitPlan, cfg: dict,
              progress: Optional[Callable[[float, str], None]] = None, tune: bool = False) -> TrainingResult:
    """Run (optional tuning and) stages A, B and C for every configured model plus the baselines and ensemble."""
    progress = progress or (lambda p, m: None)
    H = plan.horizon
    horizons = [h for h in cfg.get("eval_horizons", [1, 5, 10, 21]) if h <= H]
    names = [m for m in cfg.get("models", []) if m in FAMILY and FAMILY[m] in ("ML", "DL")]
    res = TrainingResult(plan=plan, models=[])
    order = ["naive", "drift"] + names
    lo = 0.15
    params: Dict[str, dict] = {}

    if tune:
        tunable = [n for n in names if n in SEARCH_SPACE and (n != "xgboost" or HAS_XGB)
                   and (n != "transformer" or HAS_TORCH)]
        for i, name in enumerate(tunable):
            progress(lo + 0.25 * i / max(1, len(tunable)), f"Tuning {LABEL[name]} (time-aware cross-validation)")
            try:
                t = tune_model(name, F_raw, Y, plan, cfg, horizons)
                res.tuning[name] = t
                params[name] = t["params"]
            except Exception as exc:
                log.warning("tuning %s failed: %s", name, exc)
                res.notes.append(f"Tuning {LABEL[name]} failed ({type(exc).__name__}); defaults were used.")
        lo = 0.40
    total_steps = len(order) * 3
    step = 0
    fixed: Dict[str, int] = {}

    def run(name, stage):
        nonlocal step
        step += 1
        progress(lo + (0.85 - lo) * step / total_steps, f"Training {LABEL[name]} - stage {stage}")

    for name in order:
        try:
            run(name, "A (validation)")
            sc = Scaler().fit(F_raw[plan.train])
            F = sc.transform(F_raw)
            model = build_model(name, cfg, params=params.get(name))
            if name in DEEP:
                tr = plan.train
                cut = int(0.85 * len(tr))
                inner_fit, inner_val = tr[:cut], tr[cut + H:]          # embargoed inner early-stopping block
                model.fit(F, inner_fit, Y[inner_fit], inner_val, Y[inner_val])
                fixed[name] = model.best_epoch
                res.backends[name], res.epochs[name] = model.backend, model.best_epoch
            else:
                model.fit(F, plan.train, Y[plan.train])
            pv = model.predict(F, plan.val)
            run(name, "B (held-out test)")
            sc = Scaler().fit(F_raw[plan.fit_tv])
            F = sc.transform(F_raw)
            model = build_model(name, cfg, fixed.get(name), params.get(name))
            model.fit(F, plan.fit_tv, Y[plan.fit_tv])
            pt = model.predict(F, plan.test)
            run(name, "C (final fit)")
            sc = Scaler().fit(F_raw[plan.all])
            F = sc.transform(F_raw)
            model = build_model(name, cfg, fixed.get(name), params.get(name))
            model.fit(F, plan.all, Y[plan.all])
            pf = model.predict(F, np.array([len(F_raw) - 1]))[0]
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

    base = [m for m in res.models if FAMILY[m] in ("ML", "DL")]
    if len(base) >= 2:
        w = blend_weights(res, Y, horizons)
        res.ensemble_weights = w
        res.pred_val["ensemble"] = sum(w[m] * res.pred_val[m] for m in w)
        res.pred_test["ensemble"] = sum(w[m] * res.pred_test[m] for m in w)
        res.pred_final["ensemble"] = sum(w[m] * res.pred_final[m] for m in w)
        res.models.append("ensemble")
    for m, why in res.failed.items():
        if m == "transformer" and not HAS_TORCH:
            res.notes.append("Transformer skipped: it needs PyTorch (pip install torch).")
        else:
            res.notes.append(f"{LABEL[m]} was skipped: {why}")
    return res
