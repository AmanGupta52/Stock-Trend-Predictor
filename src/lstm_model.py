"""LSTM sequence regressor with two interchangeable backends.

* ``torch``  - nn.LSTM (used automatically when PyTorch is installed).
* ``numpy``  - a from-scratch LSTM (forward + BPTT + Adam), used when PyTorch is unavailable.

Both map a window of the last ``seq_len`` scaled feature rows to the vector of cumulative log returns
for h = 1..H, trained with MSE on standardised targets, early stopping on a chronological validation block.
"""
from __future__ import annotations

import logging
from typing import Optional, Tuple

import numpy as np

log = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except Exception:  # pragma: no cover
    torch = None
    HAS_TORCH = False


def make_windows(F: np.ndarray, idx: np.ndarray, seq_len: int) -> np.ndarray:
    """Windows F[i-seq_len+1 .. i] for each i in idx (inputs only look backwards)."""
    idx = np.asarray(idx)
    if len(idx) and idx.min() < seq_len - 1:
        raise ValueError("index too early for the requested window")
    return np.stack([F[i - seq_len + 1:i + 1] for i in idx]).astype(np.float64) if len(idx) else \
        np.zeros((0, seq_len, F.shape[1]))


def _sig(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


class NumpyLSTM:
    """Single-layer LSTM + linear head, trained with Adam. Gate order: i, f, g, o."""

    def __init__(self, n_in: int, hidden: int, n_out: int, seed: int = 0):
        rng = np.random.default_rng(seed)
        H = hidden
        k = 1.0 / np.sqrt(H)
        self.p = {
            "Wx": rng.uniform(-k, k, (n_in, 4 * H)),
            "Wh": rng.uniform(-k, k, (H, 4 * H)),
            "b": np.zeros(4 * H),
            "Wy": rng.uniform(-k, k, (H, n_out)),
            "by": np.zeros(n_out),
        }
        self.p["b"][H:2 * H] = 1.0  # forget-gate bias
        self.H = H

    def forward(self, X, keep=False):
        B, L, _ = X.shape
        H, p = self.H, self.p
        h = np.zeros((B, H)); c = np.zeros((B, H))
        cache = []
        for t in range(L):
            z = X[:, t] @ p["Wx"] + h @ p["Wh"] + p["b"]
            i, f, g, o = _sig(z[:, :H]), _sig(z[:, H:2 * H]), np.tanh(z[:, 2 * H:3 * H]), _sig(z[:, 3 * H:])
            c_new = f * c + i * g
            tc = np.tanh(c_new)
            h_new = o * tc
            if keep:
                cache.append((X[:, t], h, c, i, f, g, o, tc))
            h, c = h_new, c_new
        y = h @ p["Wy"] + p["by"]
        return (y, (cache, h)) if keep else y

    def loss_and_grads(self, X, Y):
        y, (cache, h_last) = self.forward(X, keep=True)
        B, O = Y.shape
        diff = y - Y
        loss = float((diff ** 2).mean())
        dy = 2.0 * diff / (B * O)
        g = {k: np.zeros_like(v) for k, v in self.p.items()}
        g["Wy"] = h_last.T @ dy
        g["by"] = dy.sum(0)
        dh = dy @ self.p["Wy"].T
        dc = np.zeros_like(dh)
        for xt, h_prev, c_prev, i, f, gg, o, tc in reversed(cache):
            do = dh * tc * o * (1 - o)
            dct = dc + dh * o * (1 - tc ** 2)
            di = dct * gg * i * (1 - i)
            df = dct * c_prev * f * (1 - f)
            dg = dct * i * (1 - gg ** 2)
            dz = np.concatenate([di, df, dg, do], axis=1)
            g["Wx"] += xt.T @ dz
            g["Wh"] += h_prev.T @ dz
            g["b"] += dz.sum(0)
            dh = dz @ self.p["Wh"].T
            dc = dct * f
        return loss, g


def _fit_numpy(X, Y, Xv, Yv, hidden, epochs, patience, batch, lr, wd, seed, fixed_epochs):
    rng = np.random.default_rng(seed)
    net = NumpyLSTM(X.shape[2], hidden, Y.shape[1], seed)
    m = {k: np.zeros_like(v) for k, v in net.p.items()}
    v2 = {k: np.zeros_like(v) for k, v in net.p.items()}
    step, best, best_ep, best_p, bad = 0, np.inf, 0, None, 0
    n = len(X)
    total = fixed_epochs or epochs
    for ep in range(1, total + 1):
        order = rng.permutation(n)
        for s in range(0, n, batch):
            b = order[s:s + batch]
            _, g = net.loss_and_grads(X[b], Y[b])
            # gradient clipping (global norm 1.0)
            norm = np.sqrt(sum((gg ** 2).sum() for gg in g.values()))
            scale = min(1.0, 1.0 / (norm + 1e-12))
            step += 1
            for k in net.p:
                grad = g[k] * scale + (wd * net.p[k] if k.startswith("W") else 0)
                m[k] = 0.9 * m[k] + 0.1 * grad
                v2[k] = 0.999 * v2[k] + 0.001 * grad ** 2
                net.p[k] -= lr * (m[k] / (1 - 0.9 ** step)) / (np.sqrt(v2[k] / (1 - 0.999 ** step)) + 1e-8)
        if Xv is not None and not fixed_epochs:
            vl = float(((net.forward(Xv) - Yv) ** 2).mean())
            if vl < best - 1e-6:
                best, best_ep, bad = vl, ep, 0
                best_p = {k: a.copy() for k, a in net.p.items()}
            else:
                bad += 1
                if bad >= patience:
                    break
    if best_p is not None:
        net.p = best_p
    return net, (best_ep if best_ep else total)


class _TorchWrap:
    def __init__(self, net):
        self.net = net

    def forward(self, X):
        self.net.eval()
        with torch.no_grad():
            return self.net(torch.as_tensor(X, dtype=torch.float32)).numpy().astype(np.float64)


def _fit_torch(X, Y, Xv, Yv, hidden, epochs, patience, batch, lr, wd, seed, fixed_epochs):
    torch.manual_seed(seed)

    class Net(nn.Module):
        def __init__(self, d, h, o):
            super().__init__()
            self.lstm = nn.LSTM(d, h, batch_first=True)
            self.head = nn.Linear(h, o)

        def forward(self, x):
            out, _ = self.lstm(x)
            return self.head(out[:, -1])

    net = Net(X.shape[2], hidden, Y.shape[1])
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=wd)
    Xt, Yt = torch.as_tensor(X, dtype=torch.float32), torch.as_tensor(Y, dtype=torch.float32)
    Xvt = torch.as_tensor(Xv, dtype=torch.float32) if Xv is not None else None
    Yvt = torch.as_tensor(Yv, dtype=torch.float32) if Yv is not None else None
    gen = torch.Generator().manual_seed(seed)
    best, best_ep, best_state, bad = np.inf, 0, None, 0
    total = fixed_epochs or epochs
    for ep in range(1, total + 1):
        net.train()
        perm = torch.randperm(len(Xt), generator=gen)
        for s in range(0, len(Xt), batch):
            b = perm[s:s + batch]
            opt.zero_grad()
            loss = ((net(Xt[b]) - Yt[b]) ** 2).mean()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
        if Xvt is not None and not fixed_epochs:
            net.eval()
            with torch.no_grad():
                vl = float(((net(Xvt) - Yvt) ** 2).mean())
            if vl < best - 1e-6:
                best, best_ep, bad = vl, ep, 0
                best_state = {k: t.clone() for k, t in net.state_dict().items()}
            else:
                bad += 1
                if bad >= patience:
                    break
    if best_state is not None:
        net.load_state_dict(best_state)
    return _TorchWrap(net), (best_ep if best_ep else total)


class LSTMRegressor:
    """Ensemble of ``n_seeds`` LSTMs over feature windows. ``F`` is the full scaled feature matrix."""

    family = "DL"

    def __init__(self, seq_len=20, hidden=32, epochs=60, patience=8, batch_size=32, lr=3e-3,
                 weight_decay=1e-4, n_seeds=3, backend="auto", random_state=42):
        self.seq_len, self.hidden, self.epochs, self.patience = seq_len, hidden, epochs, patience
        self.batch_size, self.lr, self.wd, self.n_seeds = batch_size, lr, weight_decay, n_seeds
        self.random_state = random_state
        if backend == "auto":
            backend = "torch" if HAS_TORCH else "numpy"
        if backend == "torch" and not HAS_TORCH:
            raise RuntimeError("backend 'torch' requested but PyTorch is not installed")
        self.backend = backend
        self.nets = []
        self.best_epoch: Optional[int] = None
        self.fixed_epochs: Optional[int] = None

    def fit(self, F, idx_fit, Y_fit, idx_val=None, Y_val=None):
        Y_fit = np.asarray(Y_fit, dtype=np.float64)
        self.y_mu, self.y_sd = Y_fit.mean(0), Y_fit.std(0) + 1e-9
        X = make_windows(F, idx_fit, self.seq_len)
        Y = (Y_fit - self.y_mu) / self.y_sd
        Xv = Yv = None
        if idx_val is not None and self.fixed_epochs is None:
            Xv = make_windows(F, idx_val, self.seq_len)
            Yv = (np.asarray(Y_val) - self.y_mu) / self.y_sd
        fit = _fit_torch if self.backend == "torch" else _fit_numpy
        self.nets, eps = [], []
        for s in range(self.n_seeds):
            net, ep = fit(X, Y, Xv, Yv, self.hidden, self.epochs, self.patience, self.batch_size, self.lr,
                          self.wd, self.random_state + s, self.fixed_epochs)
            self.nets.append(net)
            eps.append(ep)
        self.best_epoch = int(round(float(np.mean(eps))))
        return self

    def predict(self, F, idx):
        X = make_windows(F, idx, self.seq_len)
        preds = np.mean([n.forward(X) for n in self.nets], axis=0)
        return preds * self.y_sd + self.y_mu
