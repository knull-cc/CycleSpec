"""Usage: python scripts/CycleSpec/proxy_band_diag.py ETTh1=./dataset/ETT-small/ETTh1.csv ECL=<electricity.txt.gz>

Band-wise error analysis with linear stand-in forecasters (numpy only).

Forecasters (fit on the train split, RevIN per window/variable, shared over variables):
  phase-linear : per phase p, the 30 past values at phase p -> the H/24 future values at phase p
                 (the linear core of PhaseFormer, no cross-phase mixing)
  full-linear  : the whole 720-step window -> 96 steps (RLinear-like)
Errors on val and test are split along the horizon into DC / low / mid / high bands
(log-frequency thirds of bins 1..H/2; orthogonal, band MSEs add up to the MSE).
Q1: share of the test MSE per band, and band MSE / band power of the truth.
Q2: probes fit on VAL errors, scored on TEST errors (held-out R^2):
    linear probe    : ridge from the input window (720, RevIN)
    nonlinear probe : ridge from [input, 1024 random ReLU features of the input]
"""
import gzip
import sys

import numpy as np
import pandas as pd

L, H, P = 720, 96, 24
rng = np.random.default_rng(0)


def load(name, path):
    if path.endswith(".csv"):
        d = pd.read_csv(path).iloc[:, 1:].values.astype(np.float64)
        f = 4 if name.startswith("ETTm") else 1
        b1 = [0, 8640 * f - L, (8640 + 2880) * f - L]
        b2 = [8640 * f, (8640 + 2880) * f, (8640 + 5760) * f]
    else:
        d = np.loadtxt(gzip.open(path), delimiter=",")
        n = len(d)
        ntr, nte = int(n * 0.7), int(n * 0.2)
        b1 = [0, ntr - L, n - nte - L]
        b2 = [ntr, n - nte, n]
    tr = d[b1[0]:b2[0]]
    return (d - tr.mean(0)) / (tr.std(0) + 1e-8), b1, b2


def windows(d, b1, b2, split, cols, max_w):
    seg = d[b1[split]:b2[split]][:, cols]
    n = len(seg) - L - H + 1
    stride = max(1, int(np.ceil(n / max_w)))
    idx = np.arange(0, n, stride)
    X = np.stack([seg[i:i + L] for i in idx]).transpose(0, 2, 1)       # (w, c, L)
    Y = np.stack([seg[i + L:i + L + H] for i in idx]).transpose(0, 2, 1)
    mu, sd = X.mean(-1, keepdims=True), np.sqrt(X.var(-1, keepdims=True) + 1e-5)
    return ((X - mu) / sd).astype(np.float32), ((Y - mu) / sd).astype(np.float32), sd


def ridge(X, Y, lam):
    A = np.hstack([X, np.ones((len(X), 1), X.dtype)]).astype(np.float64)
    return np.linalg.solve(A.T @ A + lam * np.eye(A.shape[1]), A.T @ Y)


def apply(W, X):
    return np.hstack([X, np.ones((len(X), 1), X.dtype)]) @ W


LAMS = (1e-1, 1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6)


def pick_lam(X, Y, lams=LAMS):
    cut = int(len(X) * 0.8)
    return min(lams, key=lambda l: np.mean((apply(ridge(X[:cut], Y[:cut], l), X[cut:]) - Y[cut:]) ** 2))


def to_phase(X, n_out):  # (w, c, T*P) -> rows (w*c*P, T)
    w, c, T = X.shape
    return X.reshape(w, c, T // P, P).transpose(0, 1, 3, 2).reshape(-1, T // P)


def from_phase(Yp, w, c):  # rows (w*c*P, H/P) -> (w, c, H)
    return Yp.reshape(w, c, P, H // P).transpose(0, 1, 3, 2).reshape(w, c, H)


class PhaseLinear:
    def fit(self, X, Y, Xv, Yv):
        xr, yr = to_phase(X, None), to_phase(Y, None)
        xv, yv = to_phase(Xv, None), to_phase(Yv, None)
        self.W = min((ridge(xr, yr, l) for l in LAMS), key=lambda W: np.mean((apply(W, xv) - yv) ** 2))
        return self

    def predict(self, X):
        w, c, _ = X.shape
        return from_phase(apply(self.W, to_phase(X, None)), w, c)


class FullLinear:
    def fit(self, X, Y, Xv, Yv):
        xr, yr = X.reshape(-1, L), Y.reshape(-1, H)
        xv, yv = Xv.reshape(-1, L), Yv.reshape(-1, H)
        self.W = min((ridge(xr, yr, l) for l in LAMS), key=lambda W: np.mean((apply(W, xv) - yv) ** 2))
        return self

    def predict(self, X):
        w, c, _ = X.shape
        return apply(self.W, X.reshape(-1, L)).reshape(w, c, H)


def masks():
    k = np.arange(H // 2 + 1)
    e = (H // 2) ** (np.array([1, 2]) / 3.0)
    return {"DC": k == 0, "low": (k >= 1) & (k < e[0]), "mid": (k >= e[0]) & (k < e[1]), "high": k >= e[1]}


M = masks()


def bands(x):
    F = np.fft.rfft(x, axis=-1)
    return {b: np.fft.irfft(F * m, n=H, axis=-1) for b, m in M.items()}


RF = None


def features(X, nonlinear):
    x = X.reshape(-1, L)
    if not nonlinear:
        return x
    global RF
    if RF is None:
        RF = (rng.standard_normal((L, 1024)) / np.sqrt(L)).astype(np.float32)
    return np.hstack([x, np.maximum(x @ RF, 0)])


def r2(Ftr, Ttr, Fte, Tte):
    W = ridge(Ftr, Ttr, pick_lam(Ftr, Ttr))
    return 1 - np.sum((apply(W, Fte) - Tte) ** 2) / np.sum((Tte - Ttr.mean(0)) ** 2)


def run(name, path, max_c=32, max_w=None):
    max_w = max_w or 3000
    global RF
    RF = None
    d, b1, b2 = load(name, path)
    cols = np.arange(d.shape[1]) if d.shape[1] <= max_c else np.sort(rng.choice(d.shape[1], max_c, replace=False))
    if len(cols) > 8:
        max_w = 1200
    Xtr, Ytr, _ = windows(d, b1, b2, 0, cols, max_w)
    Xva, Yva, _ = windows(d, b1, b2, 1, cols, max_w)
    Xte, Yte, sdte = windows(d, b1, b2, 2, cols, max_w)
    out = []
    for fname, F in (("phase-linear", PhaseLinear()), ("full-linear", FullLinear())):
        F.fit(Xtr, Ytr, Xva, Yva)
        Rva, Rte = F.predict(Xva) - Yva, F.predict(Xte) - Yte
        # MSE in the global standardised scale (comparable to papers): undo RevIN on the test errors
        mse_paper = float(np.mean((Rte * sdte) ** 2))
        rb_te, yb_te, rb_va = bands(Rte), bands(Yte), bands(Rva)
        tot = sum(np.mean(rb_te[b] ** 2) for b in M)
        res = dict(model=fname, mse=mse_paper)
        for b in M:
            res[f"share_{b}"] = np.mean(rb_te[b] ** 2) / tot
            res[f"ratio_{b}"] = np.mean(rb_te[b] ** 2) / np.mean(yb_te[b] ** 2)
            for nl in (False, True):
                res[f"{'nl' if nl else 'lin'}_{b}"] = r2(features(Xva, nl), rb_va[b].reshape(-1, H).astype(np.float32),
                                                         features(Xte, nl), rb_te[b].reshape(-1, H).astype(np.float32))
        out.append(res)
    return out, len(cols)


if __name__ == "__main__":
    for spec in sys.argv[1:]:
        name, path = spec.split("=", 1)
        res, nc = run(name, path)
        print(f"\n### {name}  (L=720 -> H=96, {nc} variables)")
        print(f"{'model':<13}{'MSE':>7} | " + " | ".join(f"{b:^27}" for b in M))
        print(f"{'':<13}{'':>7} | " + " | ".join(f"{'share ratio  R2lin R2nonlin':^27}" for b in M))
        for r in res:
            print(f"{r['model']:<13}{r['mse']:>7.3f} | " + " | ".join(
                f"{r[f'share_{b}'] * 100:5.1f}% {r[f'ratio_{b}']:5.2f} {r[f'lin_{b}']:+6.3f} {r[f'nl_{b}']:+6.3f}" for b in M))
        sys.stdout.flush()
