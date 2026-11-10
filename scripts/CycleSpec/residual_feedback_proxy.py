"""Proxy test: can past, already-revealed forecast errors correct the current forecast?  (no training)

    python scripts/CycleSpec/residual_feedback_proxy.py --results_dir ./results
    python scripts/CycleSpec/residual_feedback_proxy.py --results_dir ./results --match etth2 --seed 2021

Reads pred.npy / true.npy (N x H x C, test windows with stride 1, in the scale the metrics use) of
every finished run. Window i is issued at time t_i = i + L and predicts t_i .. t_i+H-1, so at t_i
the truth up to t_i - 1 is known (it is inside window i's input) and the error of an earlier
window j = i - k is revealed for its first k steps. Only those revealed errors are used:

    f1  r_{i-1}[0]                        lead-1 error at the last observed time step
    f2  mean_{k=1..24} r_{i-k}[0]         mean lead-1 error over the last 24 steps
    f3  mean_{k=1..m} r_{i-k}[k-1]        errors at the last observed step from forecasts made
                                          1..m steps earlier (m = min(H, 96)): the "long-lead" bias
    f4  mean_h r_{i-H}[h]                 mean error of the last fully revealed window

Corrected forecast: pred_i[h, c] - sum_f beta_{f,h} * f(i, c), ridge fit per horizon step h,
shared by all channels. beta is fit on the first half of the test windows and scored on the
second half (a gap of H windows in between, so no fitted target lies after a scored origin).
In a real plug-in beta would be fit on the validation set; here the test halves stand in for it.

Rows:
    base          the run as is, on the scored half
    recent        f1 f2 f3
    recent+full   f1 f2 f3 f4  (needs N > ~3H windows; n/a otherwise)
    oracle-DC     subtract the TRUE mean error of window i (upper bound of a pure level fix,
                  not achievable; tells how much of the error is level)
"""
import argparse
import glob
import json
import os

import numpy as np


def feats(r, H):
    N, _, C = r.shape
    m = min(H, 96)
    i0 = max(24, m)
    lead1 = r[:, 0, :]                                  # (N, C)
    cs = np.cumsum(np.vstack([np.zeros((1, C)), lead1]), axis=0)
    F, idx = [], np.arange(i0, N)
    f1 = lead1[idx - 1]
    f2 = (cs[idx] - cs[idx - 24]) / 24.0
    f3 = np.zeros((len(idx), C))
    for k in range(1, m + 1):
        f3 += r[idx - k, k - 1, :]
    f3 /= m
    F = [f1, f2, f3]
    full = None
    if N > H + 10:
        wmean = r.mean(axis=1)                          # (N, C)
        full = np.full((len(idx), C), np.nan)
        ok = idx - H >= 0
        full[ok] = wmean[idx[ok] - H]
    return idx, np.stack(F, axis=-1), full              # (n,), (n, C, 3), (n, C)


def ridge_per_step(X, Y, lam=1e-2):
    """X (n, C, p), Y (n, H, C) -> beta (H, p); one regression per horizon step, channels pooled."""
    n, C, p = X.shape
    A = X.reshape(n * C, p)
    G = A.T @ A
    G += lam * np.trace(G) / p * np.eye(p)
    B = np.linalg.solve(G, A.T @ Y.transpose(0, 2, 1).reshape(n * C, -1))   # (p, H)
    return B.T


def score(pred, true):
    e = pred - true
    return float((e ** 2).mean()), float(np.abs(e).mean())


def analyse(pred, true):
    N, H, C = pred.shape
    r = pred - true
    idx, X, full = feats(r, H)
    n = len(idx)
    half = n // 2
    cal = np.arange(0, half)
    ev = np.arange(min(n, half + H), n)
    out = {"N": N, "n_eval": len(ev)}
    if len(ev) < 20 or len(cal) < 20:
        return out
    P, T = pred[idx], true[idx]
    out["base"] = score(P[ev], T[ev])

    B = ridge_per_step(X[cal], r[idx][cal])
    corr = np.einsum("ncp,hp->nhc", X[ev], B)
    out["recent"] = score(P[ev] - corr, T[ev])

    if full is not None:
        okc = cal[~np.isnan(full[cal]).any(axis=1)]
        oke = ev[~np.isnan(full[ev]).any(axis=1)]
        if len(okc) >= 20 and len(oke) == len(ev):
            X4 = np.concatenate([X, np.nan_to_num(full)[..., None]], axis=-1)
            B4 = ridge_per_step(X4[okc], r[idx][okc])
            corr4 = np.einsum("ncp,hp->nhc", X4[ev], B4)
            out["recent+full"] = score(P[ev] - corr4, T[ev])

    dc = r[idx][ev].mean(axis=1, keepdims=True)
    out["oracle-DC"] = score(P[ev] - dc, T[ev])
    # how persistent is the level error? corr(true window-mean error, f2) on the scored half
    wm = r[idx][ev].mean(axis=1).ravel()
    out["corr_f2"] = float(np.corrcoef(wm, X[ev][..., 1].ravel())[0, 1])
    out["corr_f3"] = float(np.corrcoef(wm, X[ev][..., 2].ravel())[0, 1])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", default="./results")
    ap.add_argument("--match", default="", help="substring of the run folder, e.g. etth1 or etth2_pl96")
    ap.add_argument("--seed", default="", help="only runs with this seed")
    ap.add_argument("--all_variants", action="store_true", help="also CycleSpec runs (default: PhaseFormer only)")
    a = ap.parse_args()

    runs = []
    for js in sorted(glob.glob(os.path.join(a.results_dir, "*", "summary.json"))):
        d = os.path.dirname(js)
        if a.match and a.match not in os.path.basename(d):
            continue
        if not (os.path.exists(os.path.join(d, "pred.npy")) and os.path.exists(os.path.join(d, "true.npy"))):
            continue
        s = json.load(open(js))
        v = s.get("variant", "?")
        if not a.all_variants and v != "PhaseFormer":
            continue
        if a.seed and f"_s{a.seed}_" not in os.path.basename(d):
            continue
        mid = s.get("model_id", os.path.basename(d))
        runs.append((mid, v, d))
    if not runs:
        print("no finished runs found under", a.results_dir)
        return

    rows = ["recent", "recent+full", "oracle-DC"]
    print(f"{'run':<26}{'variant':<24}{'n_eval':>7}{'base MSE/MAE':>17}"
          + "".join(f"{r + ' dMSE/dMAE%':>26}" for r in rows) + f"{'corr f2/f3':>13}")
    for mid, v, d in runs:
        o = analyse(np.load(os.path.join(d, "pred.npy")), np.load(os.path.join(d, "true.npy")))
        line = f"{mid:<26}{v[:23]:<24}{o.get('n_eval', 0):>7}"
        if "base" not in o:
            print(line + "   too few test windows")
            continue
        bm, ba = o["base"]
        line += f"{bm:>9.4f}/{ba:.4f}"
        for r in rows:
            if r in o:
                m, aa = o[r]
                line += f"{(m / bm - 1) * 100:>+16.2f}/{(aa / ba - 1) * 100:+.2f}%  "
            else:
                line += f"{'n/a':>26}"
        line += f"{o['corr_f2']:>7.2f}/{o['corr_f3']:.2f}"
        print(line)
    print("\nnegative % = better than the run itself on the scored half. oracle-DC is NOT achievable "
          "(uses the true error); it bounds what a pure level correction could give.")


if __name__ == "__main__":
    main()
