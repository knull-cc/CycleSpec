"""Band-wise error analysis of finished runs (no training).

    python scripts/CycleSpec/error_spectrum.py --results_dir ./results --data_root ./dataset
    python scripts/CycleSpec/error_spectrum.py --results_dir /content/drive/MyDrive/CycleSpec_runs/results \
        --data_root ./dataset --match etth1_pl96

For every run folder with summary.json + pred.npy + true.npy, the forecast error r = pred - true
(test set, shape N x H x C) is split along the horizon axis into orthogonal frequency bands
(rFFT masks, so the band MSEs add up exactly to the total MSE):

    DC      bin 0                         mean level of the error over the horizon
    low     bins 1 .. ~(H/2)^(1/3)        slow drift across the horizon
    mid     the middle third (log scale)
    high    the top third (log scale)
    (bands are equal thirds of log-frequency over bins 1..H/2, fixed before looking at results)

Q1  where does the error come from?   share_b = MSE_b / MSE_total
    how much of the band is missed?   ratio_b = MSE_b / power of the true series in that band
Q2  is the missed part predictable?   R2_b = held-out R^2 of a ridge probe that predicts the band-b
    error from the input window (RevIN-normalised, channel-independent, time-ordered split with a
    gap so train/test windows do not overlap). R2 > 0 means a linear map of the input recovers
    part of what the model missed in that band; it is a lower bound on what is recoverable.
Q3  where does CycleSpec help?        per-band MSE change of each CycleSpec run vs PhaseFormer of
    the same dataset / H / tag / seed.

The input windows are rebuilt from the csv with the same splits and scaler as data_provider;
the rebuilt targets are checked against true.npy before the probe is run.
"""
import argparse
import csv
import glob
import json
import os
import re
from collections import defaultdict

import numpy as np
import pandas as pd

MODEL_ID_RE = re.compile(r"^(?P<ds>[A-Za-z0-9]+)_pl(?P<h>\d+)(?P<rest>(?:_[A-Za-z0-9]+)*)$")


def parse_model_id(mid):
    m = MODEL_ID_RE.match(mid or "")
    if not m:
        return None
    toks = [t for t in m.group("rest").split("_") if t]
    if toks and toks[-1] == "frozen":
        toks.pop()
    if toks and toks[-1].startswith("cs"):
        toks.pop()
    return None if len(toks) > 1 else (m.group("ds"), toks[0] if toks else "")
BANDS = ["DC", "low", "mid", "high"]


# ----------------------------------------------------------------------------- bands
def band_masks(H):
    nb = H // 2 + 1
    k = np.arange(nb)
    top = H // 2
    edges = top ** (np.array([1, 2]) / 3.0)  # log-thirds of 1..top
    masks = {
        "DC": k == 0,
        "low": (k >= 1) & (k < edges[0]),
        "mid": (k >= edges[0]) & (k < edges[1]),
        "high": k >= edges[1],
    }
    return masks


def band_split(x, masks, H):  # x (..., H) -> dict band -> (..., H)
    F = np.fft.rfft(x, axis=-1)
    return {b: np.fft.irfft(F * m, n=H, axis=-1) for b, m in masks.items()}


def band_periods(masks, H):
    out = {}
    for b, m in masks.items():
        ks = np.nonzero(m)[0]
        if b == "DC":
            out[b] = "level"
        else:
            out[b] = f"{H / ks.max():.1f}-{H / ks.min():.1f} steps"
    return out


# ----------------------------------------------------------------------------- data
def find_csv(data_root, data_path):
    hits = glob.glob(os.path.join(data_root, "**", data_path), recursive=True)
    return hits[0] if hits else None


def test_windows(s, data_root):
    """Rebuild the test-set inputs/targets exactly as data_provider does (features M, scale on)."""
    path = find_csv(data_root, s["data_path"])
    if path is None:
        return None, None, f"csv {s['data_path']} not found under {data_root}"
    L, H, data = s["seq_len"], s["pred_len"], s["data"]
    df = pd.read_csv(path)
    if data in ("ETTh1", "ETTh2", "ETTm1", "ETTm2"):
        f = 4 if data.startswith("ETTm") else 1
        b1 = [0, 12 * 30 * 24 * f - L, 12 * 30 * 24 * f + 4 * 30 * 24 * f - L]
        b2 = [12 * 30 * 24 * f, 12 * 30 * 24 * f + 4 * 30 * 24 * f, 12 * 30 * 24 * f + 8 * 30 * 24 * f]
        vals = df[df.columns[1:]].values.astype(np.float64)
    else:  # Dataset_Custom: target column moved last
        cols = list(df.columns)
        cols.remove("OT")
        cols.remove("date")
        df = df[["date"] + cols + ["OT"]]
        n = len(df)
        ntr, nte = int(n * 0.7), int(n * 0.2)
        b1 = [0, ntr - L, n - nte - L]
        b2 = [ntr, n - nte, n]
        vals = df[df.columns[1:]].values.astype(np.float64)
    tr = vals[b1[0]:b2[0]]
    vals = (vals - tr.mean(0)) / tr.std(0)  # sklearn StandardScaler (ddof=0)
    seg = vals[b1[2]:b2[2]]
    return seg, (L, H), None


# ----------------------------------------------------------------------------- probe
def ridge_r2(Xtr, Ytr, Xte, Yte, lams=(1e-1, 1e0, 1e1, 1e2, 1e3)):
    """Ridge with lambda picked on the last 20% of the training part; returns held-out R^2."""
    def fit(X, Y, lam):
        Xb = np.hstack([X, np.ones((len(X), 1), X.dtype)]).astype(np.float64)
        return np.linalg.solve(Xb.T @ Xb + lam * np.eye(Xb.shape[1]), Xb.T @ Y)

    def pred(W, X):
        return np.hstack([X, np.ones((len(X), 1), X.dtype)]) @ W

    cut = int(len(Xtr) * 0.8)
    best = min(lams, key=lambda lam: np.mean((pred(fit(Xtr[:cut], Ytr[:cut], lam), Xtr[cut:]) - Ytr[cut:]) ** 2))
    W = fit(Xtr, Ytr, best)
    sse = np.sum((pred(W, Xte) - Yte) ** 2)
    sst = np.sum((Yte - Ytr.mean(0)) ** 2)
    return 1 - sse / sst


def probe(run_dir, s, pred, true, masks, data_root, max_channels, max_windows):
    seg, (L, H), err = test_windows(s, data_root)
    if err:
        return None, err
    N = len(seg) - L - H + 1
    if N != pred.shape[0]:
        return None, f"window count mismatch ({N} rebuilt vs {pred.shape[0]} saved)"
    C = pred.shape[2]
    chans = np.arange(C) if C <= max_channels else np.sort(np.random.default_rng(0).choice(C, max_channels, replace=False))
    stride = max(1, int(np.ceil(N / max_windows)))
    idx = np.arange(0, N, stride)
    X = np.stack([seg[i:i + L] for i in idx])[:, :, chans]          # (n, L, c)
    Yrec = np.stack([seg[i + L:i + L + H] for i in idx])[:, :, chans]
    if np.abs(Yrec - true[idx][:, :, chans]).max() > 1e-3:
        return None, "rebuilt targets do not match true.npy (different data file or split?)"
    R = (pred - true)[idx][:, :, chans]                              # (n, H, c)
    mu, sd = X.mean(1, keepdims=True), np.sqrt(X.var(1, keepdims=True) + 1e-5)
    Xn = ((X - mu) / sd).transpose(0, 2, 1)                          # (n, c, L)  RevIN view
    Rn = (R / sd).transpose(0, 2, 1)                                 # (n, c, H)  same scale
    parts = band_split(Rn, masks, H)
    n = len(idx)
    gap = int(np.ceil((L + H) / stride))
    ntr = int(n * 0.6)
    tr, te = np.arange(ntr), np.arange(ntr + gap, n)
    if len(te) < 20:
        return None, "too few test windows for the probe"
    flat = lambda a, rows: a[rows].reshape(-1, a.shape[-1]).astype(np.float32)
    r2 = {b: ridge_r2(flat(Xn, tr), flat(parts[b], tr), flat(Xn, te), flat(parts[b], te)) for b in BANDS}
    return r2, None


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", default="./results")
    ap.add_argument("--data_root", default="./dataset")
    ap.add_argument("--match", default="", help="only run folders whose name contains this text")
    ap.add_argument("--no_probe", action="store_true", help="skip Q2 (no csv needed)")
    ap.add_argument("--max_channels", type=int, default=32)
    ap.add_argument("--max_windows", type=int, default=2000)
    ap.add_argument("--out_csv", default="error_spectrum.csv")
    a = ap.parse_args()

    rows, by_group = [], defaultdict(dict)
    dirs = sorted(d for d in glob.glob(os.path.join(a.results_dir, "*")) if a.match in os.path.basename(d))
    for d in dirs:
        js, pp, tp = (os.path.join(d, f) for f in ("summary.json", "pred.npy", "true.npy"))
        if not (os.path.isfile(js) and os.path.isfile(pp) and os.path.isfile(tp)):
            continue
        s = json.load(open(js))
        parsed = parse_model_id(s.get("model_id", ""))
        if not parsed:
            continue
        pred, true = np.load(pp).astype(np.float64), np.load(tp).astype(np.float64)
        H = pred.shape[1]
        masks = band_masks(H)
        r = np.moveaxis(pred - true, 1, -1)       # (N, C, H)
        y = np.moveaxis(true, 1, -1)
        rb, yb = band_split(r, masks, H), band_split(y, masks, H)
        mse = {b: float(np.mean(rb[b] ** 2)) for b in BANDS}
        tot = sum(mse.values())
        row = dict(variant=s["variant"], ds=parsed[0], H=H, tag=parsed[1], seed=s.get("seed"),
                   mse_total=tot, mse_check=float(np.mean(r ** 2)))
        for b in BANDS:
            row[f"share_{b}"] = mse[b] / tot
            row[f"ratio_{b}"] = mse[b] / float(np.mean(yb[b] ** 2))
            row[f"mse_{b}"] = mse[b]
        if not a.no_probe:
            r2, err = probe(d, s, pred, true, masks, a.data_root, a.max_channels, a.max_windows)
            for b in BANDS:
                row[f"r2_{b}"] = None if r2 is None else r2[b]
            row["probe_note"] = err or ""
        rows.append(row)
        by_group[(row["ds"], H, row["tag"], row["seed"])][row["variant"]] = row
        print(f"done {os.path.basename(d)}")

    if not rows:
        print("no finished runs with pred.npy/true.npy found")
        return

    periods = band_periods(band_masks(rows[0]["H"]), rows[0]["H"])
    print("\nbands for H=%d: " % rows[0]["H"] + ", ".join(f"{b} = {periods[b]}" for b in BANDS))
    print("share = band's part of the test MSE | ratio = band MSE / band power of the truth (1.0 = nothing captured)")
    print("R2 = held-out ridge R^2 of the band error from the input (>0: linearly recoverable signal left)\n")
    hdr = f"{'run':<28}" + "".join(f"{b + ' share/ratio/R2':>24}" for b in BANDS)
    print(hdr)
    print("-" * len(hdr))
    for key in sorted(by_group, key=lambda k: (k[0], k[2], k[1])):
        for v in sorted(by_group[key], key=lambda v: (v != "PhaseFormer", "-frozen" in v, v)):
            row = by_group[key].get(v)
            if not row:
                continue
            name = f"{key[0]}{'[' + key[2] + ']' if key[2] else ''} H{key[1]} {v.replace('CycleSpec-', 'CS-')}"
            cells = ""
            for b in BANDS:
                r2 = row.get(f"r2_{b}")
                r2s = "  -  " if r2 is None else f"{r2:+.3f}"
                cells += f"{row[f'share_{b}'] * 100:>9.1f}% {row[f'ratio_{b}']:>5.2f} {r2s:>7}"
            print(f"{name:<28}{cells}")
            if row.get("probe_note"):
                print(f"{'':<28}probe skipped: {row['probe_note']}")
        base = by_group[key].get("PhaseFormer")
        for v in [v for v in by_group[key] if v != "PhaseFormer"]:
            cs = by_group[key].get(v)
            if base and cs:
                ch = "  ".join(f"{b} {(cs[f'mse_{b}'] - base[f'mse_{b}']) / base[f'mse_{b}'] * 100:+.1f}%" for b in BANDS)
                print(f"{'':<28}{v.replace('CycleSpec-', 'CS-')} vs PhaseFormer, per-band MSE: {ch}")
        print()

    with open(a.out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {a.out_csv}")


if __name__ == "__main__":
    main()
