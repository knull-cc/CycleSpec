"""Seed-ensemble ceiling: how much can ANY variance-reduction trick (EMA, BDO, augmentation, ...)
give on this backbone?  numpy only.

Usage (after running several seeds, each with a distinct --des so results do not overwrite):
    python tools/seed_ensemble.py 'results/*PhaseFormer_ETTh1*_A_s*_0'
    python tools/seed_ensemble.py dirA dirB dirC ...        # explicit result dirs

Each result dir must contain pred.npy and true.npy (written by exp.test()).

Math. For N seeds with predictions p_i and truth y, p_bar = mean_i p_i:
    mean_i MSE(p_i) = MSE(p_bar) + mean_i mean|p_i - p_bar|^2          (exact)
Model  MSE(p_i) = B + V  in expectation (B = MSE of the expected prediction, V = seed variance),
       MSE(p_bar of N) = B + V/N.  Unbiased  V_hat = N/(N-1) * mean_i mean|p_i - p_bar|^2.
So an infinite ensemble would reach about  B_hat = mean_single - V_hat; that is the ceiling
for a mechanism that only removes seed variance. Bias (B) is not touched by averaging.
MAE of the mean prediction is reported as measured (no such decomposition).
"""
import glob
import itertools
import sys

import numpy as np


def load(dirs):
    P, Y = [], None
    for d in dirs:
        p = np.load(d.rstrip('/') + '/pred.npy').astype(np.float64)
        y = np.load(d.rstrip('/') + '/true.npy').astype(np.float64)
        if Y is None:
            Y = y
        elif Y.shape != y.shape or not np.allclose(Y, y):
            raise SystemExit('true.npy differs between %s and the first dir: different split / settings' % d)
        P.append(p)
    return np.stack(P), Y  # (N, S, H, C), (S, H, C)


def mse(p, y):
    return float(np.mean((p - y) ** 2))


def mae(p, y):
    return float(np.mean(np.abs(p - y)))


def analyse(P, Y):
    N = P.shape[0]
    single_mse = np.array([mse(p, Y) for p in P])
    single_mae = np.array([mae(p, Y) for p in P])
    pbar = P.mean(axis=0)
    out = dict(N=N, single_mse=single_mse, single_mae=single_mae,
               ens_mse=mse(pbar, Y), ens_mae=mae(pbar, Y))
    spread = float(np.mean([(p - pbar) ** 2 for p in P]))  # mean_i mean |p_i - p_bar|^2
    out['spread'] = spread
    out['identity_gap'] = float(single_mse.mean() - out['ens_mse'] - spread)  # should be ~0
    if N > 1:
        v_hat = spread * N / (N - 1)
        out['v_hat'] = v_hat
        out['b_hat'] = float(single_mse.mean() - v_hat)
    # ensemble-size curve (all subsets when cheap, else first-k)
    curve = {}
    for k in range(1, N + 1):
        combos = list(itertools.combinations(range(N), k)) if N <= 8 else [tuple(range(k))]
        ms = [mse(P[list(c)].mean(axis=0), Y) for c in combos]
        ma = [mae(P[list(c)].mean(axis=0), Y) for c in combos]
        curve[k] = (float(np.mean(ms)), float(np.mean(ma)))
    out['curve'] = curve
    return out


def report(o):
    N = o['N']
    sm, sa = o['single_mse'], o['single_mae']
    print('seeds: %d' % N)
    print('single seed  MSE %.4f +- %.4f   MAE %.4f +- %.4f   (best MSE %.4f)'
          % (sm.mean(), sm.std(ddof=1) if N > 1 else 0.0, sa.mean(), sa.std(ddof=1) if N > 1 else 0.0, sm.min()))
    print('ensemble(N)  MSE %.4f   MAE %.4f' % (o['ens_mse'], o['ens_mae']))
    g_mse = (sm.mean() - o['ens_mse']) / sm.mean() * 100
    g_mae = (sa.mean() - o['ens_mae']) / sa.mean() * 100
    print('ensemble gain over mean single seed:  MSE %+.2f%%   MAE %+.2f%%' % (g_mse, g_mae))
    if N > 1:
        g_inf = (sm.mean() - o['b_hat']) / sm.mean() * 100
        print('estimated N->inf ceiling (MSE only):  %+.2f%%   [V_hat=%.5f, B_hat=%.4f]'
              % (g_inf, o['v_hat'], o['b_hat']))
    print('identity check  mean_single - ens - spread = %.2e (should be ~0)' % o['identity_gap'])
    print('ensemble size k -> mean MSE / MAE over subsets')
    for k, (m, a) in o['curve'].items():
        print('  k=%d  MSE %.4f  MAE %.4f' % (k, m, a))
    print()
    print('Reading: a variance-reduction plug-in (EMA, BDO, augmentation) should not be expected to beat the')
    print('ceiling above on average. If the ceiling is < ~1%, this whole family has no room on this setting.')


if __name__ == '__main__':
    args = sys.argv[1:]
    if not args:
        raise SystemExit(__doc__)
    dirs = []
    for a in args:
        hits = sorted(glob.glob(a))
        dirs.extend(hits if hits else [a])
    print('result dirs:')
    for d in dirs:
        print('  ', d)
    P, Y = load(dirs)
    report(analyse(P, Y))
