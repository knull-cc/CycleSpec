"""Summarise the multi-seed arms (A/B/C/D/...) written by scripts/plugin/run_etth1.sh. numpy only.

    python tools/summarize_arms.py 'results/*etth1_pl96_*'

Reads results/<setting>/metrics.npy ([mae, mse, rmse, mape, mspe]); the arm and seed come from the
setting suffix  ..._<ARM>_s<SEED>_<itr>.  For every arm: mean +- std over seeds, and the paired
difference to the reference arm (default A) on matching seeds, with win counts.
"""
import glob
import re
import sys
from collections import defaultdict

import numpy as np

PAT = re.compile(r'_([A-Za-z0-9]+)_s(\d+)_(\d+)/?$')


def collect(pattern):
    runs = defaultdict(dict)  # arm -> seed -> (mse, mae)
    for d in sorted(glob.glob(pattern)):
        m = PAT.search(d)
        if not m:
            continue
        try:
            mae, mse = np.load(d.rstrip('/') + '/metrics.npy')[:2]
        except OSError:
            continue
        runs[m.group(1)][int(m.group(2))] = (float(mse), float(mae))
    return runs


def main(pattern, ref='A'):
    runs = collect(pattern)
    if not runs:
        raise SystemExit('no finished runs match ' + pattern)
    print('%-5s %-3s  %-17s %-17s %s' % ('arm', 'n', 'MSE mean+-std', 'MAE mean+-std', 'paired diff vs %s (negative = better)' % ref))
    for arm in sorted(runs):
        seeds = sorted(runs[arm])
        mse = np.array([runs[arm][s][0] for s in seeds])
        mae = np.array([runs[arm][s][1] for s in seeds])
        line = '%-5s %-3d  %.4f +- %.4f   %.4f +- %.4f' % (arm, len(seeds), mse.mean(), mse.std(ddof=1) if len(seeds) > 1 else 0.0,
                                                          mae.mean(), mae.std(ddof=1) if len(seeds) > 1 else 0.0)
        if arm != ref and ref in runs:
            common = sorted(set(seeds) & set(runs[ref]))
            if common:
                dm = np.array([runs[arm][s][0] - runs[ref][s][0] for s in common])
                da = np.array([runs[arm][s][1] - runs[ref][s][1] for s in common])
                rm = np.mean([runs[arm][s][0] / runs[ref][s][0] - 1 for s in common]) * 100
                ra = np.mean([runs[arm][s][1] / runs[ref][s][1] - 1 for s in common]) * 100
                line += '   MSE %+.4f (%+.2f%%, %d/%d seeds better)  MAE %+.4f (%+.2f%%, %d/%d better)' % (
                    dm.mean(), rm, int((dm < 0).sum()), len(common), da.mean(), ra, int((da < 0).sum()), len(common))
        print(line)
    print('\nNote: only compare "plugin+EMA" with "baseline+EMA" (arms D vs B), and "plugin" with "baseline" (C vs A).')
    print('A gain smaller than the seed std is not evidence.')


if __name__ == '__main__':
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else 'A')
