"""VarHA results:  python scripts/VarHA/summarize.py [--ds etth1] [--seed 2021] [--abl]

One row per (dataset, H): frozen PhaseFormer vs PhaseFormer+VarHA on test (val-selected
config), and whether BOTH MSE and MAE went down. --abl adds the ablations.
"""
import argparse
import glob
import json
import os
import re

ap = argparse.ArgumentParser()
ap.add_argument("--ds", default=None)
ap.add_argument("--seed", type=int, default=None)
ap.add_argument("--abl", action="store_true")
ap.add_argument("--results_dir", default="./results")
a = ap.parse_args()

rows = []
for p in glob.glob(os.path.join(a.results_dir, "*_PhaseFormer_*", "summary.json")):
    s = json.load(open(p))
    m = re.match(r"^(.+)_pl(\d+)$", s.get("model_id", ""))
    if not m or (a.ds and m.group(1) != a.ds) or (a.seed is not None and s.get("seed") != a.seed):
        continue
    vj = os.path.join(os.path.dirname(p), "varha.json")
    v = json.load(open(vj)) if os.path.isfile(vj) else None
    rows.append((m.group(1), int(m.group(2)), s, v))

order = {"etth1": 0, "etth2": 1, "ettm1": 2, "ettm2": 3}
rows.sort(key=lambda r: (order.get(r[0], 9), r[0], r[1]))
pct = lambda x, b: (x / b - 1) * 100
print(f"{'dataset':<8}{'H':>5}{'PF MSE':>9}{'PF MAE':>9}{'VarHA MSE':>11}{'VarHA MAE':>11}"
      f"{'dMSE%':>8}{'dMAE%':>8}  {'both↓':<6}selected on val")
print("-" * 110)
for ds, h, s, v in rows:
    if v is None:
        print(f"{ds:<8}{h:>5}{s['mse']:>9.4f}{s['mae']:>9.4f}{'(VarHA not run yet)':>22}")
        continue
    r = v["results"]
    b, x = r["B0_PhaseFormer"], r["B2_VarHA"]
    both = "yes" if (x[0] < b[0] and x[1] < b[1]) else "no"
    sel = v.get("selected_tag") or "none -> identity"
    print(f"{ds:<8}{h:>5}{b[0]:>9.4f}{b[1]:>9.4f}{x[0]:>11.4f}{x[1]:>11.4f}"
          f"{pct(x[0], b[0]):>+8.2f}{pct(x[1], b[1]):>+8.2f}  {both:<6}{sel}")
    if abs(b[0] - s["mse"]) > 1e-4:
        print(f"   ! B0 {b[0]:.5f} differs from the PhaseFormer test MSE {s['mse']:.5f}")
    if a.abl:
        for k, (mse, mae) in r.items():
            if k not in ("B0_PhaseFormer", "B2_VarHA"):
                print(f"{'':<13}{k:<22}mse {pct(mse, b[0]):+.2f}%  mae {pct(mae, b[1]):+.2f}%")
running = os.popen("pgrep -f '[r]un.py --task_name' | wc -l").read().strip()
print(f"\nrunning run.py processes: {running or 0}   (one seed: a few tenths of a percent is within noise)")
