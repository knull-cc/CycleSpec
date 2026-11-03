"""Compare PhaseFormer and CycleSpec runs.

    python scripts/CycleSpec/summarize.py --compact                  # one row per dataset/H, MSE+MAE
    python scripts/CycleSpec/summarize.py --ds etth1 --pred_len 96   # detailed table
    python scripts/CycleSpec/summarize.py --compact --tag bs256      # only runs tagged bs256
    python scripts/CycleSpec/summarize.py --compact --tag none       # only official-config runs

Reads ./results/*/summary.json (written at the end of each run). Runs are grouped by
dataset, horizon, tag and seed (model_id = <ds>_pl<H>[_<tag>][_frozen]).
"""
import argparse
import glob
import json
import os
import re
from collections import defaultdict

ORDER = {"PhaseFormer": 0, "CycleSpec-joint": 1, "CycleSpec-frozen": 2}
DS_ORDER = ["etth1", "etth2", "ettm1", "ettm2", "weather", "electricity", "traffic"]
MODEL_ID_RE = re.compile(r"^(?P<ds>[A-Za-z0-9]+)_pl(?P<h>\d+)(?:_(?P<tag>(?!frozen$)[A-Za-z0-9]+))?(?P<fz>_frozen)?$")


def pct(new, ref):
    return (new - ref) / ref * 100.0


def sort_key(kv):
    ds, h, tag, seed = kv[0]
    return (DS_ORDER.index(ds) if ds in DS_ORDER else 99, ds, tag, h, str(seed))


def label(ds, tag):
    return f"{ds}[{tag}]" if tag else ds


def compact(groups):
    """One row per (dataset, H, tag, seed): MSE/MAE for each variant and % change vs PhaseFormer."""
    def cell(s):
        return f"{s['mse']:.4f} {s['mae']:.4f}" if s else f"{'-':>6} {'-':>6}"

    def delta(s, b):
        if not s or not b:
            return f"{'.':>15}"
        return f"{pct(s['mse'], b['mse']):+6.2f}% {pct(s['mae'], b['mae']):+6.2f}%"

    w = max(12, max(len(label(k[0], k[2])) for k in groups) + 1)
    hdr = (f"{'dataset':<{w}}{'H':>4}  {'PhaseFormer':^13}  {'CycleSpec(joint)':^13}  {'dMSE   dMAE':^15}  "
           f"{'CycleSpec(frozen)':^13}  {'dMSE   dMAE':^15}")
    print(hdr)
    print(f"{'':<{w}}{'':>4}  {'MSE    MAE':^13}  {'MSE    MAE':^13}  {'':^15}  {'MSE    MAE':^13}")
    print("-" * len(hdr))
    wins = defaultdict(lambda: [0, 0, 0])
    for (ds, h, tag, seed), runs in sorted(groups.items(), key=sort_key):
        by = {s["variant"]: s for s in runs}
        b, j, f = by.get("PhaseFormer"), by.get("CycleSpec-joint"), by.get("CycleSpec-frozen")
        print(f"{label(ds, tag):<{w}}{h:>4}  {cell(b)}  {cell(j)}  {delta(j, b)}  {cell(f)}  {delta(f, b)}")
        for name, s in (("joint", j), ("frozen", f)):
            if s and b:
                wins[name][0] += s["mse"] < b["mse"]
                wins[name][1] += s["mae"] < b["mae"]
                wins[name][2] += 1
    print()
    for name, (wm, wa, n) in wins.items():
        print(f"CycleSpec({name}) beats PhaseFormer: MSE {wm}/{n}, MAE {wa}/{n}")


def detailed(groups):
    for (ds, h, tag, seed), runs in sorted(groups.items(), key=sort_key):
        runs.sort(key=lambda s: ORDER.get(s["variant"], 9))
        base = next((s for s in runs if s["variant"] == "PhaseFormer"), None)
        print(f"\n== {label(ds, tag)}  L={runs[0]['seq_len']} -> H={h}  seed={seed} ==")
        hdr = (f"{'variant':<18}{'MSE':>9}{'MAE':>9}{'dMSE%':>8}{'dMAE%':>8}{'best/ran':>10}"
               f"{'backbone':>10}{'adapter':>9}{'trainable':>10}{'time(s)':>9}")
        print(hdr)
        print("-" * len(hdr))
        for s in runs:
            rel = base is not None and s is not base
            dm = f"{pct(s['mse'], base['mse']):+.2f}" if rel else "-"
            da = f"{pct(s['mae'], base['mae']):+.2f}" if rel else "-"
            t = s.get("train_time_s")
            print(f"{s['variant']:<18}{s['mse']:>9.4f}{s['mae']:>9.4f}{dm:>8}{da:>8}"
                  f"{str(s.get('best_epoch')) + '/' + str(s.get('epochs_run')):>10}"
                  f"{s.get('params_backbone', 0):>10}{s.get('params_adapter', 0):>9}{s.get('params_trainable', 0):>10}"
                  f"{(f'{t:.0f}' if t else '-'):>9}")
        if base is None:
            print("(no PhaseFormer run in this group yet)")
            continue
        for s in runs:
            if s is base:
                continue
            both = s["mse"] < base["mse"] and s["mae"] < base["mae"]
            print(f"  {s['variant']}: {'better than' if both else 'NOT better than'} PhaseFormer on both MSE and MAE")
        fz = next((s for s in runs if s["variant"] == "CycleSpec-frozen"), None)
        if fz and fz.get("init_vali_loss") is not None and base.get("best_vali_loss") is not None:
            d = pct(fz["init_vali_loss"], base["best_vali_loss"])
            print(f"  frozen epoch-0 vali {fz['init_vali_loss']:.7f} vs PhaseFormer best vali {base['best_vali_loss']:.7f} "
                  f"({d:+.2f}%; the validation loader is shuffled, so small differences are expected)")
            if fz.get("best_epoch") == 0:
                print("  note: the frozen adapter never beat PhaseFormer on validation, so it fell back to the base model")
    print("\nOne seed per group: on ETTh1 a gap of a few tenths of a percent is within seed noise.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds", default=None, help="etth1, etth2, ettm1, ettm2, weather, electricity, traffic")
    ap.add_argument("--pred_len", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--tag", default=None, help="only this tag (e.g. bs256); 'none' or '' = official-config runs only")
    ap.add_argument("--results_dir", default="./results")
    ap.add_argument("--compact", action="store_true", help="one row per dataset/H with MSE and MAE of every variant")
    a = ap.parse_args()
    want_tag = None if a.tag is None else ("" if a.tag in ("", "none") else a.tag)

    groups = defaultdict(list)
    for p in glob.glob(os.path.join(a.results_dir, "*", "summary.json")):
        with open(p) as f:
            s = json.load(f)
        m = MODEL_ID_RE.match(s.get("model_id", ""))
        if not m:
            continue
        ds, h, tag = m.group("ds"), int(m.group("h")), m.group("tag") or ""
        if (a.ds and ds != a.ds) or (a.pred_len and h != a.pred_len) or \
                (a.seed is not None and s.get("seed") != a.seed) or (want_tag is not None and tag != want_tag):
            continue
        groups[(ds, h, tag, s.get("seed"))].append(s)

    if not groups:
        print("no summary.json matched under", a.results_dir)
        return
    (compact if a.compact else detailed)(groups)


if __name__ == "__main__":
    main()
