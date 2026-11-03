"""Compare PhaseFormer and CycleSpec runs.

    python scripts/CycleSpec/summarize.py                         # every dataset / H / seed found
    python scripts/CycleSpec/summarize.py --ds etth1 --pred_len 96

Reads ./results/*/summary.json (written at the end of each run). Runs are grouped by
dataset id, horizon and seed (model_id = <ds>_pl<H>[_frozen]).
"""
import argparse
import glob
import json
import os
import re
from collections import defaultdict

ORDER = {"PhaseFormer": 0, "CycleSpec-joint": 1, "CycleSpec-frozen": 2}


def pct(new, ref):
    return (new - ref) / ref * 100.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds", default=None, help="etth1, etth2, ettm1, ettm2, weather, electricity, traffic")
    ap.add_argument("--pred_len", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--results_dir", default="./results")
    a = ap.parse_args()

    groups = defaultdict(list)
    for p in glob.glob(os.path.join(a.results_dir, "*", "summary.json")):
        with open(p) as f:
            s = json.load(f)
        m = re.match(r"^(.+)_pl(\d+)(_frozen)?$", s.get("model_id", ""))
        if not m:
            continue
        ds, h = m.group(1), int(m.group(2))
        if (a.ds and ds != a.ds) or (a.pred_len and h != a.pred_len) or (a.seed is not None and s.get("seed") != a.seed):
            continue
        groups[(ds, h, s.get("seed"))].append(s)

    if not groups:
        print("no summary.json matched under", a.results_dir)
        return

    for (ds, h, seed), runs in sorted(groups.items(), key=lambda kv: (kv[0][0], kv[0][1], str(kv[0][2]))):
        runs.sort(key=lambda s: ORDER.get(s["variant"], 9))
        base = next((s for s in runs if s["variant"] == "PhaseFormer"), None)
        print(f"\n== {ds}  L={runs[0]['seq_len']} -> H={h}  seed={seed} ==")
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
            d = abs(fz["init_vali_loss"] - base["best_vali_loss"])
            print(f"  identity check (frozen epoch-0 vali == PhaseFormer best vali): "
                  f"{fz['init_vali_loss']:.7f} vs {base['best_vali_loss']:.7f} -> {'OK' if d < 1e-5 else 'MISMATCH'}")
            if fz.get("best_epoch") == 0:
                print("  note: the frozen adapter never beat PhaseFormer on validation, so it fell back to the base model")
    print("\nOne seed per group: on ETTh1 a gap of a few tenths of a percent is within seed noise.")


if __name__ == "__main__":
    main()
