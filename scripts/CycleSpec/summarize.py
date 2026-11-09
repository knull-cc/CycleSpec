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

DS_ORDER = ["etth1", "etth2", "ettm1", "ettm2", "weather", "electricity", "traffic"]
MODEL_ID_RE = re.compile(r"^(?P<ds>[A-Za-z0-9]+)_pl(?P<h>\d+)(?P<rest>(?:_[A-Za-z0-9]+)*)$")


def parse_model_id(mid):
    """<ds>_pl<H>[_<tag>][_cs<variant>][_frozen] -> (ds, H, tag) or None"""
    m = MODEL_ID_RE.match(mid or "")
    if not m:
        return None
    toks = [t for t in m.group("rest").split("_") if t]
    if toks and toks[-1] == "frozen":
        toks.pop()
    if toks and toks[-1].startswith("cs"):
        toks.pop()
    if len(toks) > 1:
        return None
    return m.group("ds"), int(m.group("h")), (toks[0] if toks else "")


def variant_key(v):
    """PhaseFormer first, then CycleSpec variants, joint before frozen."""
    return (0 if v == "PhaseFormer" else 1, "-frozen" in v, v)


def pct(new, ref):
    return (new - ref) / ref * 100.0


def sort_key(kv):
    ds, h, tag, seed = kv[0]
    return (DS_ORDER.index(ds) if ds in DS_ORDER else 99, ds, tag, h, str(seed))


def label(ds, tag):
    return f"{ds}[{tag}]" if tag else ds


def compact(groups):
    """One row per (dataset, H, tag, seed, variant): MSE/MAE and % change vs PhaseFormer of the same group."""
    w = max(12, max(len(label(k[0], k[2])) for k in groups) + 1)
    vw = max(18, max(len(s["variant"]) for runs in groups.values() for s in runs) + 1)
    hdr = f"{'dataset':<{w}}{'H':>4} {'seed':>5}  {'variant':<{vw}}{'MSE':>8}{'MAE':>8}{'dMSE%':>8}{'dMAE%':>8}{'trainable':>10}"
    print(hdr)
    print("-" * len(hdr))
    wins = defaultdict(lambda: [0, 0, 0, 0.0, 0.0])  # variant -> [mse wins, mae wins, n, sum dMSE, sum dMAE]
    for (ds, h, tag, seed), runs in sorted(groups.items(), key=sort_key):
        runs = sorted(runs, key=lambda s: variant_key(s["variant"]))
        b = next((s for s in runs if s["variant"] == "PhaseFormer"), None)
        for s in runs:
            rel = b is not None and s is not b
            dm = f"{pct(s['mse'], b['mse']):+.2f}" if rel else "-"
            da = f"{pct(s['mae'], b['mae']):+.2f}" if rel else "-"
            print(f"{label(ds, tag):<{w}}{h:>4} {str(seed):>5}  {s['variant']:<{vw}}{s['mse']:>8.4f}{s['mae']:>8.4f}"
                  f"{dm:>8}{da:>8}{s.get('params_trainable', 0):>10}")
            if rel:
                x = wins[s["variant"]]
                x[0] += s["mse"] < b["mse"]
                x[1] += s["mae"] < b["mae"]
                x[2] += 1
                x[3] += pct(s["mse"], b["mse"])
                x[4] += pct(s["mae"], b["mae"])
        print()
    print(f"{'variant':<{vw}}{'beats MSE':>10}{'beats MAE':>10}{'mean dMSE%':>12}{'mean dMAE%':>12}")
    for v in sorted(wins, key=variant_key):
        wm, wa, n, sm, sa = wins[v]
        print(f"{v:<{vw}}{f'{wm}/{n}':>10}{f'{wa}/{n}':>10}{sm / n:>+12.2f}{sa / n:>+12.2f}")


def seed_mean(groups):
    """Per (dataset, H, tag, variant): mean MSE/MAE over seeds, mean and std over seeds of the
    paired % change vs PhaseFormer (same seed), and in how many seeds the variant wins."""
    import statistics as st
    agg = defaultdict(lambda: defaultdict(list))  # (ds,h,tag) -> variant -> [(seed, run, base)]
    for (ds, h, tag, seed), runs in groups.items():
        b = next((s for s in runs if s["variant"] == "PhaseFormer"), None)
        for s in runs:
            agg[(ds, h, tag)][s["variant"]].append((seed, s, b))
    print("\n=== mean over seeds (dMSE/dMAE paired with PhaseFormer of the same seed; +- = std over seeds) ===")
    vw = max(18, max(len(v) for d in agg.values() for v in d) + 1)
    hdr = (f"{'dataset':<14}{'H':>4}  {'variant':<{vw}}{'seeds':>6}{'MSE':>8}{'MAE':>8}"
           f"{'dMSE%':>15}{'dMAE%':>15}{'wins MSE/MAE':>14}")
    print(hdr)
    print("-" * len(hdr))
    overall = defaultdict(lambda: [[], []])
    for key in sorted(agg, key=lambda k: (DS_ORDER.index(k[0]) if k[0] in DS_ORDER else 99, k[2], k[1])):
        ds, h, tag = key
        for v in sorted(agg[key], key=variant_key):
            rows = agg[key][v]
            mse = st.mean(r["mse"] for _, r, _ in rows)
            mae = st.mean(r["mae"] for _, r, _ in rows)
            paired = [(pct(r["mse"], b["mse"]), pct(r["mae"], b["mae"])) for _, r, b in rows if b is not None and r is not b]
            if v == "PhaseFormer" or not paired:
                dm = da = "-"
                w = ""
            else:
                ms, as_ = [p[0] for p in paired], [p[1] for p in paired]
                sd = lambda x: st.stdev(x) if len(x) > 1 else 0.0
                dm = f"{st.mean(ms):+.2f}+-{sd(ms):.2f}"
                da = f"{st.mean(as_):+.2f}+-{sd(as_):.2f}"
                w = f"{sum(x < 0 for x in ms)}/{len(ms)} {sum(x < 0 for x in as_)}/{len(as_)}"
                overall[v][0].extend(ms)
                overall[v][1].extend(as_)
            print(f"{label(ds, tag):<14}{h:>4}  {v:<{vw}}{len(rows):>6}{mse:>8.4f}{mae:>8.4f}{dm:>15}{da:>15}{w:>14}")
        print()
    if overall:
        print(f"{'all H and seeds':<20}{'variant':<{vw}}{'n':>4}{'mean dMSE%':>12}{'mean dMAE%':>12}{'wins MSE':>10}{'wins MAE':>10}")
        for v in sorted(overall, key=variant_key):
            ms, as_ = overall[v]
            print(f"{'':<20}{v:<{vw}}{len(ms):>4}{st.mean(ms):>+12.2f}{st.mean(as_):>+12.2f}"
                  f"{f'{sum(x < 0 for x in ms)}/{len(ms)}':>10}{f'{sum(x < 0 for x in as_)}/{len(as_)}':>10}")


def detailed(groups):
    for (ds, h, tag, seed), runs in sorted(groups.items(), key=sort_key):
        runs.sort(key=lambda s: variant_key(s["variant"]))
        base = next((s for s in runs if s["variant"] == "PhaseFormer"), None)
        print(f"\n== {label(ds, tag)}  L={runs[0]['seq_len']} -> H={h}  seed={seed} ==")
        hdr = (f"{'variant':<26}{'MSE':>9}{'MAE':>9}{'dMSE%':>8}{'dMAE%':>8}{'best/ran':>10}"
               f"{'backbone':>10}{'adapter':>9}{'trainable':>10}{'time(s)':>9}")
        print(hdr)
        print("-" * len(hdr))
        for s in runs:
            rel = base is not None and s is not base
            dm = f"{pct(s['mse'], base['mse']):+.2f}" if rel else "-"
            da = f"{pct(s['mae'], base['mae']):+.2f}" if rel else "-"
            t = s.get("train_time_s")
            print(f"{s['variant']:<26}{s['mse']:>9.4f}{s['mae']:>9.4f}{dm:>8}{da:>8}"
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
        for fz in [s for s in runs if "-frozen" in s["variant"]]:
          if fz.get("init_vali_loss") is not None and base.get("best_vali_loss") is not None:
            d = pct(fz["init_vali_loss"], base["best_vali_loss"])
            print(f"  {fz['variant']}: epoch-0 vali {fz['init_vali_loss']:.7f} vs PhaseFormer best vali {base['best_vali_loss']:.7f} "
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
    ap.add_argument("--seed_mean", action="store_true", help="also print the mean over seeds (and seed std of dMSE/dMAE)")
    a = ap.parse_args()
    want_tag = None if a.tag is None else ("" if a.tag in ("", "none") else a.tag)

    groups = defaultdict(list)
    for p in glob.glob(os.path.join(a.results_dir, "*", "summary.json")):
        with open(p) as f:
            s = json.load(f)
        parsed = parse_model_id(s.get("model_id", ""))
        if not parsed:
            continue
        ds, h, tag = parsed
        if (a.ds and ds != a.ds) or (a.pred_len and h != a.pred_len) or \
                (a.seed is not None and s.get("seed") != a.seed) or (want_tag is not None and tag != want_tag):
            continue
        groups[(ds, h, tag, s.get("seed"))].append(s)

    if not groups:
        print("no summary.json matched under", a.results_dir)
        return
    (compact if a.compact else detailed)(groups)
    if a.seed_mean:
        seed_mean(groups)


if __name__ == "__main__":
    main()
