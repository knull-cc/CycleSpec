#!/bin/bash
# Progress of scripts/VarHA/run_ett.sh:  bash scripts/VarHA/status.sh
#   DATASETS="etth1 etth2 ettm1 ettm2" (default)  PRED_LENS="96 192 336 720"  SEED=2021
# PhaseFormer: test MSE when finished, "epN" while training (N finished epochs), "-" not started
# VarHA:       test MSE when finished, "run" while adapting, "wait" PhaseFormer done but VarHA
#              not run yet, "ERROR" if the log has a traceback
# dMSE / dMAE: VarHA vs PhaseFormer on test; both↓ = both went down; sel = config chosen on val

cd "$(dirname "$0")/../.."
export SEED=${SEED:-2021}
export DATASETS=${DATASETS:-"etth1 etth2 ettm1 ettm2"}
export PRED_LENS=${PRED_LENS:-"96 192 336 720"}

python3 - <<'PY'
import glob, json, os, re
seed, dss, hs = os.environ["SEED"], os.environ["DATASETS"].split(), os.environ["PRED_LENS"].split()

def read(p):
    try:
        return open(p, errors="ignore").read()
    except OSError:
        return None

def bad(txt):
    return txt is not None and re.search(r"Traceback|out of memory|Error", txt) is not None

print(f"{'dataset':<12} {'H':<5} {'PhaseFormer':<12} {'VarHA':<10} {'dMSE':>8} {'dMAE':>8}  {'both↓':<6} sel")
for ds in dss:
    for h in hs:
        res = glob.glob(f"results/long_term_forecast_{ds}_pl{h}_PhaseFormer_*_s{seed}_0")
        res = res[0] if res else None
        summ = os.path.join(res, "summary.json") if res else None
        vj = os.path.join(res, "varha.json") if res else None
        log_tr = read(f"logs/{ds}/PhaseFormer_pl{h}_s{seed}.log")        # train (+ VarHA)
        log_va = read(f"logs/{ds}/PhaseFormer_pl{h}_s{seed}_varha.log")  # VarHA on a reused checkpoint
        # PhaseFormer cell
        if summ and os.path.isfile(summ):
            pf = "{:.4f}".format(json.load(open(summ))["mse"])
        elif log_tr is not None:
            pf = "ERROR" if bad(log_tr) else "ep{}".format(len(re.findall(r"Epoch: \d+, Steps", log_tr)))
        else:
            pf = "-"
        # VarHA cell
        dm = da = both = sel = ""
        if vj and os.path.isfile(vj):
            v = json.load(open(vj))
            b, x = v["results"]["B0_PhaseFormer"], v["results"]["B2_VarHA"]
            va = "{:.4f}".format(x[0])
            dm, da = "{:+.2f}%".format((x[0] / b[0] - 1) * 100), "{:+.2f}%".format((x[1] / b[1] - 1) * 100)
            both = "yes" if (x[0] < b[0] and x[1] < b[1]) else "no"
            sel = v.get("selected_tag") or "none->identity"
        elif bad(log_va) or (log_tr and "VarHA (test-time" in log_tr and bad(log_tr)):
            va = "ERROR"
        elif log_va is not None or (log_tr and "VarHA (test-time" in log_tr):
            va = "run"
        elif pf not in ("-", "ERROR") and not pf.startswith("ep"):
            va = "wait"
        else:
            va = "-"
        print(f"{ds:<12} {h:<5} {pf:<12} {va:<10} {dm:>8} {da:>8}  {both:<6} {sel}")
PY

echo
echo "running run.py processes: $(pgrep -f '[r]un.py --task_name' | wc -l)"
grep -l "Traceback\|out of memory" logs/*/PhaseFormer_*.log logs/varha_*.log 2>/dev/null | sed 's/^/error in: /'
command -v nvidia-smi > /dev/null && nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader
free -g 2>/dev/null | awk 'NR==2{print "RAM used " $3 "G / " $2 "G"}'
