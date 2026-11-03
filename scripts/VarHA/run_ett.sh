#!/bin/bash
# VarHA (TT-HA plug-in) on the ETT datasets. Every (dataset, H) is its own process;
# up to MAX_PARALLEL run at the same time (default 16 = 4 datasets x 4 horizons).
#
#   DATA_ROOT=./dataset nohup bash scripts/VarHA/run_ett.sh > logs_varha.txt 2>&1 &
#   bash scripts/VarHA/status.sh               # progress / results, any time
#   bash scripts/CycleSpec/stop.sh             # stop everything
#
# Per (dataset, H): train PhaseFormer (skipped if that run already finished, its checkpoint
# is reused), then the VarHA protocol on the frozen model: hyper-parameters chosen on val
# only, test run once. Finished VarHA runs (varha.json present) are skipped.
#
# Env vars:
#   DATA_ROOT     folder with ETT-small/                       default ./resources/all_datasets
#   DATASETS      default "etth1 etth2 ettm1 ettm2"
#   PRED_LENS     default "96 192 336 720"
#   MAX_PARALLEL  processes at once, default 16 (1 = strictly one after another)
#   NUM_WORKERS   dataloader workers per process, default 1
#   SEED, GPU     default 2021, 0
#   SAVE_DIR      if set, logs + results of each run are copied there when it finishes
#   FORCE=1       rerun VarHA even if varha.json exists (PhaseFormer is still reused)

cd "$(dirname "$0")/../.."
export DATA_ROOT=${DATA_ROOT:-./resources/all_datasets}
export SEED=${SEED:-2021}
export NUM_WORKERS=${NUM_WORKERS:-1}
export MODEL=PhaseFormer VARHA=1
DATASETS=${DATASETS:-"etth1 etth2 ettm1 ettm2"}
PRED_LENS_ALL=${PRED_LENS:-"96 192 336 720"}
MAX_PARALLEL=${MAX_PARALLEL:-16}
SAVE_DIR=${SAVE_DIR:-}

for ds in $DATASETS; do
  case $ds in etth1) f=ETTh1;; etth2) f=ETTh2;; ettm1) f=ETTm1;; ettm2) f=ETTm2;;
    *) echo "❌ unknown ETT dataset: $ds"; exit 1;; esac
  if [ -f "$DATA_ROOT/ETT-small/$f.csv" ]; then echo "✅ $DATA_ROOT/ETT-small/$f.csv"
  else echo "❌ missing $DATA_ROOT/ETT-small/$f.csv"; exit 1; fi
done
if pgrep -f "[r]un.py --task_name" > /dev/null; then
  echo "❌ run.py processes are already running; not starting a second batch"; exit 1
fi

mkdir -p logs
echo "[$(date +%T)] start: datasets=[$DATASETS] H=[$PRED_LENS_ALL] seed=$SEED max_parallel=$MAX_PARALLEL"
for ds in $DATASETS; do
  for H in $PRED_LENS_ALL; do
    while [ "$(jobs -rp | wc -l)" -ge "$MAX_PARALLEL" ]; do sleep 15; done
    (
      PRED_LENS="$H" bash scripts/run_${ds}.sh > "logs/varha_${ds}_pl${H}.log" 2>&1
      rc=$?
      line=$(grep -a "\[VarHA\] test B2_VarHA" "logs/varha_${ds}_pl${H}.log" | tail -n 1)
      echo "[$(date +%T)] finished $ds H=$H (exit $rc) ${line:-'(no VarHA result, see logs/varha_'${ds}'_pl'${H}'.log)'}"
      if [ -n "$SAVE_DIR" ]; then
        mkdir -p "$SAVE_DIR/logs/$ds" "$SAVE_DIR/results"
        cp logs/$ds/PhaseFormer_pl${H}_s${SEED}*.log "logs/varha_${ds}_pl${H}.log" "$SAVE_DIR/logs/$ds/" 2>/dev/null
        cp -r results/long_term_forecast_${ds}_pl${H}_PhaseFormer_* "$SAVE_DIR/results/" 2>/dev/null
      fi
    ) &
    echo "[$(date +%T)] launched $ds H=$H"
    sleep 2
  done
done
wait

bash scripts/VarHA/status.sh | tee logs/summary_varha.txt
[ -n "$SAVE_DIR" ] && cp logs/summary_varha.txt "$SAVE_DIR/"
echo "[$(date +%T)] all done -> logs/summary_varha.txt"
