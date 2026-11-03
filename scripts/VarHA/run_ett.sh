#!/bin/bash
# VarHA (TT-HA plug-in) on the ETT datasets, one dataset at a time, one run at a time.
#
#   DATA_ROOT=./dataset DATASETS="etth1" nohup bash scripts/VarHA/run_ett.sh > logs_varha.txt 2>&1 &
#   python scripts/VarHA/summarize.py          # progress / results, any time
#   bash scripts/CycleSpec/stop.sh             # stop everything
#
# Per (dataset, H): train PhaseFormer (skipped if that run already finished, its checkpoint
# is reused), then the VarHA protocol on the frozen model: hyper-parameters chosen on val
# only, test run once. Finished VarHA runs (varha.json present) are skipped.
#
# Env vars:
#   DATA_ROOT   folder with ETT-small/                         default ./resources/all_datasets
#   DATASETS    default "etth1 etth2 ettm1 ettm2"
#   PRED_LENS   default "96 192 336 720"
#   SEED, GPU, NUM_WORKERS                                     default 2021, 0, 2
#   SAVE_DIR    if set, logs + results of each dataset are copied there when it finishes
#   FORCE=1     rerun VarHA even if varha.json exists (PhaseFormer is still reused)

cd "$(dirname "$0")/../.."
export DATA_ROOT=${DATA_ROOT:-./resources/all_datasets}
export SEED=${SEED:-2021}
export MODEL=PhaseFormer VARHA=1
DATASETS=${DATASETS:-"etth1 etth2 ettm1 ettm2"}
export PRED_LENS=${PRED_LENS:-"96 192 336 720"}
SAVE_DIR=${SAVE_DIR:-}

for ds in $DATASETS; do
  case $ds in etth1) f=ETTh1;; etth2) f=ETTh2;; ettm1) f=ETTm1;; ettm2) f=ETTm2;;
    *) echo "❌ unknown ETT dataset: $ds"; exit 1;; esac
  [ -f "$DATA_ROOT/ETT-small/$f.csv" ] || { echo "❌ missing $DATA_ROOT/ETT-small/$f.csv"; exit 1; }
done
if pgrep -f "run.py --task_name" > /dev/null; then
  echo "❌ run.py processes are already running; not starting a second batch"; exit 1
fi

mkdir -p logs
for ds in $DATASETS; do
  echo "[$(date +%T)] start $ds  H=[$PRED_LENS] seed=$SEED"
  bash scripts/run_${ds}.sh > "logs/varha_${ds}.log" 2>&1
  echo "[$(date +%T)] finished $ds (exit $?)"
  grep -a "\[VarHA\] test\|\[VarHA\] selected\|Error\|Traceback" "logs/varha_${ds}.log"
  if [ -n "$SAVE_DIR" ]; then
    mkdir -p "$SAVE_DIR/logs" "$SAVE_DIR/results"
    cp -r "logs/$ds" "logs/varha_${ds}.log" "$SAVE_DIR/logs/" 2>/dev/null
    cp -r results/long_term_forecast_${ds}_pl*_PhaseFormer_* "$SAVE_DIR/results/" 2>/dev/null
    echo "[$(date +%T)] copied $ds to $SAVE_DIR"
  fi
done

python scripts/VarHA/summarize.py --seed "$SEED" | tee logs/summary_varha.txt
[ -n "$SAVE_DIR" ] && cp logs/summary_varha.txt "$SAVE_DIR/"
echo "[$(date +%T)] all done -> logs/summary_varha.txt"
