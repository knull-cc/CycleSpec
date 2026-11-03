#!/bin/bash
# All datasets x all horizons x {PhaseFormer, CycleSpec-joint, CycleSpec-frozen}, one seed.
#
#   DATA_ROOT=./dataset nohup bash scripts/CycleSpec/run_all.sh > logs_run_all.txt 2>&1 &
#
# Each dataset runs scripts/CycleSpec/compare.sh (PhaseFormer and CycleSpec-joint in parallel,
# then CycleSpec-frozen on the PhaseFormer checkpoints). Datasets run in parallel, at most
# MAX_PARALLEL at a time (each one uses up to 2 GPU processes).
#
# Env vars:
#   DATA_ROOT     folder with ETT-small/ weather/ electricity/ traffic/   default ./resources/all_datasets
#   DATASETS      default "etth1 etth2 ettm1 ettm2 weather electricity traffic"
#   PRED_LENS     default "96 192 336 720"
#   MAX_PARALLEL  datasets at once, default 7
#   NUM_WORKERS   dataloader workers per process, default 1
#   SEED, GPU     default 2021, 0
#   SAVE_DIR      if set (e.g. /content/drive/MyDrive/CycleSpec_runs), each dataset's logs and
#                 results are copied there as soon as that dataset finishes
# Progress: bash scripts/CycleSpec/status.sh

cd "$(dirname "$0")/../.."
export DATA_ROOT=${DATA_ROOT:-./resources/all_datasets}
export NUM_WORKERS=${NUM_WORKERS:-1}
export SEED=${SEED:-2021}
DATASETS=${DATASETS:-"etth1 etth2 ettm1 ettm2 weather electricity traffic"}
PRED_LENS=${PRED_LENS:-"96 192 336 720"}
MAX_PARALLEL=${MAX_PARALLEL:-7}
SAVE_DIR=${SAVE_DIR:-}

data_file() {
  case $1 in
    etth1) echo ETT-small/ETTh1.csv ;;
    etth2) echo ETT-small/ETTh2.csv ;;
    ettm1) echo ETT-small/ETTm1.csv ;;
    ettm2) echo ETT-small/ETTm2.csv ;;
    weather) echo weather/weather.csv ;;
    electricity) echo electricity/electricity.csv ;;
    traffic) echo traffic/traffic.csv ;;
    *) echo "" ;;
  esac
}

# 1. every data file must be there before anything starts
missing=0
for ds in $DATASETS; do
  f=$(data_file "$ds")
  if [ -z "$f" ]; then echo "❌ unknown dataset: $ds"; missing=1; continue; fi
  if [ -f "$DATA_ROOT/$f" ]; then echo "✅ $DATA_ROOT/$f"; else echo "❌ missing $DATA_ROOT/$f"; missing=1; fi
done
[ $missing -eq 1 ] && { echo "fix the data paths first (DATA_ROOT=$DATA_ROOT)"; exit 1; }

# 2. refuse to start twice
if pgrep -f "run.py --task_name" > /dev/null; then
  echo "❌ run.py processes are already running; not starting a second batch"; exit 1
fi

mkdir -p logs
echo "[$(date +%T)] start: datasets=[$DATASETS] H=[$PRED_LENS] seed=$SEED max_parallel=$MAX_PARALLEL"

for ds in $DATASETS; do
  while [ "$(jobs -rp | wc -l)" -ge "$MAX_PARALLEL" ]; do sleep 15; done
  (
    bash scripts/CycleSpec/compare.sh "$ds" "$PRED_LENS" > "logs/compare_${ds}.log" 2>&1
    echo "[$(date +%T)] finished $ds"
    if [ -n "$SAVE_DIR" ]; then
      mkdir -p "$SAVE_DIR/logs" "$SAVE_DIR/results"
      cp -r "logs/$ds" "logs/compare_${ds}.log" "$SAVE_DIR/logs/" 2>/dev/null
      cp -r results/long_term_forecast_${ds}_pl* "$SAVE_DIR/results/" 2>/dev/null
      echo "[$(date +%T)] copied $ds to $SAVE_DIR"
    fi
  ) &
  echo "[$(date +%T)] launched $ds"
  sleep 2
done
wait

python scripts/CycleSpec/summarize.py --seed "$SEED" > logs/summary_all.txt
[ -n "$SAVE_DIR" ] && cp logs/summary_all.txt "$SAVE_DIR/"
echo "[$(date +%T)] all done -> logs/summary_all.txt"
cat logs/summary_all.txt
