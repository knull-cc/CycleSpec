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
#   SAVE_DIR      if set (e.g. /content/drive/MyDrive/CycleSpec_runs), each job's logs, results
#                 and PhaseFormer checkpoints are copied there as soon as that job finishes
#   SPLIT_H       0 (default): one job per dataset, its horizons run one after another
#                 1: one job per (dataset, H), so all horizons of a dataset run at the same time;
#                    MAX_PARALLEL then counts (dataset, H) jobs
#   BS, RUN_TAG   non-official batch size / tag, see scripts/CycleSpec/common.sh
# Progress: bash scripts/CycleSpec/status.sh

cd "$(dirname "$0")/../.."
export DATA_ROOT=${DATA_ROOT:-./resources/all_datasets}
export NUM_WORKERS=${NUM_WORKERS:-1}
export SEED=${SEED:-2021}
DATASETS=${DATASETS:-"etth1 etth2 ettm1 ettm2 weather electricity traffic"}
PRED_LENS=${PRED_LENS:-"96 192 336 720"}
MAX_PARALLEL=${MAX_PARALLEL:-7}
SAVE_DIR=${SAVE_DIR:-}
SPLIT_H=${SPLIT_H:-0}
export BS=${BS:-}
export RUN_TAG=${RUN_TAG:-${BS:+bs$BS}}
TAG_SUFFIX=${RUN_TAG:+_$RUN_TAG}

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
JOBS=()
for ds in $DATASETS; do
  if [ "$SPLIT_H" = "1" ]; then
    for H in $PRED_LENS; do JOBS+=("$ds:$H"); done
  else
    JOBS+=("$ds:$PRED_LENS")
  fi
done
echo "[$(date +%T)] start: ${#JOBS[@]} jobs, datasets=[$DATASETS] H=[$PRED_LENS] seed=$SEED tag=${RUN_TAG:-none} batch=${BS:-official} max_parallel=$MAX_PARALLEL"

for job in "${JOBS[@]}"; do
  ds=${job%%:*}
  hs=${job#*:}
  name="${ds}${TAG_SUFFIX}_$( [ "$SPLIT_H" = "1" ] && echo "pl$hs" || echo all )"
  while [ "$(jobs -rp | wc -l)" -ge "$MAX_PARALLEL" ]; do sleep 15; done
  (
    bash scripts/CycleSpec/compare.sh "$ds" "$hs" > "logs/compare_${name}.log" 2>&1
    echo "[$(date +%T)] finished $name"
    if [ -n "$SAVE_DIR" ]; then
      mkdir -p "$SAVE_DIR/logs/$ds" "$SAVE_DIR/results" "$SAVE_DIR/checkpoints"
      cp "logs/compare_${name}.log" "$SAVE_DIR/logs/" 2>/dev/null
      for H in $hs; do
        cp logs/$ds/*_pl${H}${TAG_SUFFIX}_s${SEED}.log "$SAVE_DIR/logs/$ds/" 2>/dev/null
        cp -r results/long_term_forecast_${ds}_pl${H}${TAG_SUFFIX}_* "$SAVE_DIR/results/" 2>/dev/null
        cp -r checkpoints/long_term_forecast_${ds}_pl${H}${TAG_SUFFIX}_PhaseFormer_* "$SAVE_DIR/checkpoints/" 2>/dev/null
      done
      echo "[$(date +%T)] copied $name to $SAVE_DIR"
    fi
  ) &
  echo "[$(date +%T)] launched $name"
  sleep 2
done
wait

SUMMARY=logs/summary_all${TAG_SUFFIX}.txt
python scripts/CycleSpec/summarize.py --compact --seed "$SEED" --tag "${RUN_TAG:-none}" > "$SUMMARY"
[ -n "$SAVE_DIR" ] && cp "$SUMMARY" "$SAVE_DIR/"
echo "[$(date +%T)] all done -> $SUMMARY"
cat "$SUMMARY"
