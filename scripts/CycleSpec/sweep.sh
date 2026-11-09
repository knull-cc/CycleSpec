#!/bin/bash
# Port / architecture sweep for CycleSpec on one dataset, several seeds and horizons.
#
#   DATA_ROOT=./dataset nohup bash scripts/CycleSpec/sweep.sh etth1 > logs_sweep_etth1.txt 2>&1 &
#
# For every (seed, H) job:
#   1. PhaseFormer and every CycleSpec variant (joint training) start together
#   2. when PhaseFormer is done, every CycleSpec variant is trained again on the frozen PhaseFormer
# Finished runs are skipped, so the sweep can be stopped (scripts/CycleSpec/stop.sh) and restarted.
#
# Env vars:
#   SEEDS        default "2021 2022 2023"
#   PRED_LENS    default "96 192 336 720"
#   VARIANTS     port:arch list, default "input:spectral block:spectral both:spectral block:dense"
#   MODES        default "joint frozen"
#   MAX_PARALLEL (seed, H) jobs at once, default 4
#   DATA_ROOT, BS, RUN_TAG, NUM_WORKERS, GPU  as in scripts/CycleSpec/common.sh
# Progress:  python scripts/CycleSpec/summarize.py --compact --ds <ds> --seed_mean

cd "$(dirname "$0")/../.."
DS=${1:?usage: sweep.sh <etth1|etth2|ettm1|ettm2|weather|electricity|traffic>}
SCRIPT=scripts/run_${DS}.sh
[ -f "$SCRIPT" ] || { echo "no such script: $SCRIPT"; exit 1; }
export DATA_ROOT=${DATA_ROOT:-./resources/all_datasets}
export NUM_WORKERS=${NUM_WORKERS:-1}
SEEDS=${SEEDS:-"2021 2022 2023"}
PRED_LENS=${PRED_LENS:-"96 192 336 720"}
VARIANTS=${VARIANTS:-"input:spectral block:spectral both:spectral block:dense"}
MODES=${MODES:-"joint frozen"}
MAX_PARALLEL=${MAX_PARALLEL:-4}

if pgrep -f "run.py --task_name" > /dev/null; then
  echo "❌ run.py processes are already running; not starting a second batch"; exit 1
fi
mkdir -p logs

job() {  # seed H
  local S=$1 H=$2 pids=() v port arch
  SEED=$S PRED_LENS=$H MODEL=PhaseFormer bash "$SCRIPT" > /dev/null 2>&1 &
  local base_pid=$!
  if [[ " $MODES " == *" joint "* ]]; then
    for v in $VARIANTS; do
      port=${v%%:*}; arch=${v#*:}
      SEED=$S PRED_LENS=$H MODEL=CycleSpec CS_MODE=joint CS_PORT=$port CS_ARCH=$arch bash "$SCRIPT" > /dev/null 2>&1 &
      pids+=($!)
    done
  fi
  wait $base_pid
  if [[ " $MODES " == *" frozen "* ]]; then
    for v in $VARIANTS; do
      port=${v%%:*}; arch=${v#*:}
      SEED=$S PRED_LENS=$H MODEL=CycleSpec CS_MODE=frozen CS_PORT=$port CS_ARCH=$arch bash "$SCRIPT" > /dev/null 2>&1 &
      pids+=($!)
    done
  fi
  wait "${pids[@]}"
  echo "[$(date +%T)] finished seed=$S H=$H"
}

n=0
for S in $SEEDS; do for H in $PRED_LENS; do n=$((n + 1)); done; done
echo "[$(date +%T)] sweep $DS: $n jobs, seeds=[$SEEDS] H=[$PRED_LENS] variants=[$VARIANTS] modes=[$MODES] batch=${BS:-official}"
for S in $SEEDS; do
  for H in $PRED_LENS; do
    while [ "$(jobs -rp | wc -l)" -ge "$MAX_PARALLEL" ]; do sleep 10; done
    job "$S" "$H" &
    echo "[$(date +%T)] launched seed=$S H=$H"
    sleep 2
  done
done
wait

TAGARG=${RUN_TAG:-${BS:+bs$BS}}
python scripts/CycleSpec/summarize.py --compact --ds "$DS" --tag "${TAGARG:-none}" --seed_mean | tee "logs/sweep_${DS}${TAGARG:+_$TAGARG}.txt"
echo "[$(date +%T)] sweep done -> logs/sweep_${DS}${TAGARG:+_$TAGARG}.txt"
