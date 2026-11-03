#!/bin/bash
# PhaseFormer vs CycleSpec on one dataset, in parallel.
#   bash scripts/CycleSpec/compare.sh etth1          # H=96 (default)
#   bash scripts/CycleSpec/compare.sh etth1 "96 192" # several horizons
#
#   1. PhaseFormer (base)  and  CycleSpec (joint)  start at the same time
#   2. when PhaseFormer is done, CycleSpec (frozen) starts on its checkpoint
#   3. when everything is done, prints the comparison table
# Logs: ./logs/<dataset>/{PhaseFormer,CycleSpec,CycleSpec-frozen}_pl<H>_s<seed>.log
# SEED / GPU / NUM_WORKERS / DATA_ROOT / CS_* / BS / RUN_TAG can be set as for the run_*.sh scripts.

cd "$(dirname "$0")/../.."
DS=${1:?usage: compare.sh <etth1|etth2|ettm1|ettm2|weather|electricity|traffic> [pred_lens]}
export PRED_LENS=${2:-96}
export SEED=${SEED:-2021}
SCRIPT=scripts/run_${DS}.sh
[ -f "$SCRIPT" ] || { echo "no such script: $SCRIPT"; exit 1; }

echo "[$(date +%T)] ${DS} H=${PRED_LENS} seed=${SEED}: PhaseFormer + CycleSpec(joint) in parallel"
MODEL=PhaseFormer bash "$SCRIPT" > /dev/null 2>&1 &
PID_BASE=$!
MODEL=CycleSpec CS_MODE=joint bash "$SCRIPT" > /dev/null 2>&1 &
PID_JOINT=$!

wait $PID_BASE
echo "[$(date +%T)] PhaseFormer done (exit $?); starting CycleSpec(frozen)"
MODEL=CycleSpec CS_MODE=frozen bash "$SCRIPT" > /dev/null 2>&1
echo "[$(date +%T)] CycleSpec(frozen) done (exit $?)"

wait $PID_JOINT
echo "[$(date +%T)] CycleSpec(joint) done (exit $?)"

for H in $PRED_LENS; do
  python scripts/CycleSpec/summarize.py --ds "$DS" --pred_len "$H" --seed "$SEED" --tag "${RUN_TAG:-${BS:+bs$BS}}"
done
