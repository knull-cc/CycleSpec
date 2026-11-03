#!/bin/bash
# Shared settings for scripts/run_*.sh. Sourced, not run directly.
#
# Every value can be overridden from the environment, e.g.
#   MODEL=CycleSpec PRED_LENS="96" GPU=0 bash scripts/run_etth1.sh
#
#   MODEL        PhaseFormer (base) | CycleSpec (base + adapter)            default PhaseFormer
#   CS_MODE      joint  : adapter and backbone trained together from scratch default joint
#                frozen : load the PhaseFormer checkpoint of the same dataset/H/seed,
#                         freeze it and train the adapter only (run MODEL=PhaseFormer first)
#   PRED_LENS    horizons to run, e.g. "96" or "96 192"                    default "96 192 336 720"
#   SEED         random seed                                                default 2021
#   GPU          GPU index                                                  default 0
#   NUM_WORKERS  dataloader workers                                         default 2
#   DATA_ROOT    folder that holds ETT-small/, weather/, electricity/, traffic/
#                                                                           default ./resources/all_datasets
#   LOG_DIR      logs go to $LOG_DIR/<dataset>/<variant>_pl<H>_s<seed>.log default ./logs
#   CS_HIDDEN CS_WINDOWS CS_KERNEL CS_BRANCHES CS_ACT   adapter hyper-parameters
#   FORCE=1      rerun even if the run already finished (default: finished runs are skipped)
#   VARHA=1      PhaseFormer only: run the VarHA (TT-HA) test-time protocol on the frozen model.
#                If the PhaseFormer run already finished, its checkpoint is reused (no retraining);
#                skipped if varha.json already exists (FORCE=1 reruns VarHA only).

MODEL=${MODEL:-PhaseFormer}
CS_MODE=${CS_MODE:-joint}
SEED=${SEED:-2021}
GPU=${GPU:-0}
NUM_WORKERS=${NUM_WORKERS:-2}
DATA_ROOT=${DATA_ROOT:-./resources/all_datasets}
LOG_DIR=${LOG_DIR:-./logs}
VARHA=${VARHA:-0}

CS_HIDDEN=${CS_HIDDEN:-8}
CS_WINDOWS=${CS_WINDOWS:-30,15,7}
CS_KERNEL=${CS_KERNEL:-3}
CS_BRANCHES=${CS_BRANCHES:-both}
CS_ACT=${CS_ACT:-gelu}

# run_one <dataset_id> <pred_len> <run.py args...>
#   model_id = <dataset_id>_pl<H>          (PhaseFormer and CycleSpec joint; the model name
#                                           is part of the checkpoint/result folder name)
#            = <dataset_id>_pl<H>_frozen   (CycleSpec frozen)
run_one() {
  local DS_ID=$1 PRED_LEN=$2
  shift 2
  local VARIANT=$MODEL
  local MODEL_ID="${DS_ID}_pl${PRED_LEN}"
  local CS_ARGS=()

  if [ "$MODEL" = "CycleSpec" ]; then
    CS_ARGS=(--cs_hidden "$CS_HIDDEN" --cs_windows "$CS_WINDOWS" --cs_kernel "$CS_KERNEL"
             --cs_branches "$CS_BRANCHES" --cs_act "$CS_ACT" --cs_train_mode "$CS_MODE")
    if [ "$CS_MODE" = "frozen" ]; then
      VARIANT="CycleSpec-frozen"
      MODEL_ID="${MODEL_ID}_frozen"
      # only use a base checkpoint whose run FINISHED (has summary.json); a killed run
      # leaves a half-trained checkpoint behind
      local BASE_RES BASE_CKPT=""
      BASE_RES=$(ls -d ./results/long_term_forecast_${DS_ID}_pl${PRED_LEN}_PhaseFormer_*_s${SEED}_0 2>/dev/null | head -n 1)
      if [ -n "$BASE_RES" ] && [ -f "$BASE_RES/summary.json" ]; then
        BASE_CKPT="./checkpoints/$(basename "$BASE_RES")/checkpoint.pth"
      fi
      if [ -z "$BASE_CKPT" ] || [ ! -f "$BASE_CKPT" ]; then
        echo "❌ CS_MODE=frozen needs a finished PhaseFormer run for ${DS_ID} H=${PRED_LEN} seed=${SEED}; run MODEL=PhaseFormer first"
        return 1
      fi
      CS_ARGS+=(--cs_init_ckpt "$BASE_CKPT")
    elif [ "$CS_MODE" != "joint" ]; then
      echo "❌ CS_MODE must be joint or frozen, got: $CS_MODE"; return 1
    fi
  elif [ "$MODEL" != "PhaseFormer" ]; then
    echo "❌ MODEL must be PhaseFormer or CycleSpec, got: $MODEL"; return 1
  fi

  # skip runs that already finished (FORCE=1 to rerun)
  local DONE IS_TRAIN=1
  DONE=$(ls ./results/long_term_forecast_${MODEL_ID}_${MODEL}_*_s${SEED}_0/summary.json 2>/dev/null | head -n 1)
  if [ "$VARHA" = "1" ]; then
    [ "$MODEL" = "PhaseFormer" ] || { echo "❌ VARHA=1 needs MODEL=PhaseFormer"; return 1; }
    if [ -n "$DONE" ] && [ -f "$(dirname "$DONE")/varha.json" ] && [ "${FORCE:-0}" != "1" ]; then
      echo "⏭  skip VarHA | ${DS_ID} | H=${PRED_LEN} | seed=${SEED} (varha.json exists)"
      return 0
    fi
    if [ -n "$DONE" ] && [ -f "./checkpoints/$(basename "$(dirname "$DONE")")/checkpoint.pth" ]; then
      IS_TRAIN=0   # finished PhaseFormer run: reuse its checkpoint, run VarHA only
    fi
  elif [ -n "$DONE" ] && [ "${FORCE:-0}" != "1" ]; then
    echo "⏭  skip ${VARIANT} | ${DS_ID} | H=${PRED_LEN} | seed=${SEED} (already finished)"
    return 0
  fi

  mkdir -p "$LOG_DIR/$DS_ID"
  local LOG="$LOG_DIR/$DS_ID/${VARIANT}_pl${PRED_LEN}_s${SEED}.log"
  [ "$IS_TRAIN" = "0" ] && LOG="$LOG_DIR/$DS_ID/${VARIANT}_pl${PRED_LEN}_s${SEED}_varha.log"
  echo "🚀 ${VARIANT} | ${DS_ID} | H=${PRED_LEN} | seed=${SEED} | log: ${LOG}"

  python -u run.py \
    --task_name long_term_forecast \
    --is_training "$IS_TRAIN" \
    --varha "$VARHA" \
    --model "$MODEL" \
    --model_id "$MODEL_ID" \
    --fix_seed "$SEED" \
    --des "s${SEED}" \
    --use_gpu 1 \
    --gpu "$GPU" \
    --num_workers "$NUM_WORKERS" \
    "$@" "${CS_ARGS[@]}" 2>&1 | tee "$LOG"
  return "${PIPESTATUS[0]}"
}
