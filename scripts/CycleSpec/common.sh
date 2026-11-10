#!/bin/bash
# Shared settings for scripts/run_*.sh. Sourced, not run directly.
#
# Every value can be overridden from the environment, e.g.
#   MODEL=CycleSpec PRED_LENS="96" GPU=0 bash scripts/run_etth1.sh
#
#   MODEL        PhaseFormer (base) | CycleSpec (base + adapter)            default PhaseFormer
#   CS_MODE      joint  : adapter and backbone trained together from scratch default joint
#                frozen : load the PhaseFormer checkpoint of the same dataset/H/seed/tag,
#                         freeze it and train the adapter only (run MODEL=PhaseFormer first)
#   PRED_LENS    horizons to run, e.g. "96" or "96 192"                    default "96 192 336 720"
#   SEED         random seed                                                default 2021
#   GPU          GPU index                                                  default 0
#   NUM_WORKERS  dataloader workers                                         default 2
#   DATA_ROOT    folder that holds ETT-small/, weather/, electricity/, traffic/
#                                                                           default ./resources/all_datasets
#   LOG_DIR      logs go to $LOG_DIR/<dataset>/<variant>_pl<H>[_<tag>]_s<seed>.log   default ./logs
#   CS_HIDDEN CS_WINDOWS CS_KERNEL CS_BRANCHES CS_ACT   adapter hyper-parameters
#   CS_PORT      input (default, before the embedding) | block (after attention in every layer) | both
#   CS_ARCH      spectral (default) | dense (Linear(T->T) instead of the STFT branch: capacity control)
#   CS_BLOCK_HIDDEN  bottleneck of the block-port adapters (0 = min(8, latent_dim))
#   FORCE=1      rerun even if the run already finished (default: finished runs are skipped)
#
#   Non-official settings (kept apart from the official-config runs by a tag):
#   BS           override the dataset's batch size, e.g. BS=256. Sets the tag to bs<BS>
#                unless RUN_TAG is given. Nothing else changes.
#   RUN_TAG      tag added to model_id / folder / log names (letters and digits only)
#   EVAL_TEST    1 (default, upstream behaviour): evaluate the test set after every epoch
#                (printed only). 0 skips it: faster, but changes the RNG stream / data order.

MODEL=${MODEL:-PhaseFormer}
CS_MODE=${CS_MODE:-joint}
SEED=${SEED:-2021}
GPU=${GPU:-0}
NUM_WORKERS=${NUM_WORKERS:-2}
DATA_ROOT=${DATA_ROOT:-./resources/all_datasets}
LOG_DIR=${LOG_DIR:-./logs}

CS_HIDDEN=${CS_HIDDEN:-8}
CS_WINDOWS=${CS_WINDOWS:-30,15,7}
CS_KERNEL=${CS_KERNEL:-3}
CS_BRANCHES=${CS_BRANCHES:-both}
CS_ACT=${CS_ACT:-gelu}
CS_PORT=${CS_PORT:-input}          # input | block | both | output | io
CS_ARCH=${CS_ARCH:-spectral}       # spectral | dense
CS_BLOCK_HIDDEN=${CS_BLOCK_HIDDEN:-0}
# CycleSpec variants other than the original (input port, spectral) get their own name token,
# e.g. etth1_pl96_csblk, etth1_pl96_csbothdense; PhaseFormer runs never carry it
case "$CS_PORT" in input) _P="" ;; block) _P="blk" ;; both) _P="both" ;; output) _P="out" ;; io) _P="io" ;; *) _P="?" ;; esac
[ "$CS_ARCH" = "dense" ] && _A="dense" || _A=""
if [ -n "$_P$_A" ]; then CS_VTAG="cs${_P:-in}${_A}"; else CS_VTAG=""; fi

BS=${BS:-}
RUN_TAG=${RUN_TAG:-${BS:+bs$BS}}
if [ -n "$RUN_TAG" ] && ! [[ "$RUN_TAG" =~ ^[A-Za-z0-9]+$ ]]; then
  echo "❌ RUN_TAG must be letters and digits only, got: $RUN_TAG"; exit 1
fi
EVAL_TEST=${EVAL_TEST:-1}
TAG_SUFFIX=${RUN_TAG:+_$RUN_TAG}

# run_one <dataset_id> <pred_len> <run.py args...>
#   model_id = <dataset_id>_pl<H>[_<tag>]          PhaseFormer and CycleSpec joint (the model
#                                                  name is part of the checkpoint/result folder)
#            = <dataset_id>_pl<H>[_<tag>]_frozen   CycleSpec frozen
run_one() {
  local DS_ID=$1 PRED_LEN=$2
  shift 2
  local VARIANT=$MODEL
  local BASE_ID="${DS_ID}_pl${PRED_LEN}${TAG_SUFFIX}"
  local MODEL_ID="$BASE_ID"
  local CS_ARGS=()
  local EXTRA_ARGS=(--eval_test_each_epoch "$EVAL_TEST")
  [ -n "$BS" ] && EXTRA_ARGS+=(--batch_size "$BS")

  if [ "$MODEL" = "CycleSpec" ]; then
    CS_ARGS=(--cs_hidden "$CS_HIDDEN" --cs_windows "$CS_WINDOWS" --cs_kernel "$CS_KERNEL"
             --cs_branches "$CS_BRANCHES" --cs_act "$CS_ACT" --cs_train_mode "$CS_MODE"
             --cs_port "$CS_PORT" --cs_arch "$CS_ARCH" --cs_block_hidden "$CS_BLOCK_HIDDEN")
    MODEL_ID="${BASE_ID}${CS_VTAG:+_$CS_VTAG}"
    VARIANT="CycleSpec${CS_VTAG:+-$CS_VTAG}"
    if [ "$CS_MODE" = "frozen" ]; then
      VARIANT="CycleSpec${CS_VTAG:+-$CS_VTAG}-frozen"
      MODEL_ID="${BASE_ID}${CS_VTAG:+_$CS_VTAG}_frozen"
      # only use a base checkpoint whose run FINISHED (has summary.json); a killed run
      # leaves a half-trained checkpoint behind
      local BASE_RES BASE_CKPT=""
      BASE_RES=$(ls -d ./results/long_term_forecast_${BASE_ID}_PhaseFormer_*_s${SEED}_0 2>/dev/null | head -n 1)
      if [ -n "$BASE_RES" ] && [ -f "$BASE_RES/summary.json" ]; then
        BASE_CKPT="./checkpoints/$(basename "$BASE_RES")/checkpoint.pth"
      fi
      if [ -z "$BASE_CKPT" ] || [ ! -f "$BASE_CKPT" ]; then
        echo "❌ CS_MODE=frozen needs a finished PhaseFormer run for ${DS_ID} H=${PRED_LEN} seed=${SEED} tag=${RUN_TAG:-none}; run MODEL=PhaseFormer first"
        return 1
      fi
      CS_ARGS+=(--cs_init_ckpt "$BASE_CKPT")
    elif [ "$CS_MODE" != "joint" ]; then
      echo "❌ CS_MODE must be joint or frozen, got: $CS_MODE"; return 1
    fi
  elif [ "$MODEL" != "PhaseFormer" ]; then
    echo "❌ MODEL must be PhaseFormer or CycleSpec, got: $MODEL"; return 1
  fi

  local LABEL="${VARIANT} | ${DS_ID} | H=${PRED_LEN} | seed=${SEED}${RUN_TAG:+ | tag=$RUN_TAG}"

  # skip runs that already finished (FORCE=1 to rerun)
  local DONE
  DONE=$(ls ./results/long_term_forecast_${MODEL_ID}_${MODEL}_*_s${SEED}_0/summary.json 2>/dev/null | head -n 1)
  if [ -n "$DONE" ] && [ "${FORCE:-0}" != "1" ]; then
    echo "⏭  skip ${LABEL} (already finished)"
    return 0
  fi

  mkdir -p "$LOG_DIR/$DS_ID"
  local LOG="$LOG_DIR/$DS_ID/${VARIANT}_pl${PRED_LEN}${TAG_SUFFIX}_s${SEED}.log"
  echo "🚀 ${LABEL} | log: ${LOG}"

  # EXTRA_ARGS come after the dataset script's args so that --batch_size overrides it
  python -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --model "$MODEL" \
    --model_id "$MODEL_ID" \
    --fix_seed "$SEED" \
    --des "s${SEED}" \
    --use_gpu 1 \
    --gpu "$GPU" \
    --num_workers "$NUM_WORKERS" \
    "$@" "${CS_ARGS[@]}" "${EXTRA_ARGS[@]}" 2>&1 | tee "$LOG"
  return "${PIPESTATUS[0]}"
}
