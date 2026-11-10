#!/bin/bash
cd "$(dirname "$0")/../.."
# ETTh1, PhaseFormer official hyper-parameters (from neumyor/PhaseFormer run_etth1.py), several seeds.
#
# Arms (all share the same data, config and seeds):
#   A  PhaseFormer                      baseline
#   B  PhaseFormer + EMA                free baseline: compare plugins against THIS
#   C  PhaseFormerBDO                   BDO output head, stage-weighted loss
#   D  PhaseFormerBDO + EMA             compare with B
#   E  PhaseFormerBDO, last-stage loss  architecture-only ablation (no prefix supervision)
#   F  PhaseFormerBDO + EMA, last-stage loss
#   G  PhaseFormerBDO, equal stage weights
#   H  PhaseFormerBDO with 1 stage (sanity check: should match A up to init noise)
#   Re-parameterised head (PhaseFormerRep; merges back to a plain PhaseFormer after training):
#   I  cascade, 3 stages      (same architecture as E, new code path)
#   J  chain, depth 2, width 16
#   K  chain, depth 3, width 16
#   L  chain, depth 2, width 64
#   M  linear                 (sanity check: same as A up to RNG)
#   N  chain, depth 2, width 128
#   O  chain, depth 2, width 256
#
# Usage examples (repo root):
#   ARMS="A" bash scripts/plugin/run_etth1.sh        # step 1: seed-ensemble ceiling, baseline only
#   ARMS="A B C D" bash scripts/plugin/run_etth1.sh
#   PRED_LEN=192 SEEDS="2021 2022 2023" EMA_DECAY=0.95 ARMS="B D" bash scripts/plugin/run_etth1.sh
#
# EMA_DECAY is per optimizer step; ETTh1 at batch 256 has only ~30 steps per epoch, so 0.98 is a
# sane start (horizon ~50 steps). Pick it on validation loss only, never on test.

PRED_LEN=${PRED_LEN:-96}
SEEDS=${SEEDS:-"2021 2022 2023 2024 2025"}
ARMS=${ARMS:-"A B C D"}
EMA_DECAY=${EMA_DECAY:-0.98}
BDO_STAGES=${BDO_STAGES:-3}
GPU=${GPU:-0}
DATA_ROOT=${DATA_ROOT:-./resources/all_datasets}
ROOT_PATH=${ROOT_PATH:-$DATA_ROOT/ETT-small/}
# LR schedule: type3 = official PhaseFormer repo; constant = scripts/run_etth1.sh of this repo.
# Non-default schedules are tagged into --des (e.g. Nconstant_s2021) so results never overwrite.
LRADJ=${LRADJ:-type3}
if [ "$LRADJ" = "type3" ]; then LRTAG=""; else LRTAG=$(echo "$LRADJ" | tr -cd 'A-Za-z0-9'); fi

if [ "$PRED_LEN" -eq 720 ]; then
  LR=0.00015; LAT=32; ENC=128; PRED=256; ROUT=16; HEADS=2; ADROP=0.0; DELTA=0.3; EPOCHS=70; PAT=14
else
  LR=0.001;   LAT=4;  ENC=16;  PRED=32;  ROUT=8;  HEADS=1; ADROP=0.1; DELTA=1.0; EPOCHS=30; PAT=8
fi

for ARM in $ARMS; do
  STAGES=$BDO_STAGES; RM=chain; RD=2; RW=0
  case $ARM in
    A) MODEL=PhaseFormer;    EMA=0;           W=inv  ;;
    B) MODEL=PhaseFormer;    EMA=$EMA_DECAY;  W=inv  ;;
    C) MODEL=PhaseFormerBDO; EMA=0;           W=inv  ;;
    D) MODEL=PhaseFormerBDO; EMA=$EMA_DECAY;  W=inv  ;;
    E) MODEL=PhaseFormerBDO; EMA=0;           W=last ;;
    F) MODEL=PhaseFormerBDO; EMA=$EMA_DECAY;  W=last ;;
    G) MODEL=PhaseFormerBDO; EMA=0;           W=equal ;;
    H) MODEL=PhaseFormerBDO; EMA=0;           W=inv;  STAGES=1 ;;
    I) MODEL=PhaseFormerRep; EMA=0;           W=inv;  RM=cascade ;;
    J) MODEL=PhaseFormerRep; EMA=0;           W=inv;  RM=chain; RD=2; RW=16 ;;
    K) MODEL=PhaseFormerRep; EMA=0;           W=inv;  RM=chain; RD=3; RW=16 ;;
    L) MODEL=PhaseFormerRep; EMA=0;           W=inv;  RM=chain; RD=2; RW=64 ;;
    M) MODEL=PhaseFormerRep; EMA=0;           W=inv;  RM=linear ;;
    N) MODEL=PhaseFormerRep; EMA=0;           W=inv;  RM=chain; RD=2; RW=128 ;;
    O) MODEL=PhaseFormerRep; EMA=0;           W=inv;  RM=chain; RD=2; RW=256 ;;
    *) echo "unknown arm $ARM"; exit 1 ;;
  esac
  for SEED in $SEEDS; do
    echo "=== arm $ARM${LRTAG} lradj $LRADJ  model $MODEL  ema $EMA  w_mode $W  H=$PRED_LEN  seed $SEED ==="
    python run.py \
      --task_name long_term_forecast --is_training 1 \
      --model_id etth1_pl${PRED_LEN} --model $MODEL \
      --data ETTh1 --root_path $ROOT_PATH --data_path ETTh1.csv --features M --enc_in 7 \
      --seq_len 720 --pred_len $PRED_LEN \
      --train_epochs $EPOCHS --patience $PAT --batch_size 256 --learning_rate $LR --lradj $LRADJ \
      --use_gpu 1 --gpu $GPU --num_workers 2 \
      --period_len 24 --phase_layers 3 --latent_dim $LAT --phase_encoder_hidden $ENC --predictor_hidden $PRED \
      --phase_num_routers $ROUT --phase_attn_heads $HEADS --phase_attn_dropout $ADROP \
      --phase_attn_use_relpos 1 --phase_use_pos_embed 1 \
      --use_revin 1 --revin_affine 0 --revin_eps 1e-5 --use_huber 1 --huber_delta $DELTA \
      --fix_seed $SEED --ema_decay $EMA --bdo_stages $STAGES --rep_mode $RM --rep_depth $RD --rep_width $RW --rep_stages $BDO_STAGES --bdo_w_mode $W \
      --des ${ARM}${LRTAG}_s${SEED}
  done
done

echo
echo "=== per-arm summary ==="
python tools/summarize_arms.py "results/*etth1_pl${PRED_LEN}_*"
for ARM in $ARMS; do
  case $ARM in A|B) M=PhaseFormer ;; I|J|K|L|M|N|O) M=PhaseFormerRep ;; *) M=PhaseFormerBDO ;; esac
  echo
  echo "=== seed-ensemble ceiling, arm $ARM ==="
  python tools/seed_ensemble.py "results/*etth1_pl${PRED_LEN}_${M}_ETTh1_*_${ARM}${LRTAG}_s*_0" || true
done
