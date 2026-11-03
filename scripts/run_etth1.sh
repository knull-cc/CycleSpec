#!/bin/bash
# ETTh1. Hyper-parameters from neumyor/PhaseFormer run_etth1.py.
#   MODEL=PhaseFormer bash scripts/run_etth1.sh                    # base
#   MODEL=CycleSpec   bash scripts/run_etth1.sh                    # + CycleSpec (joint)
#   MODEL=CycleSpec CS_MODE=frozen bash scripts/run_etth1.sh       # adapter on the frozen base
#   PRED_LENS="96" MODEL=CycleSpec bash scripts/run_etth1.sh       # only H=96
# Other switches: see scripts/CycleSpec/common.sh
# Note: the original repo uses seed 2026 for H=720 (SEED=2026 to match).

cd "$(dirname "$0")/.."
source scripts/CycleSpec/common.sh

# 基础配置
DATA="ETTh1"
ROOT_PATH="$DATA_ROOT/ETT-small/"
DATA_PATH="ETTh1.csv"
ENC_IN=7
SEQ_LEN=720
BATCH_SIZE=256
USE_REVIN=1
REVIN_AFFINE=0
REVIN_EPS=1e-5

for PRED_LEN in ${PRED_LENS:-96 192 336 720}
do
  if [ "$PRED_LEN" -eq 720 ]; then
    LR=0.00015
    LAYERS=3
    LATENT_DIM=32
    ENCODER_HIDDEN=128
    PREDICTOR_HIDDEN=256
    NUM_ROUTERS=16
    ATT_HEADS=2
    ATT_DROPOUT=0.0
    USE_HUBER=1
    HUBER_DELTA=0.3
    TRAIN_EPOCHS=70
    PATIENCE=14
  else
    LR=0.001
    LAYERS=3
    LATENT_DIM=4
    ENCODER_HIDDEN=16
    PREDICTOR_HIDDEN=32
    NUM_ROUTERS=8
    ATT_HEADS=1
    ATT_DROPOUT=0.1
    USE_HUBER=1
    HUBER_DELTA=1.0
    TRAIN_EPOCHS=30
    PATIENCE=8
  fi

  run_one etth1 "$PRED_LEN" \
    --data $DATA \
    --root_path "$ROOT_PATH" \
    --data_path $DATA_PATH \
    --features M \
    --freq h \
    --enc_in $ENC_IN --dec_in $ENC_IN --c_out $ENC_IN \
    --seq_len $SEQ_LEN \
    --label_len 48 \
    --pred_len "$PRED_LEN" \
    --train_epochs $TRAIN_EPOCHS \
    --patience $PATIENCE \
    --batch_size $BATCH_SIZE \
    --learning_rate $LR \
    --period_len 24 \
    --lradj "constant" \
    --phase_layers $LAYERS \
    --latent_dim $LATENT_DIM \
    --phase_encoder_hidden $ENCODER_HIDDEN \
    --predictor_hidden $PREDICTOR_HIDDEN \
    --phase_num_routers $NUM_ROUTERS \
    --phase_attn_heads $ATT_HEADS \
    --phase_attn_dropout $ATT_DROPOUT \
    --phase_use_pos_embed 1 \
    --use_revin $USE_REVIN \
    --revin_affine $REVIN_AFFINE \
    --revin_eps $REVIN_EPS \
    --use_huber $USE_HUBER \
    --huber_delta $HUBER_DELTA \
    --extra_tag "layers${LAYERS}-latent${LATENT_DIM}-enc${ENCODER_HIDDEN}-pred${PREDICTOR_HIDDEN}-routers${NUM_ROUTERS}-heads${ATT_HEADS}"
done
