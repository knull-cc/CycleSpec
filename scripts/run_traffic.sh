#!/bin/bash
# Traffic. Hyper-parameters exactly as in the upstream PhaseFormer_TSL run_traffic.sh
# (MSE loss, patience left at run.py's default of 3); only the model switch was added.
#   MODEL=PhaseFormer bash scripts/run_traffic.sh                  # base
#   MODEL=CycleSpec   bash scripts/run_traffic.sh                  # + CycleSpec (joint)
#   MODEL=CycleSpec CS_MODE=frozen bash scripts/run_traffic.sh     # adapter on the frozen base
#   PRED_LENS="96" MODEL=CycleSpec bash scripts/run_traffic.sh     # only H=96
# Other switches: see scripts/CycleSpec/common.sh
# Note: neumyor/PhaseFormer run_traffic.py uses Huber(delta=1) and patience 8 instead.

cd "$(dirname "$0")/.."
source scripts/CycleSpec/common.sh

# 基础配置
DATA="custom"
ROOT_PATH="$DATA_ROOT/traffic/"
DATA_PATH="traffic.csv"
ENC_IN=862
SEQ_LEN=720
TRAIN_EPOCHS=30
PATIENCE=3
BATCH_SIZE=16
USE_HUBER=0
HUBER_DELTA=1.0
USE_REVIN=1
REVIN_AFFINE=0
REVIN_EPS=1e-5

# 四种预测长度
for PRED_LEN in ${PRED_LENS:-96 192 336 720}
do
  if [ "$PRED_LEN" -eq 96 ]; then
    LR=0.001
    LAYERS=2
    LATENT_DIM=32
    ENCODER_HIDDEN=64
    PREDICTOR_HIDDEN=128
    NUM_ROUTERS=1
    ATT_HEADS=8
  else
    LR=0.001
    LAYERS=1
    LATENT_DIM=128
    ENCODER_HIDDEN=16
    PREDICTOR_HIDDEN=32
    NUM_ROUTERS=4
    ATT_HEADS=4
  fi

  run_one traffic "$PRED_LEN" \
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
    --use_revin $USE_REVIN \
    --revin_affine $REVIN_AFFINE \
    --revin_eps $REVIN_EPS \
    --use_huber $USE_HUBER \
    --huber_delta $HUBER_DELTA \
    --extra_tag "layers${LAYERS}-latent${LATENT_DIM}-enc${ENCODER_HIDDEN}-pred${PREDICTOR_HIDDEN}-routers${NUM_ROUTERS}-heads${ATT_HEADS}"
done
