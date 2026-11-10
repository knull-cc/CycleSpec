#!/bin/bash
# Exchange-rate (8 currencies, daily, strongly non-stationary). Not in neumyor/PhaseFormer:
# hyper-parameters below are ours (small model, the dataset has only ~5k training windows).
#   MODEL=PhaseFormer bash scripts/run_exchange.sh                  # base (needed: no official numbers)
#   MODEL=CycleSpec CS_BAND=nodc bash scripts/run_exchange.sh       # + CycleSpec (joint, no DC)
#   SEQ_LEN=720 ...                                                 # PhaseFormer-style long look-back
# Data: $DATA_ROOT/exchange_rate/exchange_rate.csv
#
# period_len=7 (one week of daily points): Exchange has no real cycle, this only sets the grid.
# SEQ_LEN=96 by default (the common Exchange setting; a long look-back is known to hurt here)
# -> 14 cycles of 7. The adapter windows follow the cycle count (CS_WINDOWS=auto).
# Other switches: see scripts/CycleSpec/common.sh

cd "$(dirname "$0")/.."
CS_WINDOWS=${CS_WINDOWS:-auto}
source scripts/CycleSpec/common.sh

DATA="custom"
ROOT_PATH="$DATA_ROOT/exchange_rate/"
DATA_PATH="exchange_rate.csv"
ENC_IN=8
SEQ_LEN=${SEQ_LEN:-96}
PERIOD=7
TRAIN_EPOCHS=30
PATIENCE=8
BATCH_SIZE=32
LR=0.0005
USE_HUBER=1
HUBER_DELTA=1.0
USE_REVIN=1
REVIN_AFFINE=0
REVIN_EPS=1e-5
LAYERS=1; LATENT_DIM=8; ENCODER_HIDDEN=32; PREDICTOR_HIDDEN=64; NUM_ROUTERS=4; ATT_HEADS=1

# non-default look-back gets its own name so runs never mix
DS_ID="exchange"; [ "$SEQ_LEN" != "96" ] && DS_ID="exchange_sl${SEQ_LEN}"

for PRED_LEN in ${PRED_LENS:-96 192 336 720}
do
  run_one "$DS_ID" "$PRED_LEN" \
    --data $DATA \
    --root_path "$ROOT_PATH" \
    --data_path $DATA_PATH \
    --features M \
    --target OT \
    --freq d \
    --enc_in $ENC_IN --dec_in $ENC_IN --c_out $ENC_IN \
    --seq_len $SEQ_LEN \
    --label_len 48 \
    --pred_len "$PRED_LEN" \
    --train_epochs $TRAIN_EPOCHS \
    --patience $PATIENCE \
    --batch_size $BATCH_SIZE \
    --learning_rate $LR \
    --period_len $PERIOD \
    --lradj "constant" \
    --phase_layers $LAYERS \
    --latent_dim $LATENT_DIM \
    --phase_encoder_hidden $ENCODER_HIDDEN \
    --predictor_hidden $PREDICTOR_HIDDEN \
    --phase_num_routers $NUM_ROUTERS \
    --phase_attn_heads $ATT_HEADS \
    --phase_attn_dropout 0.1 \
    --phase_use_pos_embed 1 \
    --use_revin $USE_REVIN \
    --revin_affine $REVIN_AFFINE \
    --revin_eps $REVIN_EPS \
    --use_huber $USE_HUBER \
    --huber_delta $HUBER_DELTA \
    --extra_tag "layers${LAYERS}-latent${LATENT_DIM}-enc${ENCODER_HIDDEN}-pred${PREDICTOR_HIDDEN}-routers${NUM_ROUTERS}-heads${ATT_HEADS}"
done
