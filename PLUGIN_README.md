# EMA + BDO plug-in for PhaseFormer

## What was added
| file | what |
|---|---|
| `utils/ema.py` | `ModelEMA`: weight EMA used for validation / early stopping / test. Zero inference cost. |
| `utils/bdo_utils.py` | stage prefix lengths and stage loss weights |
| `models/PhaseFormerBDO.py` | PhaseFormer with a cascaded (boosted direct output) head in phase space. `--bdo_stages 1` == PhaseFormer |
| `exp/exp_long_term_forecasting.py` | `_train_loss` (multi-stage loss), EMA hooks, EMA weights saved as the best checkpoint |
| `exp/exp_basic.py`, `run.py` | register model; new args `--fix_seed --ema_decay --ema_warmup --bdo_stages --bdo_w_mode` |
| `tools/selftest_plugin.py` | sanity checks, run BEFORE training |
| `tools/seed_ensemble.py` | seed-ensemble ceiling for any variance-reduction trick |
| `tools/summarize_arms.py` | per-arm mean+-std and paired difference |
| `scripts/plugin/run_etth1.sh` | arms A/B/C/D(/E/F) x seeds |

## Run (repo root)
    python tools/selftest_plugin.py                       # every line must say PASS
    ARMS="A" bash scripts/plugin/run_etth1.sh             # 1) baseline x5 seeds -> ensemble ceiling
    ARMS="A B C D" bash scripts/plugin/run_etth1.sh       # 2) full comparison
    PRED_LEN=192 ARMS="B D" bash scripts/plugin/run_etth1.sh

Compare D vs B (plugin+EMA vs baseline+EMA) and C vs A. A gain below the seed std is not evidence.
Choose `EMA_DECAY` on validation loss only.

## Not verified
Written without torch available: only the pure-python / numpy parts were executed.
