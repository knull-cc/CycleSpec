#!/bin/bash
# Progress of run_all.sh / compare.sh:
#   bash scripts/CycleSpec/status.sh                                   # official-config runs
#   BS=256 DATASETS="ettm1 ettm2" bash scripts/CycleSpec/status.sh     # runs tagged bs256
#   CS_PORT=block DATASETS=etth1 SEED=2022 bash scripts/CycleSpec/status.sh   # another CycleSpec variant / seed
# Cell = test MSE/MAE when finished, "epN" while training (N finished epochs), "-" not started.

cd "$(dirname "$0")/../.."
SEED=${SEED:-2021}
DATASETS=${DATASETS:-"etth1 etth2 ettm1 ettm2 weather electricity traffic"}
PRED_LENS=${PRED_LENS:-"96 192 336 720"}
RUN_TAG=${RUN_TAG:-${BS:+bs$BS}}
T=${RUN_TAG:+_$RUN_TAG}
CS_PORT=${CS_PORT:-input}
CS_ARCH=${CS_ARCH:-spectral}
case "$CS_PORT" in input) _P="" ;; block) _P="blk" ;; both) _P="both" ;; output) _P="out" ;; io) _P="io" ;; *) _P="?" ;; esac
[ "$CS_ARCH" = "dense" ] && _A="dense" || _A=""
CS_BAND=${CS_BAND:-all}            # all | nodc | mid
[ "$CS_BAND" = "all" ] && _B="" || _B="$CS_BAND"
if [ -n "$_P$_A$_B" ]; then V="cs${_P:-in}${_A}${_B}"; else V=""; fi
VS=${V:+_$V}     # suffix in model_id
VL=${V:+-$V}     # suffix in log names

cell() {  # ds H variant
  local ds=$1 H=$2 v=$3 mid model log js
  case $v in
    PhaseFormer)      mid=${ds}_pl${H}${T};        model=PhaseFormer ;;
    CycleSpec)        mid=${ds}_pl${H}${T}${VS};        model=CycleSpec; v=CycleSpec${VL} ;;
    CycleSpec-frozen) mid=${ds}_pl${H}${T}${VS}_frozen; model=CycleSpec; v=CycleSpec${VL}-frozen ;;
  esac
  js=$(ls results/long_term_forecast_${mid}_${model}_*_s${SEED}_0/summary.json 2>/dev/null | head -n 1)
  log=logs/$ds/${v}_pl${H}${T}_s${SEED}.log
  if [ -n "$js" ]; then
    awk '{ if (match($0, /"mse": [0-9.eE+-]+/)) m = substr($0, RSTART + 7, RLENGTH - 7);
           if (match($0, /"mae": [0-9.eE+-]+/)) a = substr($0, RSTART + 7, RLENGTH - 7) }
         END { printf "%.4f/%.4f", m, a }' "$js"
  elif [ -f "$log" ]; then
    if grep -aq "Traceback\|out of memory\|Error" "$log"; then printf "ERROR"
    else printf "ep%s" "$(grep -ac 'Epoch: [0-9]*, Steps' "$log")"; fi
  else
    printf "-"
  fi
}

echo "tag: ${RUN_TAG:-none (official config)}   seed: $SEED   CycleSpec port: $CS_PORT  arch: $CS_ARCH   cells: MSE/MAE"
printf "%-12s %-5s %-14s %-14s %-17s %s\n" dataset H PhaseFormer CycleSpec CycleSpec-frozen "dMSE(joint/frozen)"
for ds in $DATASETS; do
  for H in $PRED_LENS; do
    b=$(cell "$ds" "$H" PhaseFormer); j=$(cell "$ds" "$H" CycleSpec); f=$(cell "$ds" "$H" CycleSpec-frozen)
    d=$(awk -v b="${b%%/*}" -v j="${j%%/*}" -v f="${f%%/*}" \
          'function p(x){ return (x ~ /^[0-9.]+$/ && b ~ /^[0-9.]+$/) ? sprintf("%+.2f%%", (x-b)/b*100) : "." }
           BEGIN{ print p(j) " / " p(f) }')
    printf "%-12s %-5s %-14s %-14s %-17s %s\n" "$ds" "$H" "$b" "$j" "$f" "$d"
  done
done
echo
echo "running run.py processes: $(pgrep -fc "run.py --task_name")"
grep -l "Traceback\|out of memory" logs/*/*.log logs/compare_*.log 2>/dev/null | sed 's/^/error in: /'
command -v nvidia-smi > /dev/null && nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader
free -g | awk 'NR==2{print "RAM used " $3 "G / " $2 "G"}'
