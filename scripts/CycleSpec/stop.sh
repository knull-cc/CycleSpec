#!/bin/bash
# Stop everything started by run_all.sh / compare.sh / run_*.sh.
# Launchers are stopped first so they cannot start the next run.
# Finished runs keep their results/; unfinished ones are rerun from scratch next time.
pkill -f "^bash scripts/CycleSpec/run_all.sh"
pkill -f "^bash scripts/CycleSpec/compare.sh"
pkill -f "^bash scripts/run_"
pkill -f "^bash scripts/VarHA/run_ett.sh"
sleep 1
pkill -f "run.py --task_name"
sleep 3
n=$(pgrep -fc "run.py --task_name")
echo "remaining run.py processes: $n"
[ "$n" -gt 0 ] && { pkill -9 -f "run.py --task_name"; sleep 1; echo "after kill -9: $(pgrep -fc 'run.py --task_name')"; }
exit 0
