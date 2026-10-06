#!/usr/bin/env bash
# rerun_t2.sh -- batch 27: re-runs the run that failed at startup in Terminal 2, once Terminal 2's chain
# has finished. Handles both likely causes of the failure (aa-achievement.md, technical notes 9 and 19):
#   an orphaned bridge of ours still holding the port -> stopped by its own PID (never pkill);
#   a port held by something else (e.g. Windows, invisible to ss) -> runner moved to a free spare port.
set -u
cd "${V2X_DIR:-$HOME/v2x_clean}"
FAILED=A_mode0c_N10_seed3_gaefix_g05_ent01_armS        # failed at startup, port 8656
LAST=A_mode0c_seed3_5000ep_gaefix_g05_ent01_armS       # last run in Terminal 2's chain
BIN="${V2X_BRIDGE:-$HOME/v2x_clean/ns3_bridge/v2x_bridge}"
SPARE="8826 8836 8846 8856 8866 8876"
POLL="${POLL:-300}"
stamp() { date '+%a %d %b %H:%M'; }
port_of() { grep -oP '"ns3_port":\s*\K\d+' "train_d9_$1.py"; }
stop_orphan() {   # stop a v2x_bridge listening on port $1 -- ours, by its own PID only
    local pid; pid=$(ss -lntpH "sport = :$1" 2>/dev/null | grep -oP 'pid=\K\d+' | head -1)
    if [ -n "$pid" ] && [ "$(ps -p "$pid" -o comm= 2>/dev/null)" = "v2x_bridge" ]; then
        kill "$pid"; sleep 2; echo "  stopped an orphaned bridge (PID $pid) on port $1"
    fi
}
last_running() {  # Terminal 2's last run still alive? count only PYTHON processes running that file
    local pid
    for pid in $(pgrep -f "train_d9_$LAST\.py"); do
        case "$(ps -p "$pid" -o comm= 2>/dev/null)" in python*) return 0 ;; esac
    done
    return 1
}
port_free() {     # start a bridge on port $1: free only if it is still alive after 2 s
    "$BIN" --port="$1" --fcGHz=5.9 >/dev/null 2>&1 & local pid=$!
    sleep 2
    if kill -0 "$pid" 2>/dev/null; then kill "$pid"; wait "$pid" 2>/dev/null; return 0; fi
    wait "$pid" 2>/dev/null; return 1
}
echo "[$(stamp)] rerun_t2: $FAILED"
if [ -f "logs/$FAILED.log" ] && [ ! -f "logs/$FAILED.crash1.log" ]; then
    mv "logs/$FAILED.log" "logs/$FAILED.crash1.log"; echo "  crash log kept as logs/$FAILED.crash1.log"
fi
p=$(port_of "$FAILED"); stop_orphan "$p"
echo "  waiting for Terminal 2's chain to finish (checking every $((POLL / 60)) min) ..."
while [ ! -f "logs/$LAST.log" ] || last_running; do sleep "$POLL"; done
echo "[$(stamp)] Terminal 2's chain has finished"
p=$(port_of "$FAILED"); stop_orphan "$p"
if port_free "$p"; then
    echo "  port $p is free"
else
    used=$(grep -ohP '"ns3_port":\s*\K\d+' train_d9_*.py | sort -u)
    new=""
    for q in $SPARE; do
        echo "$used" | grep -qx "$q" && continue
        if port_free "$q"; then new=$q; break; fi
    done
    [ -z "$new" ] && { echo "  FAIL: port $p busy and no spare port free -- not run; send me this output"; exit 1; }
    cp "train_d9_$FAILED.py" "/tmp/train_d9_$FAILED.py.bak"
    sed -i -E "s/(\"ns3_port\":[[:space:]]*)$p,/\1$new,/" "train_d9_$FAILED.py"
    n=$(diff "/tmp/train_d9_$FAILED.py.bak" "train_d9_$FAILED.py" | grep -c '^[<>]')
    [ "$n" -ne 2 ] && { cp "/tmp/train_d9_$FAILED.py.bak" "train_d9_$FAILED.py"; echo "  FAIL: port edit changed $((n / 2)) lines -- restored, not run"; exit 1; }
    echo "  port $p busy (not a bridge of ours) -- runner moved to port $new (1 line changed)"
fi
echo "[$(stamp)] starting $FAILED"
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -u "train_d9_$FAILED.py" 2>&1 | tee "logs/$FAILED.log"
