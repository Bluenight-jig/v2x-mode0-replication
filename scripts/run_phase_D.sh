#!/bin/bash
# Phase D: Mode 0c + demand separation, N in {4,10}, M=5, M_m0=2.
# Produces JSON entries with config in {D_N4, D_N10} and policy checkpoints.
# Total wall time: approximately 1 hour on RTX 3080 Laptop.
set -euo pipefail

cd "$(dirname "$0")/.."

for N in 4 10; do
    echo ""
    echo "=== Phase D: N=$N ==="
    python3 - << EOF
import re, pathlib
p = pathlib.Path("train_mode0c" + chr(46) + "py")
src = p.read_text()
src = re.sub(r'("n_vehicles":\s*)\d+,',          r'\g<1>${N},',                          src)
src = re.sub(r'("n_subchannels":\s*)\d+,',       r'\g<1>5,',                             src)
src = re.sub(r'("m0_subchannel_count":\s*)\d+,', r'\g<1>2,',                             src)
src = re.sub(r'("demand_separation":\s*)\w+,',   r'\g<1>True,',                          src)
src = re.sub(r'("config_label":\s*)"[^"]*"',     r'\g<1>"D_N${N}"',                      src)
src = re.sub(r'("n_episodes":\s*)\d+,',          r'\g<1>3000,',                          src)
src = re.sub(r'("n_eval_episodes":\s*)\d+,',     r'\g<1>100,',                           src)
src = re.sub(r'("result_file":\s*)"[^"]*"',      r'\g<1>"results/psm001_results.json"',  src)
p.write_text(src)
EOF
    python train_mode0c.py
done

echo ""
echo "Phase D complete. JSON entries: rows 13-14 in results/psm001_results.json"
echo "Policy checkpoints: checkpoints/D_N4_actors.pt, checkpoints/D_N10_actors.pt"
