#!/bin/bash
# Phase C: Mode 0a + demand separation, N in {4,7,10}, M=5, M_m0=2.
# Produces JSON entries with config in {C_N4, C_N7, C_N10}.
# Total wall time: approximately 1.5 hours on RTX 3080 Laptop.
set -euo pipefail

cd "$(dirname "$0")/.."

for N in 4 7 10; do
    echo ""
    echo "=== Phase C: N=$N ==="
    python3 - << EOF
import re, pathlib
p = pathlib.Path("train" + chr(46) + "py")
src = p.read_text()
src = re.sub(r'("n_vehicles":\s*)\d+,',          r'\g<1>${N},',                          src)
src = re.sub(r'("n_subchannels":\s*)\d+,',       r'\g<1>5,',                             src)
src = re.sub(r'("m0_subchannel_count":\s*)\d+,', r'\g<1>2,',                             src)
src = re.sub(r'("demand_separation":\s*)\w+,',   r'\g<1>True,',                          src)
src = re.sub(r'("config_label":\s*)"[^"]*"',     r'\g<1>"C_N${N}"',                      src)
src = re.sub(r'("n_episodes":\s*)\d+,',          r'\g<1>3000,',                          src)
src = re.sub(r'("n_eval_episodes":\s*)\d+,',     r'\g<1>100,',                           src)
src = re.sub(r'("result_file":\s*)"[^"]*"',      r'\g<1>"results/psm001_results.json"',  src)
p.write_text(src)
EOF
    python train.py
done

echo ""
echo "Phase C complete. JSON entries: rows 10-12 in results/psm001_results.json"
