#!/bin/bash
# Phase B: Supply-axis sweep, M in {3,7,10} at N=4, shared pool.
# Produces JSON entries with config in {B_M3, B_M7, B_M10}.
# Total wall time: approximately 1.5 hours on RTX 3080 Laptop.
set -euo pipefail

cd "$(dirname "$0")/.."

for M in 3 7 10; do
    echo ""
    echo "=== Phase B: M=$M ==="
    python3 - << EOF
import re, pathlib
p = pathlib.Path("train" + chr(46) + "py")
src = p.read_text()
src = re.sub(r'("n_vehicles":\s*)\d+,',        r'\g<1>4,',                               src)
src = re.sub(r'("n_subchannels":\s*)\d+,',     r'\g<1>${M},',                            src)
src = re.sub(r'("demand_separation":\s*)\w+,', r'\g<1>False,',                           src)
src = re.sub(r'("config_label":\s*)"[^"]*"',   r'\g<1>"B_M${M}"',                        src)
src = re.sub(r'("n_episodes":\s*)\d+,',        r'\g<1>3000,',                            src)
src = re.sub(r'("n_eval_episodes":\s*)\d+,',   r'\g<1>100,',                             src)
src = re.sub(r'("result_file":\s*)"[^"]*"',    r'\g<1>"results/psm001_results.json"',    src)
p.write_text(src)
EOF
    python train.py
done

echo ""
echo "Phase B complete. JSON entries: rows 7-9 in results/psm001_results.json"
