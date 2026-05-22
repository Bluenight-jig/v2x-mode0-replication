#!/bin/bash
# Phase A: Mode 0a baseline sweep, N in {2,3,4,5,7,10}, M=5, shared pool
# Produces JSON entries with config="A" (distinguished by N field).
# Total wall time: approximately 3 hours on RTX 3080 Laptop.
set -euo pipefail

cd "$(dirname "$0")/.."

for N in 2 3 4 5 7 10; do
    echo ""
    echo "=== Phase A: N=$N ==="
    python3 - << EOF
import re, pathlib
p = pathlib.Path("train" + chr(46) + "py")
src = p.read_text()
src = re.sub(r'("n_vehicles":\s*)\d+,',        r'\g<1>${N},',                            src)
src = re.sub(r'("n_subchannels":\s*)\d+,',     r'\g<1>5,',                               src)
src = re.sub(r'("demand_separation":\s*)\w+,', r'\g<1>False,',                           src)
src = re.sub(r'("config_label":\s*)"[^"]*"',   r'\g<1>"A"',                              src)
src = re.sub(r'("n_episodes":\s*)\d+,',        r'\g<1>3000,',                            src)
src = re.sub(r'("n_eval_episodes":\s*)\d+,',   r'\g<1>100,',                             src)
src = re.sub(r'("result_file":\s*)"[^"]*"',    r'\g<1>"results/psm001_results.json"',    src)
p.write_text(src)
EOF
    python train.py
done

echo ""
echo "Phase A complete. JSON entries: rows 0-5 in results/psm001_results.json"
