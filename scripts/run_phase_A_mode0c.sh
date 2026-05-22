#!/bin/bash
# Phase A Mode 0c architectural comparison at N=4, shared pool (no demand separation).
# Produces JSON entry with config="A_mode0c".
# Wall time: approximately 30 minutes on RTX 3080 Laptop.
set -euo pipefail

cd "$(dirname "$0")/.."

python3 - << 'EOF'
import re, pathlib
p = pathlib.Path("train_mode0c" + chr(46) + "py")
src = p.read_text()
src = re.sub(r'("n_vehicles":\s*)\d+,',        r'\g<1>4,',                               src)
src = re.sub(r'("n_subchannels":\s*)\d+,',     r'\g<1>5,',                               src)
src = re.sub(r'("demand_separation":\s*)\w+,', r'\g<1>False,',                           src)
src = re.sub(r'("config_label":\s*)"[^"]*"',   r'\g<1>"A_mode0c"',                       src)
src = re.sub(r'("n_episodes":\s*)\d+,',        r'\g<1>3000,',                            src)
src = re.sub(r'("n_eval_episodes":\s*)\d+,',   r'\g<1>100,',                             src)
src = re.sub(r'("result_file":\s*)"[^"]*"',    r'\g<1>"results/psm001_results.json"',    src)
p.write_text(src)
EOF
python train_mode0c.py

echo ""
echo "Phase A-0c complete. JSON entry: row 6 in results/psm001_results.json"
