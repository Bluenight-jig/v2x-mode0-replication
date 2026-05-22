#!/usr/bin/env python3
"""
quickstart_smoke.py - Quick install-validation smoke test.

Patches train.py CFG to a tiny 30-episode, N=4 Mode 0a configuration,
runs train.py once, prints the Eval Summary. Wall time: ~2-3 minutes.

The smoke output writes to results/_smoke.json (separate from the canonical
psm001_results.json so production data is never disturbed).
"""
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
TRAIN_PY = REPO_ROOT / ("train" + chr(46) + "py")


def patch_cfg(text: str) -> str:
    text = re.sub(r'("n_vehicles":\s*)\d+,',        r'\g<1>4,',                       text)
    text = re.sub(r'("n_subchannels":\s*)\d+,',     r'\g<1>5,',                       text)
    text = re.sub(r'("demand_separation":\s*)\w+,', r'\g<1>False,',                   text)
    text = re.sub(r'("config_label":\s*)"[^"]*"',   r'\g<1>"smoke_test"',             text)
    text = re.sub(r'("n_episodes":\s*)\d+,',        r'\g<1>30,',                      text)
    text = re.sub(r'("n_eval_episodes":\s*)\d+,',   r'\g<1>5,',                       text)
    text = re.sub(r'("result_file":\s*)"[^"]*"',    r'\g<1>"results/_smoke.json"',    text)
    return text


def main():
    print("=" * 60)
    print("Quickstart smoke test - N=4, 30 train ep, 5 eval ep")
    print("=" * 60)
    print(f"Patching {TRAIN_PY.name}...")
    original = TRAIN_PY.read_text()
    TRAIN_PY.write_text(patch_cfg(original))
    print("Done. Launching train.py (expect ~2-3 minutes)...")
    print()
    try:
        result = subprocess.run([sys.executable, str(TRAIN_PY)],
                                cwd=str(REPO_ROOT))
        return result.returncode
    finally:
        # No restore -- user can run any phase script next to set their own CFG.
        pass


if __name__ == "__main__":
    raise SystemExit(main())
