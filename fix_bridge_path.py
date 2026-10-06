#!/usr/bin/env python3
"""Replace the hardcoded ~/v2x_thesis bridge path with a tree-relative one."""
import pathlib, sys

ROOT = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else ".").resolve()
EDITS = [
    ("envs/ns3_bridge.py",
     "import os, time, signal, json, subprocess",
     "import os, time, signal, json, subprocess, pathlib"),
    ("envs/ns3_bridge.py",
     '                 binary:     str   = os.path.expanduser(\n'
     '                                 "~/v2x_thesis/ns3_bridge/v2x_bridge"),',
     '                 binary:     str   = str(pathlib.Path(__file__).resolve().parent.parent\n'
     '                                         / "ns3_bridge" / "v2x_bridge"),'),
    ("envs/v2x_env.py",
     "import os, sys, io",
     "import os, sys, io, pathlib"),
    ("envs/v2x_env.py",
     '            binary=os.path.expanduser(\n'
     '                cfg.get("bridge_binary","~/v2x_thesis/ns3_bridge/v2x_bridge")),',
     '            binary=os.path.expanduser(cfg.get(\n'
     '                "bridge_binary",\n'
     '                str(pathlib.Path(__file__).resolve().parent.parent\n'
     '                    / "ns3_bridge" / "v2x_bridge"))),'),
    ("train.py",
     '    "bridge_binary": os.path.expanduser("~/v2x_thesis/ns3_bridge/v2x_bridge"),',
     '    "bridge_binary": str(pathlib.Path(__file__).resolve().parent / "ns3_bridge" / "v2x_bridge"),'),
    ("train_mode0c.py",
     '    "bridge_binary": os.path.expanduser("~/v2x_thesis/ns3_bridge/v2x_bridge"),',
     '    "bridge_binary": str(pathlib.Path(__file__).resolve().parent / "ns3_bridge" / "v2x_bridge"),'),
]

applied = skipped = failed = 0
for rel, old, new in EDITS:
    p = ROOT / rel
    if not p.is_file():
        print(f"  [MISSING ] {rel}"); failed += 1; continue
    src = p.read_text()
    if new in src:
        print(f"  [ALREADY ] {rel}"); skipped += 1; continue
    if src.count(old) != 1:
        print(f"  [NO MATCH] {rel}: found {src.count(old)}x"); failed += 1; continue
    p.write_text(src.replace(old, new, 1))
    print(f"  [PATCHED ] {rel}"); applied += 1

print(f"\napplied={applied} already={skipped} failed={failed}")
leftover = [r for r, _, _ in EDITS if (ROOT/r).is_file() and "v2x_thesis" in (ROOT/r).read_text()]
print("remaining hardcoded paths:", leftover if leftover else "none")
sys.exit(1 if failed else 0)
