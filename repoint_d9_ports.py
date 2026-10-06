import re, glob, subprocess, time, os, difflib
BIN = os.path.expanduser("~/v2x_clean/ns3_bridge/v2x_bridge")
MOVE = {"train_d9_A_N15_seed1_5000ep_gaefix_g05_ent01_as.py": 8756,     # ports seen busy during option C's generation
        "train_d9_A_N15_seed3_5000ep_gaefix_g05_ent01_as.py": 8776,
        "train_d9_A_N7_seed3_5000ep_gaefix_g05_ent01_as.py":  8746}
SPARE = [8786, 8796, 8806, 8816, 8826, 8836]                             # free in option C's generation, unused now
PORT = r'("ns3_port":\s*)(\d+),'
def port_ok(p):
    proc = subprocess.Popen([BIN, f"--port={p}", "--fcGHz=5.9"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(2)
    alive = proc.poll() is None
    if alive: proc.terminate(); proc.wait(timeout=5)
    return alive
def main(check=port_ok):
    used = {int(re.search(PORT, open(f).read()).group(2)) for f in glob.glob("train_d9_*.py")}
    pool = iter(p for p in SPARE if p not in used)
    for f, old in MOVE.items():
        src = open(f).read()
        cur = int(re.search(PORT, src).group(2))
        if cur != old:
            print(f"  SKIP {f}: port is {cur}, not {old} -- left unchanged"); continue
        new = next((p for p in pool if check(p)), None)
        if new is None:
            print("  FAIL: no spare port is free -- send me this output"); return
        s, n = re.subn(PORT, rf'\g<1>{new},', src)
        diff = [l for l in difflib.unified_diff(src.splitlines(), s.splitlines(), lineterm="", n=0)
                if l[:1] in "+-" and l[:3] not in ("+++", "---")]
        if n != 1 or len(diff) != 2:
            print(f"  FAIL {f}: {len(diff)//2} lines would change -- left unchanged"); continue
        open(f, "w").write(s)
        print(f"  OK   {f}: port {old} -> {new}  (1 line changed)")
    ports = sorted(int(re.search(PORT, open(f).read()).group(2)) for f in glob.glob("train_d9_*.py"))
    print(f"\n{len(ports)} runners, {len(set(ports))} distinct ports: {'all distinct' if len(ports) == len(set(ports)) else 'DUPLICATE -- send me this output'}")
if __name__ == "__main__":
    main()
