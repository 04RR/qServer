#!/usr/bin/env python3
"""5-needle recall at a calibrated token target, for the q8_0-KV retrieval gate (round 2, arm 3).
Five FRESH sentinels (none in this repo's history -> no prompt-cache/history false pass) are placed
at spread depths (~8/29/50/71/92%) in telemetry filler, then the model is asked to list every code.
Pass = codes recalled >= PASS (default 4 of 5). Self-calibrates like needle-next.py / needle35.py.

Usage: MODEL=flash-next NCTX=131072 python3 needle5.py http://127.0.0.1:8095 100000 [PASS]
Exit 0 iff recalled >= PASS."""
import sys, os, json, urllib.request, urllib.error

B      = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8095"
TARGET = int(sys.argv[2]) if len(sys.argv) > 2 else 100000
PASS   = int(sys.argv[3]) if len(sys.argv) > 3 else 4
MODEL  = os.environ.get("MODEL", "flash-next")
NCTX   = int(os.environ.get("NCTX", "131072"))
SYS    = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."

# label -> fresh sentinel (not in repo history), and the fractional depth to place it
NEEDLES = [("alpha",   "AMBER-VECTOR-19",     0.08),
           ("bravo",   "CRIMSON-LATTICE-58",  0.29),
           ("charlie", "ZEPHYR-GRANITE-30",   0.50),
           ("delta",   "VELVET-PULSAR-73",    0.71),
           ("echo",    "INDIGO-MARSHAL-64",   0.92)]
CODES = [c for _, c, _ in NEEDLES]

def post(path, obj, timeout=1200):
    r = urllib.request.Request(B + path, data=json.dumps(obj).encode(),
                               headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(r, timeout=timeout))
def ntok(s):
    return len(post("/tokenize", {"content": s}, 180)["tokens"])

def sentences(n):
    return [f"Log entry {i}: routine telemetry sample, nominal, no action required." for i in range(1, n)]

def build(n):
    s = sentences(n)
    for label, code, depth in NEEDLES:
        idx = min(len(s) - 1, max(0, int(len(s) * depth)))
        s[idx] = f"The {label} access code is {code}."
    body = " ".join(s)
    return (body + "\n\nQuestion: there are five access codes hidden above (alpha, bravo, charlie, "
            "delta, echo). List all five codes, one per line, in the form LABEL: CODE.")

probe = 500
rate = ntok(build(probe)) / probe
n = int(TARGET / rate)
for _ in range(4):
    t = ntok(build(n))
    if abs(t - TARGET) <= max(2000, int(0.02 * TARGET)):
        break
    n = int(n * TARGET / max(t, 1))
prompt = build(n); t = ntok(prompt)
print(f"  calibrated: {n} sentences -> {t} tokens (target {TARGET}); needles={CODES}")
if t > NCTX - 2048:
    print(f"  ABORT: {t} would exceed n_ctx={NCTX}"); sys.exit(2)

try:
    d = post("/v1/chat/completions", {"model": MODEL,
        "messages": [{"role": "system", "content": SYS}, {"role": "user", "content": prompt}],
        "max_tokens": 2048, "temperature": 0.0, "stream": False})
except urllib.error.HTTPError as e:
    print(f"  HTTP {e.code}: {e.read().decode()[:400]}"); sys.exit(2)

c = d["choices"][0]["message"].get("content", "") or ""
tm = d.get("timings", {}); u = d.get("usage", {})
found = [code for code in CODES if code in c]
missing = [code for code in CODES if code not in c]
print(f"  prompt_tokens={u.get('prompt_tokens')} pp={tm.get('prompt_per_second',0):.0f} "
      f"tg={tm.get('predicted_per_second',0):.1f}")
print(f"  recalled {len(found)}/5  found={found}  missing={missing}")
print(f"  answer_tail={c.strip()[-200:]!r}")
sys.exit(0 if len(found) >= PASS else 1)
