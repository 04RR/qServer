#!/usr/bin/env python3
"""100K needle recall, self-calibrating to the target token count.
The 27B's hardcoded 10500-sentence filler tokenizes to ~188K on the 35B (~18 tok/sentence vs ~10),
which 400s against n_ctx=131072. Calibrate against /tokenize instead of assuming a constant."""
import sys, os, json, urllib.request, urllib.error

B = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8082"
MODEL = os.environ.get("MODEL", "qwen36-35b")   # model-agnostic: set MODEL for the 122B etc.
TARGET = int(sys.argv[2]) if len(sys.argv) > 2 else 100000
SYS = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
NEEDLE = "The vault access code is MERIDIAN-COBALT-7."

def post(path, obj, timeout=900):
    r = urllib.request.Request(B + path, data=json.dumps(obj).encode(),
                               headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(r, timeout=timeout))

def ntok(s):
    return len(post("/tokenize", {"content": s}, 180)["tokens"])

def build(n):
    f = " ".join(f"Log entry {i}: routine telemetry sample, nominal, no action required."
                 for i in range(1, n))
    h = len(f) // 2
    return (f[:h] + "\n\n" + NEEDLE + "\n\n" + f[h:] +
            "\n\nQuestion: what is the vault access code? Answer with just the code.")

# calibrate: measure tok/sentence on a small sample, then scale
probe = 500
rate = ntok(build(probe)) / probe
n = int(TARGET / rate)
for _ in range(4):
    t = ntok(build(n))
    if abs(t - TARGET) <= 2000:
        break
    n = int(n * TARGET / t)
prompt = build(n)
t = ntok(prompt)
print(f"  calibrated: {n} sentences -> {t} tokens (rate {rate:.1f} tok/sentence, target {TARGET})")
if t > 131072 - 2048:
    print(f"  ABORT: {t} would exceed n_ctx"); sys.exit(1)

try:
    d = post("/v1/chat/completions", {
        "model": MODEL,
        "messages": [{"role": "system", "content": SYS}, {"role": "user", "content": prompt}],
        "max_tokens": 2048, "temperature": 0.0, "stream": False})
except urllib.error.HTTPError as e:
    print(f"  HTTP {e.code}: {e.read().decode()[:400]}"); sys.exit(1)

c = d["choices"][0]["message"].get("content", "") or ""
u = d.get("usage", {})
tm = d.get("timings", {})
print(f"  prompt_tokens={u.get('prompt_tokens')} pp={tm.get('prompt_per_second',0):.0f} t/s "
      f"tg={tm.get('predicted_per_second',0):.1f} t/s")
print(f"  answer={c.strip()[:90]!r}")
sys.exit(0 if "MERIDIAN-COBALT-7" in c else 1)
