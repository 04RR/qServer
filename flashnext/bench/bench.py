#!/usr/bin/env python3
"""Flash-Next upgrade bench — one arm's measurements against an ALREADY-RUNNING server (:8095).
arm.sh launches the server + waits /health, then runs this, then SIGTERMs the server.
Measures: cold TTFT, D-short (3x median) + MTP accept + pgmajfault delta, P-8K, P-100K + D-100K,
VRAM per GPU after load and peak after 100K, MemAvailable. Writes raw JSON to $OUT.
Every block is isolated so one failure doesn't lose the rest."""
import sys, os, json, time, subprocess, urllib.request, statistics

B      = os.environ.get("BASE", "http://127.0.0.1:8095")
ARM    = os.environ.get("ARM", "arm")
OUT    = os.environ.get("OUT", f"/home/rohit/ai/qwen36/flashnext/bench/{ARM}.json")
P8K    = int(os.environ.get("P8K", "8000"))
P100K  = int(os.environ.get("P100K", "100000"))
NREP   = int(os.environ.get("DSHORT_REP", "3"))
CODE   = "Write a Python function that merges two sorted lists.\n\ndef merge("
SAMP   = {"temperature": 1.0, "top_p": 0.95, "top_k": 20}

def _req(path, obj, timeout):
    return urllib.request.Request(B + path, data=json.dumps(obj).encode(),
                                  headers={"Content-Type": "application/json"})
def post(path, obj, timeout=1800):
    with urllib.request.urlopen(_req(path, obj, timeout), timeout=timeout) as r:
        return json.load(r)
def ntok(s):
    return len(post("/tokenize", {"content": s}, 300)["tokens"])
def gpu_used():
    out = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used",
                                   "--format=csv,noheader,nounits"]).decode().split()
    return [int(x) for x in out if x.strip().lstrip("-").isdigit()]
def mem_avail_mib():
    for ln in open("/proc/meminfo"):
        if ln.startswith("MemAvailable"):
            return int(ln.split()[1]) // 1024
def pgmajfault():
    for ln in open("/proc/vmstat"):
        if ln.startswith("pgmajfault "):
            return int(ln.split()[1])

res = {"arm": ARM, "base": B, "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
       "targets": {"P8K": P8K, "P100K": P100K, "dshort_rep": NREP}}

# ---- VRAM + mem right after load (experts still lazy) ----
try:
    res["vram_after_load_mib"] = gpu_used()
    res["memavail_after_load_mib"] = mem_avail_mib()
except Exception as e:
    res["load_state_err"] = repr(e)

# ---- cold TTFT: first gen request, streaming; also faults experts for warm runs ----
try:
    body = dict(prompt=CODE, n_predict=256, cache_prompt=False, stream=True, **SAMP)
    t0 = time.time(); ttft = None; last = {}
    with urllib.request.urlopen(_req("/completion", body, 1200), timeout=1200) as r:
        for raw in r:
            s = raw.decode("utf-8", "ignore").strip()
            if not s.startswith("data:"):
                continue
            if ttft is None:
                ttft = time.time() - t0
            try:
                last = json.loads(s[5:].strip())
            except Exception:
                pass
    res["cold_ttft_s"] = round(ttft, 2) if ttft else None
    res["cold_total_s"] = round(time.time() - t0, 2)
    if isinstance(last, dict) and last.get("timings"):
        res["cold_decode_tps"] = round(last["timings"].get("predicted_per_second", 0), 2)
except Exception as e:
    res["cold_ttft_err"] = repr(e)

# ---- D-short: NREP repeats (warm), median predicted_per_second; MTP accept; pgmajfault delta ----
try:
    pf0 = pgmajfault()
    tps, accs, dn_tot, da_tot = [], [], 0, 0
    dshort = []
    for _ in range(NREP):
        d = post("/completion", dict(prompt=CODE, n_predict=256, cache_prompt=False, **SAMP), 600)
        tm = d.get("timings", {})
        tps.append(tm.get("predicted_per_second", 0))
        dn, da = tm.get("draft_n", 0) or 0, tm.get("draft_n_accepted", 0) or 0
        dn_tot += dn; da_tot += da
        dshort.append({"tps": round(tm.get("predicted_per_second", 0), 2),
                       "draft_n": dn, "draft_n_accepted": da,
                       "predicted_n": tm.get("predicted_n")})
    pf1 = pgmajfault()
    res["dshort_tps_median"] = round(statistics.median(tps), 2)
    res["dshort_tps_all"] = [round(x, 2) for x in tps]
    res["dshort_runs"] = dshort
    res["mtp_acceptance"] = round(da_tot / dn_tot, 3) if dn_tot else None
    res["dshort_pgmajfault_delta"] = pf1 - pf0
except Exception as e:
    res["dshort_err"] = repr(e)

# ---- calibrate a prompt to a token target (filler + /tokenize), like needle-next.py ----
def build(n):
    return " ".join(f"Log entry {i}: routine telemetry sample, nominal, no action required."
                    for i in range(1, n))
def calibrate(target):
    rate = ntok(build(500)) / 500.0
    n = int(target / rate)
    for _ in range(4):
        t = ntok(build(n))
        if abs(t - target) <= max(500, int(0.02 * target)):
            break
        n = int(n * target / max(t, 1))
    p = build(n)
    return p, ntok(p)

# ---- P-8K prefill ----
try:
    p8, n8 = calibrate(P8K)
    d = post("/completion", dict(prompt=p8, n_predict=1, cache_prompt=False, temperature=0.0), 600)
    res["P8K"] = {"prompt_n": d.get("timings", {}).get("prompt_n"),
                  "prefill_tps": round(d.get("timings", {}).get("prompt_per_second", 0), 2)}
except Exception as e:
    res["P8K_err"] = repr(e)

# ---- P-100K prefill + D-100K decode (one request yields both) + peak VRAM ----
if os.environ.get("FAST", "0") == "1":
    res["P100K"] = "skipped (FAST=1)"
else:
  try:
    p100, n100 = calibrate(P100K)
    t0 = time.time()
    d = post("/completion", dict(prompt=p100, n_predict=128, cache_prompt=False, **SAMP), 1800)
    res["P100K_wall_s"] = round(time.time() - t0, 1)
    tm = d.get("timings", {})
    res["P100K"] = {"prompt_n": tm.get("prompt_n"),
                    "prefill_tps": round(tm.get("prompt_per_second", 0), 2)}
    res["D100K"] = {"decode_tps": round(tm.get("predicted_per_second", 0), 2),
                    "draft_n": tm.get("draft_n", 0), "draft_n_accepted": tm.get("draft_n_accepted", 0)}
    res["vram_peak_100k_mib"] = gpu_used()
    res["memavail_after_100k_mib"] = mem_avail_mib()
  except Exception as e:
    res["P100K_err"] = repr(e)

os.makedirs(os.path.dirname(OUT), exist_ok=True)
json.dump(res, open(OUT, "w"), indent=2)
print(json.dumps(res, indent=2))
