#!/usr/bin/env python3
"""Flash-Next ROUND 2 bench against an already-running server (:8095). arm.sh launches the
server, waits /health, runs this, then SIGTERMs it. Implements the round-2 harness rules:

 1. warm-up: one discarded ~1K-token generation after /health, before any measurement.
 2. natural length: no ignore_eos; prompts force long output; n_predict ceiling 2048;
    flag+drop any run that produced < 400 tokens.
 3. chat template: POST /apply-template (system+user, thinking ON = production) then /completion;
    temp 1.0 / top_p 0.95 / top_k 20; 3 seeds/cell; report median + min..max.
 4. decode cells: code-short, prose-short, report-100K (100K haystack + 1500-word summary).
 5. prefill cells: P-8K, P-100K (cache_prompt false, n_predict 1).
 6. ctx checkpoints: report-100K prefills the haystack once (seed 1), seeds 2/3 reuse it
    (cache_prompt=True). Reuse is VERIFIED: seed-2 prompt_n must be ~suffix only, else we say
    reuse failed for qwen4exp and each seed paid full prefill. RSS cost of the ckpt is reported.
 7. per arm: VRAM/GPU after load + peak after 100K, MemAvailable, server VmRSS, pgmajfault delta
    during code-short.

Cells are selectable via CELLS (default code,prose,report100k,p8k,p100k). Writes JSON to $OUT."""
import os, json, time, subprocess, urllib.request, statistics

B      = os.environ.get("BASE", "http://127.0.0.1:8095")
ARM    = os.environ.get("ARM", "arm")
OUT    = os.environ.get("OUT", f"/home/rohit/ai/qwen36/flashnext/bench/{ARM}.json")
CELLS  = set(os.environ.get("CELLS", "code,prose,report100k,p8k,p100k").split(","))
NSEED  = int(os.environ.get("NSEED", "3"))
NPRED  = int(os.environ.get("NPRED", "2048"))          # ceiling; EOS stops earlier
P8K    = int(os.environ.get("P8K", "8000"))
P100K  = int(os.environ.get("P100K", "100000"))
PIDF   = os.environ.get("PIDFILE", f"/home/rohit/ai/qwen36/flashnext/bench/{ARM}.pid")
SYS    = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
SAMP   = {"temperature": 1.0, "top_p": 0.95, "top_k": 20}

CODE_Q  = ("Write a complete Python implementation of a thread-safe LRU cache with TTL expiry, "
           "full docstrings, and pytest unit tests.")
PROSE_Q = "Write a detailed 1,500-word essay on how the printing press changed European politics."
REPORT_Q = "Write a 1,500-word structured report summarizing the document above."

def _req(path, obj):
    return urllib.request.Request(B + path, data=json.dumps(obj).encode(),
                                  headers={"Content-Type": "application/json"})
def post(path, obj, timeout=1800):
    with urllib.request.urlopen(_req(path, obj), timeout=timeout) as r:
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
def server_rss_mib():
    try:
        pid = open(PIDF).read().strip()
        for ln in open(f"/proc/{pid}/status"):
            if ln.startswith("VmRSS"):
                return int(ln.split()[1]) // 1024
    except Exception:
        return None

def apply_template(user, system=SYS):
    """Return the production-templated prompt string (thinking ON by default for Qwen3)."""
    d = post("/apply-template", {"messages": [{"role": "system", "content": system},
                                              {"role": "user", "content": user}]}, 300)
    # llama.cpp returns {"prompt": "..."}
    return d.get("prompt") if isinstance(d, dict) else None

def gen(prompt, seed, n_predict=NPRED, cache_prompt=False):
    body = dict(prompt=prompt, n_predict=n_predict, cache_prompt=cache_prompt,
                stream=False, seed=seed, **SAMP)
    t0 = time.time()
    d = post("/completion", body, 1800)
    tm = d.get("timings", {})
    return {"seed": seed,
            "tps": round(tm.get("predicted_per_second", 0), 2),
            "predicted_n": tm.get("predicted_n", 0),
            "prompt_n": tm.get("prompt_n", 0),
            "prompt_tps": round(tm.get("prompt_per_second", 0), 2),
            "draft_n": tm.get("draft_n", 0) or 0,
            "draft_n_accepted": tm.get("draft_n_accepted", 0) or 0,
            "wall_s": round(time.time() - t0, 1)}

def summarize(runs, min_tok=400):
    good = [r for r in runs if r["predicted_n"] >= min_tok]
    used = good if good else runs
    tps = [r["tps"] for r in used]
    dn = sum(r["draft_n"] for r in runs); da = sum(r["draft_n_accepted"] for r in runs)
    return {"tps_median": round(statistics.median(tps), 2),
            "tps_min": round(min(tps), 2), "tps_max": round(max(tps), 2),
            "tokens_median": int(statistics.median([r["predicted_n"] for r in used])),
            "acceptance": round(da / dn, 3) if dn else None,
            "draft_n": dn, "draft_n_accepted": da,
            "dropped_short": [r["predicted_n"] for r in runs if r["predicted_n"] < min_tok],
            "runs": runs}

def build_filler(n):
    return " ".join(f"Log entry {i}: routine telemetry sample, nominal, no action required."
                    for i in range(1, n))
def calibrate(target):
    rate = ntok(build_filler(500)) / 500.0
    n = int(target / rate)
    for _ in range(4):
        t = ntok(build_filler(n))
        if abs(t - target) <= max(500, int(0.02 * target)):
            break
        n = int(n * target / max(t, 1))
    p = build_filler(n)
    return p, ntok(p)

res = {"arm": ARM, "base": B, "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
       "cells_requested": sorted(CELLS), "nseed": NSEED, "npred_ceiling": NPRED}

# ---- state right after load (experts still lazy) ----
try:
    res["vram_after_load_mib"] = gpu_used()
    res["memavail_after_load_mib"] = mem_avail_mib()
    res["rss_after_load_mib"] = server_rss_mib()
except Exception as e:
    res["load_state_err"] = repr(e)

# ---- harness rule 1: discarded warm-up (~1K tokens), also faults experts ----
try:
    wp = apply_template("Briefly introduce yourself and list three things you can help with.")
    res["template_ok"] = bool(wp)
    res["template_head"] = (wp or "")[-220:]        # tail shows assistant/think preamble
    t0 = time.time()
    wd = gen(wp, seed=1, n_predict=1024, cache_prompt=False)
    res["warmup"] = {"tokens": wd["predicted_n"], "tps": wd["tps"], "wall_s": round(time.time()-t0,1)}
except Exception as e:
    res["warmup_err"] = repr(e)

# ---- decode cell: code-short (+ pgmajfault delta here, rule 7) ----
if "code" in CELLS:
    try:
        p = apply_template(CODE_Q)
        pf0 = pgmajfault()
        runs = [gen(p, seed=s) for s in range(1, NSEED + 1)]
        res.setdefault("decode", {})["code"] = summarize(runs)
        res["pgmajfault_delta_code"] = pgmajfault() - pf0
    except Exception as e:
        res.setdefault("decode", {})["code_err"] = repr(e)

# ---- decode cell: prose-short ----
if "prose" in CELLS:
    try:
        p = apply_template(PROSE_Q)
        runs = [gen(p, seed=s) for s in range(1, NSEED + 1)]
        res.setdefault("decode", {})["prose"] = summarize(runs)
    except Exception as e:
        res.setdefault("decode", {})["prose_err"] = repr(e)

# ---- decode cell: report-100K + checkpoint-reuse verification ----
if "report100k" in CELLS:
    try:
        hay, hay_tok = calibrate(P100K)
        user = hay + "\n\n" + REPORT_Q
        prompt = apply_template(user)
        res.setdefault("decode", {}).setdefault("report100k", {})["haystack_tokens"] = hay_tok
        rss0 = server_rss_mib()
        runs = []
        for s in range(1, NSEED + 1):
            r = gen(prompt, seed=s, cache_prompt=True)   # seed1 prefills; 2/3 should reuse
            if s == 1:
                res["decode"]["report100k"]["rss_before_mib"] = rss0
                res["decode"]["report100k"]["rss_after_prefill_mib"] = server_rss_mib()
            runs.append(r)
        sm = summarize(runs)
        # reuse check: seed-2 prompt_n should be ~suffix only (<< full), else reuse failed
        pn1, pn2 = runs[0]["prompt_n"], runs[1]["prompt_n"]
        sm["prefill_reuse"] = {"seed1_prompt_n": pn1, "seed2_prompt_n": pn2,
                               "reused": pn2 < pn1 * 0.25,
                               "ckpt_rss_delta_mib": (res["decode"]["report100k"].get("rss_after_prefill_mib") or 0)
                                                     - (rss0 or 0)}
        res["decode"]["report100k"].update(sm)
        res["vram_peak_mib"] = gpu_used()
        res["memavail_after_100k_mib"] = mem_avail_mib()
        res["rss_after_100k_mib"] = server_rss_mib()
    except Exception as e:
        res.setdefault("decode", {})["report100k_err"] = repr(e)

# ---- prefill cells ----
if "p8k" in CELLS:
    try:
        p8, n8 = calibrate(P8K)
        d = post("/completion", dict(prompt=p8, n_predict=1, cache_prompt=False, temperature=0.0), 600)
        tm = d.get("timings", {})
        res.setdefault("prefill", {})["p8k"] = {"prompt_n": tm.get("prompt_n"),
                                                "tps": round(tm.get("prompt_per_second", 0), 2)}
    except Exception as e:
        res.setdefault("prefill", {})["p8k_err"] = repr(e)
if "p100k" in CELLS:
    try:
        p100, n100 = calibrate(P100K)
        t0 = time.time()
        d = post("/completion", dict(prompt=p100, n_predict=1, cache_prompt=False, temperature=0.0), 1800)
        tm = d.get("timings", {})
        res.setdefault("prefill", {})["p100k"] = {"prompt_n": tm.get("prompt_n"),
                                                  "tps": round(tm.get("prompt_per_second", 0), 2),
                                                  "wall_s": round(time.time() - t0, 1)}
        res.setdefault("vram_peak_mib", gpu_used())
    except Exception as e:
        res.setdefault("prefill", {})["p100k_err"] = repr(e)

os.makedirs(os.path.dirname(OUT), exist_ok=True)
json.dump(res, open(OUT, "w"), indent=2)
# compact console summary
d = res.get("decode", {})
def cell(n):
    c = d.get(n, {})
    return f"{n}={c.get('tps_median','-')}t/s(tok{c.get('tokens_median','-')},acc{c.get('acceptance','-')})"
print(f"[{ARM}] " + "  ".join(cell(n) for n in ("code","prose","report100k") if n in d))
print(f"       prefill={res.get('prefill')}  vram_load={res.get('vram_after_load_mib')} "
      f"vram_peak={res.get('vram_peak_mib')} rss100k={res.get('rss_after_100k_mib')} "
      f"faults_code={res.get('pgmajfault_delta_code')}")
r100 = d.get("report100k", {})
if "prefill_reuse" in r100:
    print(f"       ckpt_reuse={r100['prefill_reuse']}")
