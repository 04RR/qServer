#!/usr/bin/env python3
"""Concurrent aggregate-throughput bench for the BATCH lane (arm G). Fires N simultaneous
short-code /completion requests and reports aggregate decode tokens/sec (sum of predicted_n /
wall time) at each concurrency. Also a single-stream point. Writes JSON to $OUT."""
import os, json, time, threading, urllib.request

B    = os.environ.get("BASE", "http://127.0.0.1:8095")
ARM  = os.environ.get("ARM", "G")
OUT  = os.environ.get("OUT", f"/home/rohit/ai/qwen36/flashnext/bench/{ARM}.conc.json")
CONC = [int(x) for x in os.environ.get("CONC", "1,4,8").split(",")]
NPRED= int(os.environ.get("NPRED", "256"))
CODE = "Write a Python function that merges two sorted lists.\n\ndef merge("
SAMP = {"temperature": 1.0, "top_p": 0.95, "top_k": 20}

def one(results, idx):
    body = dict(prompt=CODE, n_predict=NPRED, cache_prompt=False, stream=False, **SAMP)
    req = urllib.request.Request(B + "/completion", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    try:
        d = json.load(urllib.request.urlopen(req, timeout=900))
        tm = d.get("timings", {})
        results[idx] = {"predicted_n": tm.get("predicted_n", 0),
                        "tps": tm.get("predicted_per_second", 0),
                        "wall": time.time() - t0}
    except Exception as e:
        results[idx] = {"err": repr(e), "wall": time.time() - t0}

res = {"arm": ARM, "base": B, "npred": NPRED, "runs": {}}
# warm once (fault experts) so the concurrency numbers aren't dominated by first-touch
one({}, 0)
for c in CONC:
    results = {}
    threads = [threading.Thread(target=one, args=(results, i)) for i in range(c)]
    t0 = time.time()
    for t in threads: t.start()
    for t in threads: t.join()
    wall = time.time() - t0
    toks = sum(r.get("predicted_n", 0) for r in results.values())
    errs = [r for r in results.values() if "err" in r]
    res["runs"][str(c)] = {"concurrency": c, "wall_s": round(wall, 2),
                           "total_tokens": toks, "aggregate_tps": round(toks / wall, 2),
                           "per_req_tps_mean": round(sum(r.get("tps", 0) for r in results.values()) / max(c, 1), 2),
                           "errors": len(errs)}
    print(f"conc={c}: aggregate {toks/wall:.1f} t/s  (wall {wall:.1f}s, {toks} tok, errs {len(errs)})")

os.makedirs(os.path.dirname(OUT), exist_ok=True)
json.dump(res, open(OUT, "w"), indent=2)
