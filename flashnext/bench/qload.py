#!/usr/bin/env python3
"""Realistic RAG query-embed load (round 2, follow-up): fire ONE ~50-token /v1/embeddings call
every INTERVAL seconds against the CPU embedder (:8097), until a stop-file appears or MAX_S elapses.
This models light production query traffic (one search query embedded per turn), NOT the heavy
batch-corpus indexing of the earlier contention test. Prints a cadence line periodically and the
measured query-token count once. Writes a small JSON summary to $OUT on exit."""
import os, json, time, urllib.request

B        = os.environ.get("EMBED_BASE", "http://127.0.0.1:8097")
INTERVAL = float(os.environ.get("INTERVAL", "2.0"))
MAX_S    = float(os.environ.get("MAX_S", "600"))
STOP     = os.environ.get("STOP_FILE", "/home/rohit/ai/qwen36/flashnext/bench/.qload_stop")
OUT      = os.environ.get("OUT", "/home/rohit/ai/qwen36/flashnext/bench/qload.json")

# A realistic ~50-token RAG search query (verified against the embedder's /tokenize at startup).
QUERY = ("What are the documented trade-offs between tensor-split placement and per-GPU VRAM "
         "headroom when serving a large mixture-of-experts model across two heterogeneous GPUs, "
         "and how does the choice interact with KV-cache quantization at long context lengths?")

def post(path, obj, timeout=120):
    r = urllib.request.Request(B + path, data=json.dumps(obj).encode(),
                               headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(r, timeout=timeout) as resp:
        return json.load(resp)

def main():
    if os.path.exists(STOP):
        os.remove(STOP)
    # measure the query's token count once (best-effort; embedder exposes /tokenize)
    qtok = None
    try:
        qtok = len(post("/tokenize", {"content": QUERY}, 30)["tokens"])
    except Exception:
        pass
    n = 0; errs = 0; lat = []
    t0 = time.time(); last = t0
    while not os.path.exists(STOP) and (time.time() - t0) < MAX_S:
        tick = time.time()
        try:
            rt0 = time.time()
            post("/v1/embeddings", {"model": "qwen-embed", "input": QUERY}, 60)
            lat.append(time.time() - rt0); n += 1
        except Exception:
            errs += 1
        if time.time() - last >= 10:
            avg = sum(lat) / len(lat) if lat else 0
            print(f"[qload] {n} embeds, {errs} errs, avg_lat={avg*1000:.0f}ms, qtok={qtok}", flush=True)
            last = time.time()
        # maintain the INTERVAL cadence (subtract the time the request already took)
        sleep = INTERVAL - (time.time() - tick)
        if sleep > 0:
            time.sleep(sleep)
    dur = time.time() - t0
    summ = {"embeds": n, "errors": errs, "dur_s": round(dur, 1), "interval_s": INTERVAL,
            "qtok": qtok, "avg_lat_ms": round(sum(lat) / len(lat) * 1000, 1) if lat else None,
            "rate_per_s": round(n / dur, 3) if dur else None}
    try:
        os.makedirs(os.path.dirname(OUT), exist_ok=True)
        json.dump(summ, open(OUT, "w"), indent=2)
    except Exception:
        pass
    print(f"[qload] DONE {json.dumps(summ)}", flush=True)

if __name__ == "__main__":
    main()
