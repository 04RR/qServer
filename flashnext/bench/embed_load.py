#!/usr/bin/env python3
"""Continuous embedding load against a standalone CPU embedder (:8097) to create CPU-thread
contention for the thread-contention test (round 2). Fires NWORK parallel /v1/embeddings
requests in a tight loop until a stop-file appears. Each request embeds a chunk near the
embedder's CTX so it does real CPU work. Prints a throughput line every few seconds."""
import os, sys, json, time, threading, urllib.request

B    = os.environ.get("EMBED_BASE", "http://127.0.0.1:8097")
NWORK= int(os.environ.get("NWORK", "4"))
STOP = os.environ.get("STOP_FILE", "/home/rohit/ai/qwen36/flashnext/bench/.embed_stop")
# chunk sized to stay under the embedder's CTX (one ubatch); keeps CPU busy without big RAM buffers
NSENT = int(os.environ.get("CHUNK_SENTS", "90"))
CHUNK = (" ".join(f"Document sentence {i}: retrieval-augmented generation chunk content, "
                  f"semantically dense text for embedding." for i in range(1, NSENT)))

def worker(stats, idx):
    while not os.path.exists(STOP):
        body = {"model": "qwen-embed", "input": CHUNK}
        req = urllib.request.Request(B + "/v1/embeddings", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        try:
            urllib.request.urlopen(req, timeout=120).read()
            stats[idx] += 1
        except Exception:
            time.sleep(0.2)

def main():
    if os.path.exists(STOP):
        os.remove(STOP)
    stats = [0]*NWORK
    ts = [threading.Thread(target=worker, args=(stats, i), daemon=True) for i in range(NWORK)]
    for t in ts: t.start()
    t0 = time.time(); last = 0
    while not os.path.exists(STOP):
        time.sleep(4)
        tot = sum(stats)
        print(f"[embed_load] {tot} embeds done ({(tot-last)/4:.1f}/s), {NWORK} workers", flush=True)
        last = tot
    print(f"[embed_load] stop-file seen; {sum(stats)} embeds in {time.time()-t0:.0f}s", flush=True)

if __name__ == "__main__":
    main()
