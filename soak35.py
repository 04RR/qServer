#!/usr/bin/env python3
"""35B (MoE) soak — sustained load through the SHIPPED chain (:8000 router -> llama-swap -> model).

Differences from the 27B soak (soak.py), which ran text-on-GPU0 + vision-on-GPU1 as two servers:
  - ONE model spans BOTH GPUs, so the new risk is CROSS-GPU sustained decode, not two independent
    servers. VRAM is sampled on both cards; a climb on GPU1 would mean the spilled experts leak.
  - Runs through :8000 (router + llama-swap), i.e. the real client path. The 27B soak hit :8080
    directly and therefore had NO tool guard -- which is how the 10/806 truncation errors appeared.
    Here the guard IS in path, so tool errors should be 0. That is the point of testing this path.
  - Adds a digit-loop canary turn type: the cross-GPU MTP verify/rollback path is the one genuinely
    novel thing in this build, so it is checked continuously, not just once at M2.
  - The server is systemd/llama-swap-owned, so it survives this client dying.

CSV is flushed every sample so a killed run still yields data.
Env: SOAK_SEC (default 2700), SOAK_CSV, MODEL, ROUTER.
"""
import os, sys, json, time, threading, subprocess, urllib.request, urllib.error, statistics, re, csv

ROUTER   = os.environ.get("ROUTER", "http://127.0.0.1:8000")
MODEL    = os.environ.get("MODEL", "qwen-35b")
SOAK_SEC = int(os.environ.get("SOAK_SEC", "2700"))
# PREFIX_PROBE=0 disables the in-soak prefix-cache probe. LEARNED ON THE 35B SOAK: with
# --parallel 1 the probe QUEUES behind the worker's in-flight request, so it measures contention,
# not reprefill (it swung 939-8573 ms with NO trend; the max matched a canary turn's duration
# exactly). Probe an idle server after the soak instead. Default on for back-compat.
PREFIX_PROBE = os.environ.get("PREFIX_PROBE", "1") == "1"
# The 122B thinks far longer than the 35B (reasoning 4472-6647 chars vs ~1500), so its canary
# turns need a bigger budget or they truncate and shrink the sample the detector sees.
CANARY_MAX = int(os.environ.get("CANARY_MAX", "1200"))
SCR      = "/tmp/claude-1000/-mnt-d-LMServer/284c36d2-a897-4f4c-9c73-764205ad0495/scratchpad"
CSVP     = os.environ.get("SOAK_CSV", f"{SCR}/soak35.csv")
ERRP     = f"{SCR}/soak35_errors.jsonl"
SYS      = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."

sys.path.insert(0, "/home/rohit/ai/qwen36")
from canary import analyse as canary_analyse   # same detector as the M2 hard gate

lock = threading.Event()
stats = {"turns": 0, "errors": 0, "tool_turns": 0, "tool_errors": 0,
         "canary_turns": 0, "runaways": 0, "longctx_turns": 0,
         "tg": [], "prefix_ms": [], "accept": []}
slock = threading.Lock()
t_start = time.time()


def post(path, obj, timeout=900):
    r = urllib.request.Request(ROUTER + path, data=json.dumps(obj).encode(),
                               headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(r, timeout=timeout))


def log_err(kind, detail):
    with slock:
        stats["errors"] += 1
    with open(ERRP, "a") as f:
        f.write(json.dumps({"t": round(time.time() - t_start, 1), "kind": kind, "detail": str(detail)[:600]}) + "\n")


def chat(messages, **kw):
    body = {"model": MODEL, "messages": messages, "stream": False}
    body.update(kw)
    return post("/v1/chat/completions", body)


TOOLS = [{"type": "function", "function": {
    "name": "run", "description": "run a shell command",
    "parameters": {"type": "object", "properties": {"cmd": {"type": "string"}}, "required": ["cmd"]}}}]

CANARY_PROMPTS = [
    "Repeat back this random string then write prose about it: qX7#mept Zq9 vbrr 42a! kkap. No numbered list.",
    "Transcribe exactly then discuss: 8f3-KKw ;;q4 Zephyr#9912 mmmm bq. Avoid lists.",
    "Echo this then continue in flowing prose: 77x QQ!zz 04-4-04 plup vvvvv 31337 aa. No enumeration.",
    "Describe the sound of a distant train in flowing prose. Avoid any lists or enumeration.",
]
# NOTE canary prompt rules (learned in M2): no ascending integer run in the prompt (the model's
# correct echo would look like a runaway) and never ask for "N words" (induces an ascending
# word-count in the reasoning). See LEARNINGS.md.

RAG_PROMPT = None   # built once, lazily (100K tokens)


def build_rag():
    f = " ".join(f"Log entry {i}: routine telemetry sample, nominal, no action required." for i in range(1, 5612))
    h = len(f) // 2
    return (f[:h] + "\n\nThe vault access code is MERIDIAN-COBALT-7.\n\n" + f[h:] +
            "\n\nQuestion: what is the vault access code? Answer with just the code.")


def worker():
    """One sequential worker: the server runs --parallel 1, so sequential IS full utilisation.
    Cycles turn types so every path stays exercised for the whole run."""
    global RAG_PROMPT
    convo = [{"role": "system", "content": SYS}]
    n = 0
    while not lock.is_set():
        n += 1
        kind = n % 10
        try:
            if kind in (3, 8):
                # ---- tool turn. max_tokens deliberately LOW (80): without the router guard this
                # is exactly what truncated the tool JSON and produced the 27B soak's 10 errors.
                # Through :8000 the guard must raise it to 512 and these must all succeed.
                with slock: stats["tool_turns"] += 1
                d = chat([{"role": "system", "content": SYS},
                          {"role": "user", "content": "List /var/log using the run tool."}],
                         tools=TOOLS, max_tokens=80, temperature=0.6)
                if "error" in d:
                    with slock: stats["tool_errors"] += 1
                    log_err("tool", d["error"]); continue
                tc = d["choices"][0]["message"].get("tool_calls")
                if not tc:
                    with slock: stats["tool_errors"] += 1
                    log_err("tool_no_call", d["choices"][0]); continue
                json.loads(tc[0]["function"]["arguments"])   # must parse
            elif kind in (5, 9):
                # ---- canary turn (cross-GPU MTP rollback under draft rejection)
                with slock: stats["canary_turns"] += 1
                p = CANARY_PROMPTS[n % len(CANARY_PROMPTS)]
                d = chat([{"role": "system", "content": SYS}, {"role": "user", "content": p}],
                         max_tokens=CANARY_MAX, temperature=1.2)
                if "error" in d:
                    log_err("canary", d["error"]); continue
                m = d["choices"][0]["message"]
                txt = (m.get("reasoning_content") or "") + "\n" + (m.get("content") or "")
                asc, gap, mrep = canary_analyse(txt)
                if (asc >= 15 and gap is not None and gap <= 4.0) or mrep >= 30:
                    with slock: stats["runaways"] += 1
                    log_err("RUNAWAY", f"asc={asc} gap={gap} rep={mrep} :: {txt[-300:]}")
            elif kind == 7:
                # ---- long-context turn: KV at depth, cross-GPU
                if RAG_PROMPT is None:
                    RAG_PROMPT = build_rag()
                with slock: stats["longctx_turns"] += 1
                d = chat([{"role": "system", "content": SYS}, {"role": "user", "content": RAG_PROMPT}],
                         max_tokens=2048, temperature=0.0)
                if "error" in d:
                    log_err("longctx", d["error"]); continue
                c = d["choices"][0]["message"].get("content") or ""
                if "MERIDIAN-COBALT-7" not in c:
                    log_err("longctx_recall_miss", c[:200])
            else:
                # ---- agent turn: multi-turn, growing context
                convo.append({"role": "user", "content":
                              f"Turn {n}: explain one more subtle aspect of B-tree rebalancing, "
                              f"building on what you already said. Be specific."})
                d = chat(convo, max_tokens=600, temperature=0.7)
                if "error" in d:
                    log_err("agent", d["error"]); convo = [{"role": "system", "content": SYS}]; continue
                m = d["choices"][0]["message"]
                convo.append({"role": "assistant", "content": m.get("content") or ""})
                if len(convo) > 15:            # reset before hitting ctx; growth is the point, not overflow
                    convo = [{"role": "system", "content": SYS}]
            with slock:
                stats["turns"] += 1
                tm = d.get("timings") or {}
                if tm.get("predicted_per_second"):
                    stats["tg"].append(tm["predicted_per_second"])
                dn, da = tm.get("draft_n", 0), tm.get("draft_n_accepted", 0)
                if dn:
                    stats["accept"].append(da / dn)
        except urllib.error.HTTPError as e:
            log_err("http", f"{e.code}: {e.read().decode()[:300]}")
        except Exception as e:
            log_err(type(e).__name__, e)


def vram(i):
    try:
        return int(subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits", "-i", str(i)],
            timeout=10).decode().strip())
    except Exception:
        return -1


def server_rss():
    """RSS of the llama-server llama-swap spawned. The 27B's feared failure mode was an MTP
    memory blowup, so this must not climb monotonically."""
    try:
        pid = subprocess.check_output(["pgrep", "-f", "bin/llama-server"], timeout=10).decode().split()[0]
        with open(f"/proc/{pid}/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) // 1024   # MiB
    except Exception:
        pass
    return -1


def prefix_probe():
    """Re-prefill a stable prefix; if the prompt cache degrades this climbs."""
    p = "You are a helpful assistant. " + ("The quick brown fox jumps over the lazy dog. " * 260)
    t0 = time.time()
    try:
        post("/v1/chat/completions", {"model": MODEL, "messages": [{"role": "user", "content": p + "\nSay OK."}],
                                      "max_tokens": 8, "temperature": 0.0, "stream": False}, timeout=300)
        return (time.time() - t0) * 1000
    except Exception:
        return -1


def sampler(w):
    with open(CSVP, "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["t_s", "gpu0_mib", "gpu1_mib", "rss_mib", "turns", "errors",
                     "tool_turns", "tool_errors", "canary_turns", "runaways", "longctx", "prefix_ms"])
        fh.flush()
        i = 0
        while not lock.is_set():
            i += 1
            t = round(time.time() - t_start, 1)
            pm = prefix_probe() if (PREFIX_PROBE and i % 4 == 1) else ""   # see PREFIX_PROBE note
            with slock:
                if isinstance(pm, float) and pm > 0:
                    stats["prefix_ms"].append(pm)
                row = [t, vram(0), vram(1), server_rss(), stats["turns"], stats["errors"],
                       stats["tool_turns"], stats["tool_errors"], stats["canary_turns"],
                       stats["runaways"], stats["longctx_turns"],
                       round(pm, 1) if isinstance(pm, float) and pm > 0 else ""]
            wr.writerow(row); fh.flush()
            print(f"  t={t:6.0f}s gpu0={row[1]:6d} gpu1={row[2]:5d} rss={row[3]:6d}MiB "
                  f"turns={row[4]:4d} err={row[5]} tool_err={row[7]} runaway={row[9]}", flush=True)
            lock.wait(30)


def main():
    open(ERRP, "w").close()
    print(f"soak35: {SOAK_SEC}s through {ROUTER} model={MODEL}")
    # make sure the model is loaded before t=0 so the load isn't counted as a stall
    try:
        chat([{"role": "system", "content": SYS}, {"role": "user", "content": "Say OK."}], max_tokens=2048)
    except Exception as e:
        print(f"  preload failed: {e}")
    w = threading.Thread(target=worker, daemon=True)
    s = threading.Thread(target=sampler, args=(w,), daemon=True)
    w.start(); s.start()
    try:
        time.sleep(SOAK_SEC)
    except KeyboardInterrupt:
        pass
    lock.set()
    time.sleep(2)

    # ---- verdict ----
    import csv as _csv
    rows = list(_csv.DictReader(open(CSVP)))
    g0 = [int(r["gpu0_mib"]) for r in rows if int(r["gpu0_mib"]) > 0]
    g1 = [int(r["gpu1_mib"]) for r in rows if int(r["gpu1_mib"]) > 0]
    rs = [int(r["rss_mib"]) for r in rows if int(r["rss_mib"]) > 0]
    pf = [float(r["prefix_ms"]) for r in rows if r["prefix_ms"]]
    dur = time.time() - t_start
    print("\n================ SOAK 35B VERDICT ================")
    print(f"duration      {dur:.0f}s   turns={stats['turns']}  errors={stats['errors']}")
    print(f"  tool turns  {stats['tool_turns']}  tool_errors={stats['tool_errors']}   (guard in path -> expect 0)")
    print(f"  canary      {stats['canary_turns']}  RUNAWAYS={stats['runaways']}         (expect 0)")
    print(f"  long-ctx    {stats['longctx_turns']}")
    if stats["tg"]:
        print(f"  tg          median={statistics.median(stats['tg']):.1f} t/s  "
              f"min={min(stats['tg']):.1f} max={max(stats['tg']):.1f}")
    if stats["accept"]:
        print(f"  acceptance  median={statistics.median(stats['accept']):.3f}")
    if g0: print(f"GPU0 VRAM     {g0[0]} -> {g0[-1]} MiB (min {min(g0)} max {max(g0)}) climb={g0[-1]-g0[0]:+d}")
    if g1: print(f"GPU1 VRAM     {g1[0]} -> {g1[-1]} MiB (min {min(g1)} max {max(g1)}) climb={g1[-1]-g1[0]:+d}")
    # RSS: the model is mmap'd, so RSS NECESSARILY climbs early as the 21 GB of weights page in.
    # That is page-in, NOT the MTP memory blowup this soak exists to detect. Reporting the whole-run
    # delta would report warm-up as a leak. The leak signal is the STEADY-STATE (second-half) climb.
    if rs:
        half = len(rs) // 2
        ss = rs[half:]
        print(f"server RSS    full-run {rs[0]} -> {rs[-1]} MiB (max {max(rs)}) "
              f"climb={rs[-1]-rs[0]:+d}  <- includes mmap page-in, expected")
        print(f"              STEADY-STATE (2nd half) {ss[0]} -> {ss[-1]} MiB climb={ss[-1]-ss[0]:+d}"
              f"  <- this is the leak signal")
    if pf:
        print(f"prefix probe  {pf[0]:.0f} -> {pf[-1]:.0f} ms (max {max(pf):.0f})")
        if len(pf) > 2:
            print(f"              after first (cold): {pf[1]:.0f} -> {pf[-1]:.0f} ms "
                  f"(median {statistics.median(pf[1:]):.0f})")
    print(f"\nCSV: {CSVP}\nerrors: {ERRP}")


if __name__ == "__main__":
    main()
