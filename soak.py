#!/usr/bin/env python3
# Text soak: agent-style multi-turn loop against ONE loaded model. Samples RSS + BOTH GPUs' VRAM
# every 30s and flags monotonic climb (a leak) or prefix-cache regression.
#
# VISION HALF REMOVED: vision was retired with the llama-swap re-architecture, and port 8081 is now
# the Q6 27B (a TEXT server). The old vision_loop POSTed image payloads to 8081 — which, post-ship,
# meant a text-only model would *answer* image prompts rather than refuse: a quietly wrong test, not
# a crash. Deleted rather than left dangling (same call regress.sh made). For long-context / MoE soak
# see soak35.py; this one is the general single-model text soak.
#
# Env: SOAK_SEC (default 3000). TEXT = backend of the model under test (8080=27b-Q4, 8081=27b-Q6,
# 8082=35b, 8084=122b), or point at the router :8000 and set MODEL to a valid id. MODEL defaults to
# "x" (ignored when hitting a backend directly).
import os, sys, json, time, threading, subprocess, urllib.request, statistics

TEXT   = os.environ.get("TEXT",  "http://127.0.0.1:8080")
MODEL  = os.environ.get("MODEL", "x")   # ignored by a direct backend; set a real id if TEXT is the router
DUR    = int(os.environ.get("SOAK_SEC", "3000"))
CSV    = os.environ.get("SOAK_CSV", "/tmp/qwen_soak.csv")

stop=False
counters={"text_turns":0,"text_err":0}
lock=threading.Lock()

def post(base, path, obj, timeout=180):
    req=urllib.request.Request(base+path, data=json.dumps(obj).encode(), headers={"Content-Type":"application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)

TOOLS=[{"type":"function","function":{"name":"run","description":"run a shell command","parameters":{"type":"object","properties":{"cmd":{"type":"string"}},"required":["cmd"]}}}]

def text_loop():
    # simulate an agent: multi-turn, growing context, periodic tool calls; reset session each cycle
    while not stop:
        msgs=[{"role":"system","content":"You are Qwen, created by Alibaba Cloud. You are a helpful assistant."}]
        for turn in range(12):
            if stop: break
            if turn%3==1:
                msgs.append({"role":"user","content":f"Call the run tool to list files in /var/log (turn {turn})."})
                body={"model":MODEL,"messages":msgs,"tools":TOOLS,"max_tokens":2048,"temperature":0.4,"stream":False}
            else:
                msgs.append({"role":"user","content":f"Step {turn}: write a short Python helper and explain it in 2 lines."})
                body={"model":MODEL,"messages":msgs,"max_tokens":2048,"temperature":0.6,"stream":False}
            try:
                d=post(TEXT,"/v1/chat/completions",body)
                m=d["choices"][0]["message"]
                # append assistant turn (with tool result if any) to grow context
                if m.get("tool_calls"):
                    msgs.append({"role":"assistant","content":m.get("content") or "","tool_calls":m["tool_calls"]})
                    msgs.append({"role":"tool","tool_call_id":m["tool_calls"][0]["id"],"content":"file1.log\nfile2.log\nsyslog"})
                else:
                    msgs.append({"role":"assistant","content":m.get("content") or m.get("reasoning_content") or "(thinking)"})
                with lock: counters["text_turns"]+=1
            except Exception:
                with lock: counters["text_err"]+=1
                time.sleep(1)

def vram(i):
    try:
        return int(subprocess.check_output(["nvidia-smi","--query-gpu=memory.used","--format=csv,noheader,nounits","-i",str(i)]).decode().strip())
    except: return -1

def rss_total():
    try:
        out=subprocess.check_output(["bash","-c","ps --no-headers -o rss -C llama-server 2>/dev/null | awk '{s+=$1} END{print s}'"]).decode().strip()
        return int(out or 0)
    except: return -1

def prefix_probe():
    # 3K-ish prefix twice; return second prompt_ms (should stay low = cache reuse alive)
    pfx="The quick brown fox jumps over the lazy dog. "*400+" Summarize in 5 words."
    body={"prompt":pfx,"n_predict":8,"cache_prompt":True,"temperature":0.0,"stream":False}
    try:
        post(TEXT,"/completion",body); d=post(TEXT,"/completion",body)
        return round(d["timings"]["prompt_ms"],1)
    except Exception: return -1

def main():
    global stop
    threading.Thread(target=text_loop,daemon=True).start()
    start=time.time(); rows=[]
    with open(CSV,"w") as f: f.write("t,rss_kb,vram0,vram1,text_turns,text_err,prefix_ms\n")
    while time.time()-start < DUR:
        time.sleep(30)
        el=int(time.time()-start)
        pm = prefix_probe() if (el//30)%4==0 else -1   # probe cache every ~2 min
        with lock: c=dict(counters)
        r0,r1,rr=vram(0),vram(1),rss_total()
        row=(el,rr,r0,r1,c["text_turns"],c["text_err"],pm)
        rows.append(row)
        with open(CSV,"a") as f: f.write(",".join(map(str,row))+"\n")
        print(f"[{el:4}s] rss={rr//1024}MiB vram0={r0} vram1={r1} | turns={c['text_turns']} err={c['text_err']} prefix_ms={pm}", flush=True)
    stop=True; time.sleep(3)
    # analysis: last-quartile minus first-quartile mean of each series (mmap'd tiers oscillate, so a
    # quartile regression is the honest leak signal, not an endpoint delta)
    def climb(idx):
        v=[r[idx] for r in rows if r[idx]>=0]
        if len(v)<6: return 0
        q=len(v)//4
        return statistics.mean(v[-q:])-statistics.mean(v[:q])
    print("\n==== SOAK SUMMARY ====")
    print(f"duration {int(time.time()-start)}s  text_turns={counters['text_turns']}  errors={counters['text_err']}")
    rssc=climb(1)//1024; v0c=climb(2); v1c=climb(3)
    pref=[r[6] for r in rows if r[6]>0]
    print(f"RSS climb (last-first quartile mean): {rssc} MiB")
    print(f"VRAM0 climb: {v0c} MiB   VRAM1 climb: {v1c} MiB")
    if pref: print(f"prefix reprefill ms: first={pref[0]} last={pref[-1]} (max {max(pref)})")
    ok = (abs(rssc)<2048) and (v0c<400) and (v1c<400) and counters['text_turns']>50 and counters['text_err']==0
    print("VERDICT:", "PASS" if ok else "REVIEW",
          "(no monotonic climb, clean turns)" if ok else "(see numbers above)")

if __name__=="__main__": main()
