#!/usr/bin/env python3
# Dual-GPU soak: agent-style text loop (GPU0) + heavy vision loop (GPU1), concurrent.
# Samples RSS + per-GPU VRAM every 30s. Flags monotonic climb, GPU1 device exceptions,
# and prefix-cache regression. Env: SOAK_SEC (default 3000), TEXT/VISION/ROUTER URLs.
import os, sys, json, time, base64, threading, subprocess, urllib.request, statistics

TEXT   = os.environ.get("TEXT",   "http://127.0.0.1:8080")
VISION = os.environ.get("VISION", "http://127.0.0.1:8081")
ROUTER = os.environ.get("ROUTER", "http://127.0.0.1:8000")
DUR    = int(os.environ.get("SOAK_SEC", "3000"))
IMG    = os.path.expanduser("~/ai/qwen36/models/test-image.png")
CSV    = os.environ.get("SOAK_CSV", "/tmp/claude-1000/-mnt-d-LMServer/284c36d2-a897-4f4c-9c73-764205ad0495/scratchpad/soak.csv")

stop=False
counters={"text_turns":0,"text_err":0,"vis_gens":0,"vis_err":0,"gpu1_exceptions":0}
lock=threading.Lock()

def post(base, path, obj, timeout=180):
    req=urllib.request.Request(base+path, data=json.dumps(obj).encode(), headers={"Content-Type":"application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)

TOOLS=[{"type":"function","function":{"name":"run","description":"run a shell command","parameters":{"type":"object","properties":{"cmd":{"type":"string"}},"required":["cmd"]}}}]

def text_loop():
    # simulate an agent: multi-turn, growing context, periodic tool calls; reset session when large
    while not stop:
        msgs=[{"role":"system","content":"You are Qwen, created by Alibaba Cloud. You are a helpful assistant."}]
        for turn in range(12):
            if stop: break
            if turn%3==1:
                msgs.append({"role":"user","content":f"Call the run tool to list files in /var/log (turn {turn})."})
                body={"model":"x","messages":msgs,"tools":TOOLS,"max_tokens":300,"temperature":0.4,"stream":False}
            else:
                msgs.append({"role":"user","content":f"Step {turn}: write a short Python helper and explain it in 2 lines."})
                body={"model":"x","messages":msgs,"max_tokens":400,"temperature":0.6,"stream":False}
            try:
                d=post(TEXT,"/v1/chat/completions",body)
                m=d["choices"][0]["message"]
                # append assistant turn (with tool result if any) to grow context
                if m.get("tool_calls"):
                    msgs.append({"role":"assistant","content":m.get("content") or "","tool_calls":m["tool_calls"]})
                    msgs.append({"role":"tool","tool_call_id":m["tool_calls"][0]["id"],"content":"file1.log\nfile2.log\nsyslog"})
                else:
                    msgs.append({"role":"assistant","content":m.get("content") or "(thinking)"})
                with lock: counters["text_turns"]+=1
            except Exception as e:
                with lock: counters["text_err"]+=1
                time.sleep(1)

def vision_loop():
    b64=base64.b64encode(open(IMG,"rb").read()).decode()
    url=f"data:image/png;base64,{b64}"
    prompts=["Read the invoice number.","What is the amount due?","Read the fine print code.","List all text you see."]
    i=0
    while not stop:
        body={"model":"x","stream":False,"max_tokens":80,"temperature":0.4,
              "messages":[{"role":"user","content":[{"type":"text","text":prompts[i%len(prompts)]},
                           {"type":"image_url","image_url":{"url":url}}]}]}
        i+=1
        try:
            post(VISION,"/v1/chat/completions",body,timeout=120)
            with lock: counters["vis_gens"]+=1
        except urllib.error.HTTPError as e:
            with lock: counters["vis_err"]+=1
        except Exception as e:
            # connection refused/reset => server may have crashed (possible #24399)
            with lock:
                counters["vis_err"]+=1
                # check if Server B is dead
                try:
                    urllib.request.urlopen(VISION+"/health",timeout=3)
                except Exception:
                    counters["gpu1_exceptions"]+=1
            time.sleep(2)

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
    t=[threading.Thread(target=text_loop,daemon=True),threading.Thread(target=vision_loop,daemon=True)]
    for x in t: x.start()
    start=time.time(); rows=[]
    with open(CSV,"w") as f: f.write("t,rss_kb,vram0,vram1,text_turns,vis_gens,text_err,vis_err,gpu1_exc,prefix_ms\n")
    while time.time()-start < DUR:
        time.sleep(30)
        el=int(time.time()-start)
        pm = prefix_probe() if (el//30)%4==0 else -1   # probe cache every ~2 min
        with lock: c=dict(counters)
        r0,r1,rr=vram(0),vram(1),rss_total()
        row=(el,rr,r0,r1,c["text_turns"],c["vis_gens"],c["text_err"],c["vis_err"],c["gpu1_exceptions"],pm)
        rows.append(row)
        with open(CSV,"a") as f: f.write(",".join(map(str,row))+"\n")
        print(f"[{el:4}s] rss={rr//1024}MiB vram0={r0} vram1={r1} | text={c['text_turns']} vis={c['vis_gens']} err(t/v)={c['text_err']}/{c['vis_err']} gpu1_exc={c['gpu1_exceptions']} prefix_ms={pm}", flush=True)
    stop=True; time.sleep(3)
    # analysis
    def climb(idx):
        v=[r[idx] for r in rows if r[idx]>=0]
        if len(v)<6: return 0
        q=len(v)//4
        return statistics.mean(v[-q:])-statistics.mean(v[:q])
    print("\n==== SOAK SUMMARY ====")
    print(f"duration {int(time.time()-start)}s  text_turns={counters['text_turns']}  vis_gens={counters['vis_gens']}")
    print(f"errors: text={counters['text_err']} vision={counters['vis_err']}  GPU1 device exceptions={counters['gpu1_exceptions']}")
    rssc=climb(1)//1024; v0c=climb(2); v1c=climb(3)
    pref=[r[9] for r in rows if r[9]>0]
    print(f"RSS climb (last-first quartile mean): {rssc} MiB")
    print(f"VRAM0 climb: {v0c} MiB   VRAM1 climb: {v1c} MiB")
    if pref: print(f"prefix reprefill ms: first={pref[0]} last={pref[-1]} (max {max(pref)})")
    ok = (abs(rssc)<2048) and (v0c<400) and (v1c<400) and counters['gpu1_exceptions']==0 and counters['vis_gens']>300
    print("VERDICT:", "PASS" if ok else "REVIEW", "(no monotonic climb, no GPU1 exception, >300 vision gens)" if ok else "(see numbers above)")

if __name__=="__main__": main()
