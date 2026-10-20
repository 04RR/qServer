#!/usr/bin/env python3
"""Concurrent-burst VRAM probe (round 2, batch lane): fire CONC simultaneous /completion requests,
each with an ~8K-token prompt, while polling nvidia-smi for peak per-GPU VRAM. Verifies the batch
server (e.g. G1 at UB=4096, NP=4) keeps >= 1 GiB free per GPU under an 8-wide 8K-prompt burst.
Writes JSON to $OUT."""
import os, json, time, threading, subprocess, urllib.request

B    = os.environ.get("BASE", "http://127.0.0.1:8095")
ARM  = os.environ.get("ARM", "burst")
OUT  = os.environ.get("OUT", f"/home/rohit/ai/qwen36/flashnext/bench/{ARM}.burst.json")
CONC = int(os.environ.get("CONC", "8"))
PTOK = int(os.environ.get("PTOK", "8000"))
NPRED= int(os.environ.get("NPRED", "64"))
TOTAL= [24564, 16311]  # 4090, 5060 Ti

def gpu_used():
    out = subprocess.check_output(["nvidia-smi","--query-gpu=memory.used",
                                   "--format=csv,noheader,nounits"]).decode().split()
    return [int(x) for x in out if x.strip().lstrip("-").isdigit()]
def post(path, obj, timeout=1800):
    r=urllib.request.Request(B+path,data=json.dumps(obj).encode(),headers={"Content-Type":"application/json"})
    with urllib.request.urlopen(r,timeout=timeout) as resp: return json.load(resp)
def ntok(s): return len(post("/tokenize",{"content":s},300)["tokens"])

def calibrate(target):
    f=lambda n: " ".join(f"Log entry {i}: routine telemetry sample, nominal." for i in range(1,n))
    rate=ntok(f(400))/400.0; n=int(target/rate)
    for _ in range(4):
        t=ntok(f(n))
        if abs(t-target)<=max(300,int(0.03*target)): break
        n=int(n*target/max(t,1))
    return f(n), ntok(f(n))

class Peak(threading.Thread):
    def __init__(self): super().__init__(daemon=True); self.stop=False; self.peak=[0,0]
    def run(self):
        while not self.stop:
            try: g=gpu_used(); self.peak=[max(self.peak[0],g[0]),max(self.peak[1],g[1])]
            except Exception: pass
            time.sleep(0.2)

prompt, ptok = calibrate(PTOK)
res={"arm":ARM,"conc":CONC,"prompt_tokens":ptok,"npred":NPRED,"vram_before":gpu_used()}
results={}
def one(i):
    try:
        d=post("/completion",dict(prompt=prompt,n_predict=NPRED,cache_prompt=False,temperature=0.0),1800)
        results[i]=d.get("timings",{}).get("predicted_n",0)
    except Exception as e:
        results[i]={"err":repr(e)}

p=Peak(); p.start()
t0=time.time()
ths=[threading.Thread(target=one,args=(i,)) for i in range(CONC)]
for t in ths: t.start()
for t in ths: t.join()
p.stop=True; p.join()
res["wall_s"]=round(time.time()-t0,1)
res["vram_peak_mib"]=p.peak
res["free_mib"]=[TOTAL[0]-p.peak[0], TOTAL[1]-p.peak[1]]
res["min_free_ge_1gib"]= (res["free_mib"][0]>=1024 and res["free_mib"][1]>=1024)
res["errors"]=sum(1 for v in results.values() if isinstance(v,dict))
os.makedirs(os.path.dirname(OUT),exist_ok=True)
json.dump(res,open(OUT,"w"),indent=2)
print(json.dumps(res,indent=2))
