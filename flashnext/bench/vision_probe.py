#!/usr/bin/env python3
"""Vision VRAM probe (round 2): peak per-GPU VRAM while the mmproj encoder processes a
MAX-RESOLUTION image, (a) alone and (b) layered on top of a ~100K-token text context.
Polls nvidia-smi in a side thread during each request and records the max. Writes JSON to $OUT."""
import os, sys, json, time, base64, threading, subprocess, urllib.request

B    = os.environ.get("BASE", "http://127.0.0.1:8095")
ARM  = os.environ.get("ARM", "vprobe")
OUT  = os.environ.get("OUT", f"/home/rohit/ai/qwen36/flashnext/bench/{ARM}.json")
IMG  = os.environ.get("IMG", "/home/rohit/ai/qwen36/flashnext/bench/maxres.png")
SYS  = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."

def gpu_used():
    out = subprocess.check_output(["nvidia-smi","--query-gpu=memory.used",
                                   "--format=csv,noheader,nounits"]).decode().split()
    return [int(x) for x in out if x.strip().lstrip("-").isdigit()]

def post(path, obj, timeout=1800):
    r = urllib.request.Request(B+path, data=json.dumps(obj).encode(),
                               headers={"Content-Type":"application/json"})
    with urllib.request.urlopen(r, timeout=timeout) as resp:
        return json.load(resp)
def ntok(s): return len(post("/tokenize", {"content": s}, 300)["tokens"])

class Peak(threading.Thread):
    def __init__(self): super().__init__(daemon=True); self.stop=False; self.peak=[0,0]
    def run(self):
        while not self.stop:
            try:
                g=gpu_used()
                self.peak=[max(self.peak[0],g[0]), max(self.peak[1],g[1])]
            except Exception: pass
            time.sleep(0.25)

b64 = base64.b64encode(open(IMG,"rb").read()).decode()
data_uri = "data:image/png;base64," + b64
res = {"arm": ARM, "img": IMG, "img_bytes": len(b64)}

def vision_req(text, max_tokens=64):
    return {"model":"flash-next","max_tokens":max_tokens,"temperature":0.2,"stream":False,
            "messages":[{"role":"system","content":SYS},
                        {"role":"user","content":[{"type":"text","text":text},
                                                   {"type":"image_url","image_url":{"url":data_uri}}]}]}

# baseline (model loaded, pre-image)
res["vram_baseline_mib"] = gpu_used()

# (a) max-res image ALONE
try:
    p = Peak(); p.start()
    d = post("/v1/chat/completions", vision_req("Describe this image in one sentence."))
    p.stop=True; p.join()
    res["image_alone"] = {"peak_vram_mib": p.peak,
                          "prompt_tokens": d.get("usage",{}).get("prompt_tokens"),
                          "answer": (d["choices"][0]["message"].get("content") or "")[:80]}
except Exception as e:
    res["image_alone_err"] = repr(e)

# (b) max-res image AFTER ~100K-token text context (worst case: vision encoder + 100K KV)
try:
    filler = " ".join(f"Log entry {i}: routine telemetry sample, nominal, no action required."
                      for i in range(1, 5700))
    ft = ntok(filler)
    p = Peak(); p.start()
    d = post("/v1/chat/completions", vision_req(filler + "\n\nDescribe the attached image in one sentence.", max_tokens=64))
    p.stop=True; p.join()
    res["image_after_100k"] = {"peak_vram_mib": p.peak,
                               "text_tokens": ft,
                               "prompt_tokens": d.get("usage",{}).get("prompt_tokens"),
                               "answer": (d["choices"][0]["message"].get("content") or "")[:80]}
except Exception as e:
    res["image_after_100k_err"] = repr(e)

os.makedirs(os.path.dirname(OUT), exist_ok=True)
json.dump(res, open(OUT,"w"), indent=2)
print(json.dumps(res, indent=2))
