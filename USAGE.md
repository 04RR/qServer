# Qwen stack — daily usage

One endpoint: **`http://localhost:8000`** (OpenAI-compatible). Three models, **one loaded at a time**.
No "start the server" step — **naming a model in a request loads it**; naming a different one swaps.

| model id | what | speed | aliases |
|---|---|---|---|
| `qwen-38-27b` | **27B dense + VISION** (Qwen3.8) Q6_K, 4090+5060Ti | ~73 t/s code · 48 t/s @100K ← quality dense, **sees images/video** | `qwen38`, `qwen38-27b`, `qwen36`, `qwen36-q6`, `qwen36-text` |
| `qwen-35b` | 35B MoE, 4090+5060Ti | **206 t/s** ← the fast one | `qwen36-35b` |
| `qwen-122b` | 122B MoE, 4090+5060Ti+RAM | 37–40 t/s ← the smart one | `qwen35-122b` |

`qwen-38-27b` **replaced the entire 3.6-27B dense line** (the old Q4 fast + Q6 quality entries) with one
Qwen3.8-27B multimodal server: Q6_K weights span both GPUs (layer split), MTP drafting works *and* it
reads images/video (native VLM). All the legacy aliases (`qwen36`, `qwen36-q6`, `qwen36-text`) now land
on it, so existing clients keep working. It is mutually exclusive with the 35B/122B.

**Vision:** send an image with the OpenAI `image_url` content part (data-URI or http URL):
```bash
curl -s localhost:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{"model":"qwen38",
  "messages":[{"role":"user","content":[{"type":"text","text":"What is in this image?"},
  {"type":"image_url","image_url":{"url":"data:image/png;base64,'"$(base64 -w0 pic.png)"'"}}]}],
  "max_tokens":2048}' | jq -r '.choices[0].message.content'
```

**Swap cost:** 27B (3.8) ~15–20 s · 35B ~16–25 s · 122B **~40–50 s** (48 GiB across three tiers).
Warm (page cache) is faster than cold. **First request after any load is ~15% slow** (graph warmup) —
not a regression.

---

## Everyday

```bash
# what's loaded?
curl -s localhost:8000/healthz | jq

# talk to a model (loads/swaps automatically)
curl -s localhost:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{"model":"qwen38", "messages":[{"role":"user","content":"hello"}], "max_tokens":2048}' | jq -r '.choices[0].message.content'

# free both GPUs (nothing resident)
curl -s -X POST localhost:9000/api/models/unload

# list models
curl -s localhost:8000/v1/models | jq -r '.data[].id'

# watch VRAM live
watch -n1 nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
```

**max_tokens matters.** These models think before answering, and the reasoning eats the budget.
Use **≥2048** for the 27B/35B and **≥4096** for the 122B, or `content` comes back EMPTY while
`reasoning_content` holds everything. (The router auto-raises it for *tool* calls only.)

**Answers live in two fields** when thinking is on:
```bash
jq -r '.choices[0].message.reasoning_content'   # the thinking
jq -r '.choices[0].message.content'             # the answer
```

---

## Point real tools at it

Anything OpenAI-compatible. Base URL `http://localhost:8000/v1`, any API key string.

```bash
export OPENAI_BASE_URL=http://localhost:8000/v1
export OPENAI_API_KEY=local
```
```python
from openai import OpenAI
c = OpenAI(base_url="http://localhost:8000/v1", api_key="local")
r = c.chat.completions.create(model="qwen-35b",
        messages=[{"role":"user","content":"hi"}], max_tokens=2048)
print(r.choices[0].message.content)
```
Aider: `aider --openai-api-base http://localhost:8000/v1 --openai-api-key local --model openai/qwen-35b`

---

## Health / troubleshooting

```bash
# units
systemctl is-active qwen-swap qwen-router

# full stack check — exercises ALL THREE models (~2.5 min cold, ~1 min warm)
~/ai/qwen36/systemd/verify-boot.sh

# logs (llama-swap + router). Add -f to follow.
journalctl -u qwen-swap -u qwen-router -n 50 --no-pager

# the model server's OWN logs (timings, draft acceptance) are NOT in the journal by default:
#   llama-swap's logToStdout defaults to "proxy". Set logToStdout: "both" in swap/config.yaml
#   + sudo systemctl restart qwen-swap  (this unloads whatever is resident).
# llama-swap's own live log:
curl -sN localhost:9000/logs/stream

# restart (needed after editing swap/config.yaml, proxy.py, or a run script)
sudo systemctl restart qwen-swap qwen-router

# reach the loaded model directly, bypassing the router
curl -s localhost:9000/upstream/qwen-38-27b/health
```

**Normal things that look wrong:**
- `running: []` and **both GPUs at 0 MiB** — correct resting state (load-on-demand). Not a failure.
- `systemctl is-system-running` → `degraded` — a pre-existing unrelated unit, not ours.
- A long pause on the first 122B request — it's loading 48 GiB. Up to ~50 s warm, minutes cold.

---

## Regression suites (run if something feels off)

```bash
export PATH=$HOME/.local/bin:$PATH        # jq

# 27B (Qwen3.8) — via the router: alias dispatch, tool-budget guard, 100K
~/ai/qwen36/regress.sh

# 27B (3.8) / 35B / 122B — these hit a STANDALONE server, so free the GPUs first:
curl -s -X POST localhost:9000/api/models/unload
~/ai/qwen36/run-38-27b.sh &              # port 8080 (both GPUs, +vision)
~/ai/qwen36/gates38.sh ; kill %1         # 17 checks: dual-GPU residency, canary, MTP+mmproj, VISION, 100K

curl -s -X POST localhost:9000/api/models/unload
~/ai/qwen36/run-text-35b.sh &            # port 8082
~/ai/qwen36/gates35.sh ; kill %1

curl -s -X POST localhost:9000/api/models/unload
~/ai/qwen36/run-text-122b.sh &           # port 8084
~/ai/qwen36/gates122.sh ; kill %1
```
`regress.sh [0]` asserts **every registered model has a live gate**, in both directions — add a
4th model without a gate and it fails immediately.

---

## Knobs worth knowing (all env vars on the run scripts; restart qwen-swap to apply)

| knob | default | when to change |
|---|---|---|
| `TS` (27B/3.8) | 3,1 | tensor-split proportion GPU0,GPU1. Measured optimal: GPU0 free ~2.7G / GPU1 free ~6.9G. If GPU0 OOMs, shift MORE to GPU1 (`2,1`) — never toward GPU0 (binding card). |
| `VISION` (3.8) | on | `off` drops `--mmproj` (text-only, frees the projector VRAM). MTP drafting and vision coexist (gate-verified) — no need to disable MTP for images. |
| `SPEC_NMAX` (35B) | 3 | `4` is **+12% on RAG**, −12% creative. Try if your work is RAG-heavy. |
| `SPEC_NMAX` (122B) | 3 | **Do not raise past 6 — it won't boot** (draft buffers OOM GPU0's 602 MiB). 6 is +10% code but makes creative 27% *slower than MTP off*. |
| `MTP` | on | `off` if you ever see repetition. Costs ~1.5× speed. |
| `CTX` | 131072 | lower only if you hit VRAM trouble |
| `TOOL_MIN_BY_MODEL` (router) | 122B: 4096 | raise if a model's tool calls truncate |

**KV stays q8_0 everywhere. Locked.** q4_0 saves ~1.75 GiB and silently breaks long-range
retrieval (2/5 vs 5/5 needles) — it benchmarks fine and guts the RAG path.

---

## What to actually watch for this week

The open questions can't be settled by another test — only by real use:

1. **Which model does your hand reach for?** That's the real measurement.
2. **Does the 122B's quality justify the ~45 s swap?** If you stop reaching for it, that's the answer.
3. **Does IQ3_S ever feel mushy on long reasoning?** Its `gate`/`up` are 2-bit. No synthetic probe
   could detect harm (the 35B aced 93% of 73 items — nothing discriminates). **If a real task
   disappoints, THAT task is the discriminating item** — save the prompt; it earns the 73 GiB
   Q4_K_XL download with something concrete to compare against.
4. **Is n_max=3 right for your creative work?** Chosen because n=6 made creative net-negative.
