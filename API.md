# Qwen local inference — API reference

One OpenAI-compatible endpoint, four models, **one resident at a time**. Everything below is served
from a single host; there is no cloud dependency and no API key check (bind is localhost by default).

```
client ──▶ router :8000 ──▶ llama-swap :9000 ──▶ llama-server (one of four)
           (tool guard,        (model manager:      (the actual model,
            system prompt,       load / swap /        MTP + q8_0 KV,
            /load /unload)       mutual exclusion)    131072 ctx)
```

- **Build against `http://localhost:8000`.** It is OpenAI-compatible (`/v1/chat/completions`,
  `/v1/models`). The `:9000` layer is llama-swap's own API; you normally don't call it directly.
- **No "start a server" step.** Naming a model in a request loads it; naming a different one swaps it.
  Only ONE model is ever resident (hard mutual-exclusion — see *Concurrency* below).

---

## Models

| model id | alias | arch / placement | decode speed | load (warm) | resident VRAM |
|---|---|---|---|---|---|
| `qwen-38-27b` | `qwen38`, `qwen38-27b`, `qwen36`, `qwen36-q6`, `qwen36-text` | 27B dense **+ VISION** (Qwen3.8, qwen3_5 arch) **Q6_K**, GPU0+GPU1 | ~73 code / 48 @100K | ~15–20 s | GPU0 21.8 / GPU1 9.4 GiB |
| `qwen-35b` | `qwen36-35b` | 35B-A3B **MoE**, GPU0+GPU1 | **~206 t/s** | ~16–25 s | GPU0 ~21.4 / GPU1 ~3.0 GiB |
| `qwen-122b` | `qwen35-122b` | 122B-A10B **MoE**, GPU0+GPU1+RAM | ~37–40 t/s | ~40–90 s | GPU0+GPU1 full + ~15 GiB RAM |

You may address a model by **id or alias** everywhere (`model` field, `/load`, `/upstream`).

**Character, for choosing:** `qwen-35b` fastest · `qwen-122b` smartest · `qwen-38-27b` the quality
dense **and the only one that sees images/video** (native VLM). It replaced the retired 3.6-27B dense
line (Q4 + Q6); all their legacy aliases now resolve to it.

**Vision:** `qwen-38-27b` accepts OpenAI `image_url` content parts (data-URI or http URL) — images and
video. The other two are text-only. See "Vision requests" below.

All three: **131072 context**, **q8_0 KV** (locked), **MTP speculative decoding on**, thinking on.

---

## Endpoints (router, :8000)

### `POST /v1/chat/completions` — OpenAI chat
Standard OpenAI schema. `stream: true` supported (SSE passthrough). The router adds two behaviors:
1. **System-prompt injection** — if you don't send a `system`/`developer` message first, it prepends
   `"You are Qwen, created by Alibaba Cloud. You are a helpful assistant."` (disable with
   `INJECT_SYSTEM=0` on the router).
2. **Tool-budget guard** — see *Gotcha: tool calls* below.

```bash
curl -s localhost:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model":"qwen36-q6",
  "messages":[{"role":"user","content":"Explain a B-tree in one sentence."}],
  "max_tokens":2048}' | jq -r '.choices[0].message.content'
```

### `GET|POST /load[/<model>]` — pre-warm a model **(no dummy chat needed)**
Loads (and swaps to) a model, **blocking until it is resident**, then returns. Use it to pay the
swap cost up front instead of on a user's first request.

```bash
curl -s localhost:8000/load/qwen38                    # by alias or id
curl -s -X POST localhost:8000/load -d '{"model":"qwen-122b"}'
```
```json
{"loaded":"qwen38","upstream_status":200,"running":["qwen-38-27b"],"seconds":16.4}
```
- `200` loaded · `404` unknown model (`loaded:null`) · `502/504` load failed.
- Idempotent: loading the already-resident model returns in ~0 s.

### `GET|POST /unload` — free all GPUs
```bash
curl -s localhost:8000/unload      # -> {"unloaded":true,"upstream_status":200}
```
Releases VRAM **and** the 122B's RAM slice (process dies → all tiers reclaimed). Idle (nothing
resident, GPUs at 0) is the **normal** resting state, not an error.

### `GET /healthz` — liveness + what's loaded
```json
{"router":"ok","backend":{"swap":200,"running":["qwen-38-27b"]}}
```
`running: []` = nothing loaded (normal). `router:"degraded"` + 503 = llama-swap unreachable.

### `GET /v1/models` — list registered models (ids, not aliases)
### passthrough — any other path proxies to llama-swap
e.g. `GET /upstream/<model>/health`, `POST /upstream/<model>/completion` (raw llama.cpp completion
with `timings`, used by the gate suites).

### Vision requests (`qwen-38-27b` only)
Send an image/video frame as an OpenAI `image_url` content part — data-URI or http URL. The model
must be `qwen-38-27b` (or any of its aliases); the 35B/122B are text-only and will ignore the image.
```bash
IMG=$(base64 -w0 photo.png)
curl -s localhost:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model":"qwen38","max_tokens":2048,
  "messages":[{"role":"user","content":[
    {"type":"text","text":"Describe this image."},
    {"type":"image_url","image_url":{"url":"data:image/png;base64,'"$IMG"'"}}]}]}' \
| jq -r '.choices[0].message.content'
```
Answers still land in `content` (+ `reasoning_content` when thinking). The vision tower (`mmproj-F16`)
is resident whenever the 27B is loaded — no separate model or endpoint.

---

## Key metrics anyone building on this needs

**Latency budget**
- **Cold model swap is the dominant cost.** Loading/swapping blocks the first request: ~11–25 s for
  the 27B/35B, **~40–90 s for the 122B** (48 GiB across VRAM+RAM). Warm (page-cache) is the low end;
  first-ever-after-boot is the high end. **Pre-warm with `/load`** if a user shouldn't wait.
- **First request after any load is ~15% slower** (CUDA graph warmup). Not a regression.
- **Set the proxy/client read timeout to `None`/infinite.** The router already does; a cold swap or a
  100K-token generation legitimately runs for minutes. Connect timeout ~10 s is fine.

**Throughput (decode, tokens/s, MTP on)**
| model | code | creative | @100K ctx |
|---|---|---|---|
| qwen-38-27b (Q6_K, +vision) | ~73 | — | ~48 |
| qwen-35b | ~206 | — | — |
| qwen-122b | ~37–40 | (MTP net-negative on creative — see below) | — |

MTP draft acceptance on the 27B code path measured **0.845, AL 4.31** — and it holds *with the mmproj
vision tower loaded* (gate-verified), so image capability costs the text path no speed.

Speed is content-dependent because of MTP: high draft acceptance (code, acc ~0.86–0.90, AL ~4.3–4.5)
runs much faster than low-acceptance creative text. Prompt-processing (prefill) is separate and far
faster (100K prefills at ~1600 t/s on the Q6).

**Context**
- **131072 tokens** for all four. Prefix cache is on: a repeated prompt prefix reprocesses in ~7% of
  the first-time cost (e.g. 1990 ms → 140 ms). Reuse prefixes for cheap multi-turn.

**Output budget — the #1 integration bug**
- These models **think before answering**; reasoning consumes the token budget. Set
  **`max_tokens` ≥ 2048** (27B/35B) / **≥ 4096** (122B), or the answer truncates and `content` comes
  back **EMPTY** while everything went to `reasoning_content`.
- **Read both fields.** With thinking on, the visible answer is in `choices[0].message.content` and
  the chain-of-thought is in `choices[0].message.reasoning_content` — the latter can be non-empty
  while the former is empty if you under-budgeted.

**Concurrency**
- **One model resident at a time; one request at a time** (`--parallel 1`, required by MTP). Concurrent
  requests **queue** — they are serialized, not run in parallel. Design clients accordingly (a burst of
  N calls takes ~N× a single call, plus a swap if they target different models).
- Cross-model calls interleaved from multiple clients will **thrash the swap** (evict/reload each time).
  Batch by model, or pin one model for a workload.

**Tool calls**
- Standard OpenAI `tools`/`tool_calls`. The router **auto-raises `max_tokens` to a floor when `tools`
  are present** (512 for 27B/35B, **4096 for the 122B**) so the tool JSON can't truncate mid-argument
  (a truncated call → llama.cpp 500). Your own `max_tokens` is used if already above the floor.

**Reliability knobs (defaults are good)**
- KV cache is **q8_0 everywhere and locked** — do not switch to q4_0 (it silently breaks long-range
  retrieval; benchmarks fine, guts RAG).
- Every model has a live regression gate (`gates38.sh`, `gates35.sh`, `gates122.sh`, plus the
  router-level `regress.sh`); `regress.sh [0]` asserts every registered model still maps to one.

---

## Minimal client setup

```python
from openai import OpenAI
c = OpenAI(base_url="http://localhost:8000/v1", api_key="local")   # key is ignored
# optional: pre-warm so the first real call is instant
import httpx; httpx.get("http://localhost:8000/load/qwen36-q6", timeout=120)
r = c.chat.completions.create(model="qwen36-q6",
      messages=[{"role":"user","content":"hi"}], max_tokens=2048)
print(r.choices[0].message.content)              # answer
print(r.choices[0].message.reasoning_content)    # thinking (may hold the bulk of the tokens)
```
```bash
export OPENAI_BASE_URL=http://localhost:8000/v1
export OPENAI_API_KEY=local
```

---

## Operational quick-reference

```bash
systemctl is-active qwen-swap qwen-router        # units
~/ai/qwen36/systemd/verify-boot.sh               # exercises all models (~1 min warm)
journalctl -u qwen-swap -u qwen-router -n 50     # router + swap logs
curl -sN localhost:9000/logs/stream              # llama-swap live log
sudo systemctl restart qwen-router               # apply proxy.py changes (does NOT unload the model)
sudo systemctl restart qwen-swap                 # apply config.yaml / run-script changes (DOES unload)
```
The model server's own timing logs (draft acceptance, t/s) are not in the journal by default
(`logToStdout: proxy`); set it to `both` in `swap/config.yaml` if you need them.
