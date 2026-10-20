# Qwen local inference — API reference

One OpenAI-compatible endpoint, several models, **one resident at a time**. Everything below is served
from a single host; there is no cloud dependency and no API key check, so the router and every llama-server bind 127.0.0.1 only.

```
client ──▶ router :8000 ──▶ llama-swap :9000 ──▶ llama-server (one at a time)
           (tool guard,        (model manager:      (the actual model,
            system prompt,       load / swap /        MTP+q8_0 KV on 27B/35B,
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
| `qwen-38-27b` | `qwen38`, `qwen38-27b`, `qwen36`, `qwen36-q6`, `qwen36-text` | 27B dense **+ VISION** (Qwen3.8, qwen3_5 arch), GPU0+GPU1 — profiles: **UD-Q6_K_M** `-ts 4,1`/n5 or **plain Q6_K** `-ts 3,1`/n4 | UD ~69 code / 57 RAG@32K / 37 @100K · plain ~+8% | ~15–20 s | UD: GPU0 22.7 / GPU1 8.9 · plain: GPU0 21.8 / GPU1 9.4 GiB |
| `qwen-35b` | `qwen36-35b` | 35B-A3B **MoE**, GPU0+GPU1 | **~206 t/s** | ~16–25 s | GPU0 ~21.4 / GPU1 ~3.0 GiB |
| `qwen-38-flash-next` | `flash-next`, `qwen38-next` | 125B-A6B **MoE + VISION** (Qwen3.8-Flash-Next, qwen4exp; PLE on NVMe, **MTP off**) | ~17–18 t/s · ~820 t/s 100K-prefill | ⚠️ **~150 s cold** | GPU0 ~23 / GPU1 ~14 GiB + ~34 GiB mmap |

You may address a model by **id or alias** everywhere (`model` field, `/load`, `/upstream`).

**Character, for choosing:** `qwen-35b` fastest · `qwen-38-flash-next` biggest/smartest (**but ~150 s to
swap in — batch its work, don't interleave it turn-by-turn**) · `qwen-38-27b` the quality dense. **Both the
27B and Flash-Next see images/video** (native VLMs); the 27B replaced the retired 3.6-27B dense line
(Q4 + Q6) and carries its legacy aliases.

**Vision:** `qwen-38-27b` **and `qwen-38-flash-next`** accept OpenAI `image_url` content parts (data-URI or
http URL) — images and video. `qwen-35b` is text-only. See "Vision requests" below.

All: **131072 context**, **thinking is ALWAYS ON — clients cannot disable it.** The router (`FORCE_THINKING`,
default on) overrides `chat_template_kwargs.enable_thinking=false`, drops a top-level `enable_thinking=false`,
remaps a non-standard `reasoning_effort` (e.g. `"none"`) to `"low"`, and strips the `<|think_off|>` soft
switch. The 27B/35B use **MTP speculative decoding** + **q8_0 KV** (locked).

**No output caps (default).** The router **strips `max_tokens` / `max_completion_tokens` / `n_predict`** from
every chat request (`NO_OUTPUT_CAPS`, default on) so an agentic turn runs to its natural stop (EOS) or the
context limit — never truncated by a client budget. **Do not rely on `max_tokens`; it is ignored.** A client
that genuinely wants a **bounded** output (e.g. datagen) opts back in **per request** with the header
**`X-Respect-Max-Tokens: 1`**; only on that path is the cap honoured *and* the per-model tool/thinking floor
applied (512 for 27B/35B, 4096 for Flash-Next, never lowering a larger client value).

**Sampling defaults** match each model's Qwen card: 27B & Flash-Next(+batch) run temp 1.0 / top_p 0.95 /
top_k 20 / min_p 0 (the 3.8 thinking profile). **`qwen-35b`/`-batch` run temp 0.6** — deliberately the
Qwen3.6 card's *"precise coding"* thinking profile (its *general* thinking profile is temp 1.0 **with**
presence_penalty 1.5 together), not a mis-set against the 3.8 temp-1.0 card.

**Flash-Next runs MTP OFF** (its decode is CPU-compute-bound — speculation ~triples CPU expert work for
no net gain; dropping it frees ~4 GiB on GPU1 for `-ub 3072`) and **f16 KV** (q8_0 asserts on its QSA
path). It uses the mainline llama.cpp build, `-t 16`, and `-ctxcp 8` context checkpoints.

---

## Endpoints (router, :8000)

### `POST /v1/chat/completions` — OpenAI chat
Standard OpenAI schema. `stream: true` supported (SSE passthrough) — **stream for long turns** (see
*Output* below). The router adds these behaviors:
1. **Output-cap strip** — `max_tokens`/`max_completion_tokens`/`n_predict` are removed by default
   (`NO_OUTPUT_CAPS`); opt out per request with `X-Respect-Max-Tokens: 1`. See *Output* below.
2. **Force-thinking** — thinking is always on; client disables are neutralized (see above).
3. **System-prompt injection** — if you don't send a `system`/`developer` message first, it prepends
   `"You are Qwen, created by Alibaba Cloud. You are a helpful assistant."` (disable with
   `INJECT_SYSTEM=0` on the router).
4. **Tool-budget floor** — on the `X-Respect-Max-Tokens:1` path only; see *Gotcha: tool calls* below.

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
curl -s -X POST localhost:8000/load -d '{"model":"flash-next"}'   # ~150 s cold — pre-warm before a user waits
```
```json
{"loaded":"qwen38","upstream_status":200,"running":["qwen-38-27b"],"seconds":16.4}
```
- `200` loaded · `404` unknown model (`loaded:null`) · `502/504` load failed.
- Idempotent: loading the already-resident model returns in ~0 s.

### `GET|POST /unload/<model>` — unload one model, leave the rest
```bash
curl -s -X POST localhost:8000/unload/qwen-35b   # by id or alias
```
```json
{"unloaded":"qwen-35b","upstream_status":200,"running":["qwen-38-27b"]}
```
- `200` unloaded (also when that model wasn't loaded: a no-op) · `404` unknown model (`unloaded:null`) ·
  `502/504` llama-swap failed or didn't answer.
- **Use this from an application.** It frees only the model you name, so it can't evict a model another
  client is in the middle of using.

### `GET|POST /unload` — free all GPUs
```bash
curl -s localhost:8000/unload      # -> {"unloaded":true,"upstream_status":200}
```
Evicts **every** loaded model, whoever loaded it — an operator action. Releases VRAM **and** Flash-Next's
~34 GiB RAM/mmap tier (process dies → all tiers reclaimed). Idle (nothing resident, GPUs at 0) is the
**normal** resting state, not an error.

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
must be `qwen-38-27b` or `qwen-38-flash-next` (or their aliases); `qwen-35b` is text-only and will ignore the image.
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
  the 27B/35B, **~150 s for Flash-Next** (cold 29 GiB VRAM read + lazy PLE/expert faulting off NVMe).
  **Pre-warm with `/load`**, and **batch Flash-Next work** — interleaving it with other models pays ~150 s a switch.
- **First request after any load is ~15% slower** (CUDA graph warmup). Not a regression.
- **Set the proxy/client read timeout to `None`/infinite.** The router already does; a cold swap or a
  100K-token generation legitimately runs for minutes. Connect timeout ~10 s is fine.

**Throughput (decode, tokens/s; MTP on for the 27B/35B, OFF for Flash-Next)**
| model | code (short) | RAG@32K | @100K ctx | @244K ctx |
|---|---|---|---|---|
| qwen-38-27b **UD-Q6_K_M** (`4,1`/n5, +vision) | ~69 | ~57 | ~37 | ~26 |
| qwen-38-27b **plain Q6_K** (`3,1`/n4, +vision) | ~77 | ~50 | ~48 | — |
| qwen-35b | ~206 | — | — | — |
| qwen-38-flash-next (MTP off, f16, `-ub 3072`, `-t 16`) | ~17–18 | — | ~16.5–18 (≈flat) | — |

Two 27B quant profiles (toggle via `swap/config.yaml.q6km` / `.plain`): **UD-Q6_K_M** is imatrix
(quality) but ~8–10% slower — intrinsic to the mixed-bit-width dequant, *not* a split artifact (re-tuning
`TS` recovers only ~2%). It's tuned `-ts 4,1`/`SPEC_NMAX=5` for code/RAG; **plain Q6_K** is uniform and faster.

MTP draft acceptance on the 27B code path measured **0.83–0.89, AL 4.3–5.0** (AL 5.0 at `n=5`) — and it holds
*with the mmproj vision tower loaded* (gate-verified), so image capability costs the text path no speed.

Speed is content-dependent because of MTP: high draft acceptance (code, acc ~0.83–0.89) runs ~2× faster
than low-acceptance creative text (acc ~0.35). Decode also halves from short → 244K ctx (growing KV read).
Prompt-processing (prefill) is separate and far faster (~1.1–1.2K t/s at 100K).

**Flash-Next is the exception** — it runs **MTP off**, so its decode is content-*independent* (~17–18 t/s
regardless of acceptance; speculation was a net loss on this CPU-compute-bound expert path). Dropping MTP
freed ~4 GiB on GPU1 for a big prefill ubatch (`-ub 3072`): **100K prefill ~820 t/s** (~4× the old-fork
MTP-on config). `-t 16` leaves the persistent CPU embedder its 8 cores (16+8=24); `-t 24` collapses to
~5 t/s under concurrent embedding. `-ctxcp 8` reuses one prefilled long context across turns (~1.1 GiB RSS).
A max-resolution image after a ~100K text context still leaves ~614 MiB free on GPU0 (above the 512 floor),
so the `TS=40,8` split holds for vision.

**Max context:** native 262144 is reachable only at `TS=2,1` (razor-thin, <1 GiB/GPU free; deep-needle-verified);
`3,1`/`4,1` cap ~180K. Safe ≥1 GiB/GPU ≈ 235–245K @ `2,1`.

**Context**
- **131072 tokens** for all. Prefix cache is on: a repeated prompt prefix reprocesses in ~7% of
  the first-time cost (e.g. 1990 ms → 140 ms). Reuse prefixes for cheap multi-turn.

**Output — no caps (agentic default)**
- The router **strips `max_tokens`/`max_completion_tokens`/`n_predict`** from every chat request, so a
  turn runs to its natural stop — **your budget is ignored; you no longer need to set one.** This fixed
  the old #1 integration bug (thinking eats a small budget → `content` empty, everything in
  `reasoning_content`). Generation stops cleanly at EOS or the 131072 context limit (context shift is off).
- **Want a bounded output?** Send header **`X-Respect-Max-Tokens: 1`** and your `max_tokens` is honoured
  (then floored to 512 / 4096 only if a tool/thinking floor applies). Use this for datagen, not agents.
- **Stream long turns** (`stream: true`): a streamed request gets a tight per-chunk read deadline; a
  non-streaming one holds the socket for the whole generation (up to a 7200s read timeout).
- **Agents: use the main entries** (`qwen-38-27b`, `qwen-35b`, `qwen-38-flash-next`) — **131072 context**.
  The `-batch` entries split context across 4 slots = **32768/slot**; they are for bulk throughput, not
  long agentic turns.
- **Read both fields.** With thinking on, the visible answer is in `choices[0].message.content` and
  the chain-of-thought is in `choices[0].message.reasoning_content`.

**Concurrency**
- **One model resident at a time; one request at a time** (`--parallel 1` on the interactive entries;
  MTP requires it on the 27B/35B, and the Flash-Next main entry keeps it for latency). Concurrent
  requests **queue** — they are serialized, not run in parallel. Design clients accordingly (a burst of
  N calls takes ~N× a single call, plus a swap if they target different models). For throughput, use the
  **`qwen-38-flash-next-batch`** entry (`--parallel 4`, MTP off, text-only).
- Cross-model calls interleaved from multiple clients will **thrash the swap** (evict/reload each time).
  Batch by model, or pin one model for a workload.

**Tool calls**
- Standard OpenAI `tools`/`tool_calls`. By default the output cap is stripped, so a tool call can't
  truncate mid-argument (a truncated call → llama.cpp 500) — the turn runs unbounded. On the
  `X-Respect-Max-Tokens:1` path, where your cap is honoured, the router instead **raises `max_tokens`
  to a floor when `tools` are present** (512 for 27B/35B, **4096 for Flash-Next**), using your own value
  if already above the floor.

**Reliability knobs (defaults are good)**
- KV cache is **q8_0 everywhere and locked** — do not switch to q4_0 (it silently breaks long-range
  retrieval; benchmarks fine, guts RAG).
- Every model has a live regression gate (`gates38.sh`, `gates35.sh`, `gates38next.sh`, plus the
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
