# Qwen stack — daily usage

One endpoint: **`http://localhost:8000`** (OpenAI-compatible). No "start the server" step —
**naming a model in a request loads it**; naming a different chat model swaps.

**Loading rules:** the four **chat** models are **mutually exclusive** — one on the GPUs at a time.
The **CPU embedder** (`qwen-embed`) is **always-resident and concurrent** — it coexists with whatever
chat model is loaded (zero VRAM). The **GPU embedder** (`qwen-embed-gpu`) is a chat-group member, so it
**does** swap the chat model out while it runs.

| model id | what | speed | aliases |
|---|---|---|---|
| `qwen-38-27b` | **27B dense + VISION** (Qwen3.8), 4090+5060Ti — **two quant profiles** (see below) | UD-Q6_K_M ~69 code · 57 RAG@32K · 37 @100K · plain Q6_K ~+8% ← quality dense, **sees images/video** | `qwen38`, `qwen38-27b`, `qwen36`, `qwen36-q6`, `qwen36-text` |
| `qwen-35b` | 35B MoE, 4090+5060Ti | **206 t/s** ← the fast one | `qwen36-35b` |
| `qwen-35b-batch` | 35B MoE, **4 parallel slots × 32k ctx** — for many small concurrent requests (classification/scoring) | same GGUF as `qwen-35b`, throughput not latency | `qwen36-35b-batch` |
| `qwen-38-flash-next` | **125B MoE + VISION** (Qwen3.8-Flash-Next), 4090+5060Ti+RAM (PLE on NVMe) | ~18.8 t/s ← biggest · ⚠️ ~150 s cold swap | `flash-next`, `qwen38-next` |
| `qwen-embed` | **Qwen3-Embedding-0.6B**, CPU, **always-on/concurrent** — light query embeds during chat | 1024-dim, zero VRAM | `embed`, `qwen3-embed` |
| `qwen-embed-gpu` | **same embedder on the 4090**, on-demand — **heavy bulk indexing** (evicts the chat model while it runs) | 1024-dim, full 1008 GB/s | `embed-gpu`, `qwen3-embed-gpu` |

`qwen-38-27b` **replaced the entire 3.6-27B dense line** (the old Q4 fast + Q6 quality entries) with one
Qwen3.8-27B multimodal server: weights span both GPUs (layer split), MTP drafting works *and* it
reads images/video (native VLM). All the legacy aliases (`qwen36`, `qwen36-q6`, `qwen36-text`) now land
on it, so existing clients keep working. It is mutually exclusive with the 35B and flash-next.

`qwen-35b-batch` is the **same 35B GGUF and placement** with `-np 4`. It exists as a separate entry (not
`-np 4` on `qwen-35b`) because `-c` is the *total* context divided across slots — flipping the main entry
to 4 slots would silently cut every caller's per-request context from 131072 → 32768. Total KV/VRAM is
identical either way. Use it for **many small requests at once** (grammar-constrained scoring); use plain
`qwen-35b` for normal chat where you want the full context window.

### 27B quant profiles — UD-Q6_K_M (quality) vs plain Q6_K (speed)
The 27B ships as **two swappable config profiles** (same aliases, same vision, same 131072 ctx — only the
underlying quant + tuning differ). Pick by editing which profile is `config.yaml`, then restart:

| profile | file | quant | split / draft | speed | when |
|---|---|---|---|---|---|
| **Q6_K_M** | `swap/config.yaml.q6km` | UD-Q6_K_M (imatrix, ~Q8-leaning quality) | `-ts 4,1`, MTP `n=5` | ~69 code · **57 RAG@32K** | quality-first (default choice) |
| **plain** | `swap/config.yaml.plain` | plain Q6_K (uniform) | `-ts 3,1`, MTP `n=4` | ~+8% faster decode | speed-first / A-B |

```bash
# switch to Q6_K_M (UD, quality)
cp ~/ai/qwen36/swap/config.yaml.q6km  ~/ai/qwen36/swap/config.yaml && sudo systemctl restart qwen-swap
# switch to plain Q6_K (speed)
cp ~/ai/qwen36/swap/config.yaml.plain ~/ai/qwen36/swap/config.yaml && sudo systemctl restart qwen-swap
# which is live?  ->  grep -q UD-Q6_K_M ~/ai/qwen36/swap/config.yaml && echo Q6_K_M || echo plain
```
Both certify **gates38 17/0**. The UD tuning (`4,1`/`n5`) was chosen for **code/RAG + vision**: `4,1` pushes
weight onto the 4090 (+7% short); `SPEC_NMAX=5` lifts RAG-depth tg +14% (AL 4.3→5.0). `5,1` is unsafe with
vision (GPU0 <1 GiB free).

**Vision:** send an image with the OpenAI `image_url` content part (data-URI or http URL):
```bash
curl -s localhost:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{"model":"qwen38",
  "messages":[{"role":"user","content":[{"type":"text","text":"What is in this image?"},
  {"type":"image_url","image_url":{"url":"data:image/png;base64,'"$(base64 -w0 pic.png)"'"}}]}],
  "max_tokens":2048}' | jq -r '.choices[0].message.content'
```

**Swap cost:** 27B (3.8) ~15–20 s · 35B ~16–25 s · flash-next **~150 s cold** (biggest model — batch its work).
Warm (page cache) is faster than cold. **First request after any load is ~15% slow** (graph warmup) —
not a regression.

---

## Embeddings / RAG

Same endpoint, OpenAI `/v1/embeddings`. **1024-dim, L2-normalized**, last-token pooling (Qwen3-Embedding).
Two placements of the *same* model — pick by workload:

| model id | where | VRAM | coexists with chat? | use for |
|---|---|---|---|---|
| `qwen-embed` | CPU | zero | **yes** (persistent) | live per-query embeds while you chat |
| `qwen-embed-gpu` | 4090 | ~7.4 GiB while running | **no** — evicts the chat model, reclaims VRAM on next chat | **bulk corpus indexing** |

```bash
# light query embed (runs alongside whatever chat model is loaded)
curl -s localhost:8000/v1/embeddings -H 'Content-Type: application/json' \
  -d '{"model":"qwen-embed","input":"how do I sort a list in python"}' | jq '.data[0].embedding | length'

# heavy bulk index on the 4090 (this swaps the chat model out for the duration)
curl -s localhost:8000/v1/embeddings -H 'Content-Type: application/json' \
  -d '{"model":"embed-gpu","input":["chunk 1","chunk 2","chunk 3"]}' | jq '.data | length'
```

- **RAG pattern:** embed the query on `qwen-embed` (CPU) and chat on `qwen38` — separate processes, so
  no queueing between them. Index the corpus once on `embed-gpu`, then serve at query time on `qwen-embed`.
- **Query prefix:** Qwen3-Embedding retrieves best with an **instruction prefix on the QUERY only** (not
  the documents), e.g. `Instruct: Given a search query, retrieve relevant passages\nQuery: <text>`.
- **One-off GPU embed without touching the config** (spare port, no restart):
  `NGL=99 DEV=0 PORT=8088 ~/ai/qwen36/run-embed.sh` — `DEV=0` 4090, `DEV=1` 5060 Ti, `NGL=99` all layers.

---

## Everyday

```bash
# what's loaded?
curl -s localhost:8000/healthz | jq

# talk to a model (loads/swaps automatically)
curl -s localhost:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{"model":"qwen-38-27b", "messages":[{"role":"user","content":"hello"}], "max_tokens":2048}' | jq -r '.choices[0].message.content'

# free both GPUs (nothing resident)
curl -s -X POST localhost:9000/api/models/unload

# list models
curl -s localhost:8000/v1/models | jq -r '.data[].id'

# watch VRAM live
watch -n1 nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
```

**max_tokens matters.** These models think before answering, and the reasoning eats the budget.
Use **≥2048** for the 27B/35B and **≥4096** for flash-next, or `content` comes back EMPTY while
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

# full stack check — exercises the chat-model swaps 27B→35B (~1 min warm).
# (Only checks flash-next is REGISTERED, not a swap-test — its ~150 s load would dominate; gate it
#  separately via gates38next.sh. Also does NOT swap-test qwen-35b-batch or the embedders.)
~/ai/qwen36/systemd/verify-boot.sh

# logs (llama-swap + router). Add -f to follow.
journalctl -u qwen-swap -u qwen-router -n 50 --no-pager -f

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
- A long pause on the first flash-next request — it's loading (**~150 s cold**; batch its work, pre-warm with `/load`).

---

## Regression suites (run if something feels off)

```bash
export PATH=$HOME/.local/bin:$PATH        # jq

# 27B (Qwen3.8) — via the router: alias dispatch, tool-budget guard, 100K
~/ai/qwen36/regress.sh

# 27B (3.8) / 35B — these hit a STANDALONE server, so free the GPUs first:
curl -s -X POST localhost:9000/api/models/unload
~/ai/qwen36/run-38-27b.sh &              # port 8080 (both GPUs, +vision)
~/ai/qwen36/gates38.sh ; kill %1         # 17 checks: dual-GPU residency, canary, MTP+mmproj, VISION, 100K

curl -s -X POST localhost:9000/api/models/unload
~/ai/qwen36/run-text-35b.sh &            # port 8082
~/ai/qwen36/gates35.sh ; kill %1

# flash-next (125B, ISOLATED binary at ~/ai/flashnext) — gate via the loaded llama-swap backend (:8085):
curl -s localhost:8000/load/flash-next                        # ~150 s cold load
~/ai/flashnext/gates38next.sh                                 # single-server gates (needs jq); MTP-identity/q8_0/perplexity are separate

# embedders (both hit /v1/embeddings directly — the CPU one on :8086, GPU one on :8088)
~/ai/qwen36/gates-embed.sh                                        # CPU qwen-embed  (:8086)
NGL=99 DEV=0 PORT=8088 ~/ai/qwen36/run-embed.sh &                 # GPU on the 4090
M=qwen-embed-gpu ~/ai/qwen36/gates-embed.sh http://127.0.0.1:8088 ; kill %1
```
`regress.sh [0]` asserts **every registered model has a live gate**, in both directions — add a
model without a `GATE_FOR` entry and it fails immediately. The stack currently registers **6 models**
(`qwen-38-27b`, `qwen-35b`, `qwen-35b-batch`, `qwen-38-flash-next`, `qwen-embed`, `qwen-embed-gpu`), each mapped.

---

## Knobs worth knowing (all env vars on the run scripts; restart qwen-swap to apply)

| knob | default | when to change |
|---|---|---|
| `MODEL` / `MMPROJ` (3.8) | plain Q6_K | absolute paths to the weights + projector; env-overridable. The `q6km` profile sets these to `models/Qwen38-27B-6_K_M/…UD-Q6_K_M.gguf`. |
| `TS` (27B/3.8) | 3,1 (plain) · **4,1 (UD)** | tensor-split proportion GPU0,GPU1. UD lands more on GPU1 at 3,1, so it uses **4,1** (weight→4090, +7% short). `5,1` unsafe with vision (GPU0 <1 GiB). If GPU0 OOMs, shift toward GPU1 — never past the 1 GiB floor. |
| `VISION` (3.8) | on | `off` drops `--mmproj` (text-only, frees the projector VRAM). MTP drafting and vision coexist (gate-verified) — no need to disable MTP for images. |
| `SPEC_NMAX` (27B/3.8) | 4 (plain) · **5 (UD)** | code/RAG lever. On UD, `5` = **+14% RAG@32K** (AL 4.3→5.0) at −0.7% short; `6` helps RAG only +1 and hurts short. Fits GPU0 with vision. |
| `SPEC_NMAX` (35B) | 3 | `4` is **+12% on RAG**, −12% creative. Try if your work is RAG-heavy. |
| `NMAX` (flash-next) | 2 | MTP draft depth (`--spec-draft-n-max`). 2 is unsloth's default; acceptance measured 0.6–0.9. Try 3 if RAG-heavy — full sweep pending (M5). GPU1 headroom is ~1 GiB, so watch for OOM when raising. |
| `NCM` / `TS` (flash-next) | 31 · 40,8 | `--n-cpu-moe` (experts to CPU) + tensor-split. **`TS=40,8` is load-bearing**: NCM offloads LOW layers' experts, `-sm layer` puts HIGH layers on GPU1, so the expert-heavy end must be steered to the 4090 or GPU1 OOMs. |
| `MTP` | on | `off` if you ever see repetition. Costs ~1.5× speed. |
| `CTX` | 131072 | max usable is the model's native **262144**, but only at `TS=2,1` (razor-thin: <1 GiB/GPU free; deep-needle-verified). `3,1`/`4,1` cap ~180K (GPU0 OOMs at 262144). Safe ≥1 GiB/GPU max ≈ 235–245K @ `2,1`. |
| `NP` (35B) | 1 · **4 (`qwen-35b-batch`)** | parallel slots. `-c` is TOTAL ctx split across slots, so `NP=4` = 32768/slot (VRAM-neutral). Raise for many small concurrent requests; leave at 1 for full-context chat. |
| `NGL` / `DEV` (embedder) | `0` (CPU) | `NGL=99` puts the embedder on GPU (`DEV=0` 4090, `DEV=1` 5060 Ti). The `qwen-embed-gpu` model sets `NGL=99 DEV=0 PORT=8088`. GPU embed uses ~7.4 GiB (8192 batch buffer) and swaps the chat model out — on-demand only. |
| `TOOL_MIN_BY_MODEL` (router) | flash-next: 4096 | raise if a model's tool calls truncate |

**KV stays q8_0 on the 27B/35B. Locked.** q4_0 saves ~1.75 GiB and silently breaks long-range
retrieval (2/5 vs 5/5 needles) — it benchmarks fine and guts the RAG path. (**Flash-Next is the one
exception — f16 KV**, since q8_0 asserts on its QSA path; q8_0 is verified clean on our build if ever needed.)

---

## What to actually watch for this week

The open questions can't be settled by another test — only by real use:

1. **Which model does your hand reach for?** That's the real measurement.
2. **Does flash-next's quality justify the ~150 s cold swap-in?** If you stop reaching for it, that's
   the answer — batch it, or drop it.
3. **Does flash-next's PLE paging ever stall long generations?** Its 51.2B-element PLE lives on NVMe
   (lazy mmap, ~2 rows/token); watch NVMe read throughput during real long-context use.
4. **Is `NMAX=2` right?** Acceptance measured 0.6–0.9, so `3` might pay on RAG-heavy work — the full
   n-max sweep is deferred to M5.
