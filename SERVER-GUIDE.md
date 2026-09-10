# Qwen Server — User Guide

Everything you need to **use** this inference server. If you can make an HTTP request or point an
OpenAI client at a URL, you can use it. You do not need to start, build, or configure anything.

---

## TL;DR

- **One endpoint:** `http://localhost:8000` — OpenAI-compatible.
- **Any API key string works** (there is no auth; use `local` or anything).
- **No "start the server" step.** Naming a model in a request loads it automatically. Naming a
  *different* chat model swaps it in (the previous one is evicted).
- **Pick a model by its `model` field.** The four chat models are mutually exclusive (one at a
  time); the CPU embedder is always available alongside whatever chat model is loaded.
- **Give it room to think:** set `max_tokens` to **≥2048** (≥4096 for flash-next), or the answer can
  come back empty. See [max_tokens](#max_tokens-the-one-gotcha-that-bites-everyone).

```bash
curl -s localhost:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "qwen-35b",
  "messages": [{"role":"user","content":"Write a haiku about GPUs."}],
  "max_tokens": 2048
}' | jq -r '.choices[0].message.content'
```

---

## The models

Send one of these as the `model` field. Any alias in the last column works too.

| model id | what it is | speed | best for | aliases |
|---|---|---|---|---|
| `qwen-38-27b` | **27B dense + VISION** (Qwen3.8), runs on both GPUs | ~69 tok/s code · 57 @32K · 37 @100K | code, RAG, **anything with images/video** | `qwen38`, `qwen38-27b`, `qwen36`, `qwen36-q6`, `qwen36-text` |
| `qwen-35b` | 35B MoE (mixture-of-experts) | **~206 tok/s** — the fast one | fast general chat, high-throughput text | `qwen36-35b` |
| `qwen-35b-batch` | same 35B, **4 parallel slots × 32K ctx** | throughput, not latency | **many small requests at once** (classification, scoring) | `qwen36-35b-batch` |
| `qwen-38-flash-next` | **125B MoE + VISION** (Qwen3.8-Flash-Next), GPUs + RAM (PLE on NVMe) | ~18.8 tok/s — the biggest · ⚠️ **~150 s cold swap-in** | hardest reasoning, quality, images/video | `flash-next`, `qwen38-next` |
| `qwen-embed` | **Qwen3-Embedding-0.6B** on CPU, always available | 1024-dim, zero VRAM | live query embeds while you chat | `embed`, `qwen3-embed` |
| `qwen-embed-gpu` | same embedder on the 4090, on-demand | 1024-dim, fast | **bulk corpus indexing** (evicts the chat model while it runs) | `embed-gpu`, `qwen3-embed-gpu` |

**Which chat model should I use?**
- **Default / fast:** `qwen-35b` — fastest, great general quality.
- **Code, RAG, or images:** `qwen-38-27b` — the only model that reads images/video, tuned for
  code and retrieval.
- **Hardest problems / biggest model:** `qwen-38-flash-next` — 125B, also sees images/video, but each
  cold swap-in costs **~150 s**, so batch its work (don't interleave it with other models turn-by-turn).
- **A flood of small requests** (e.g. scoring/classifying many items): `qwen-35b-batch`.

**Loading rules (important mental model):**
- The **four chat models share the GPUs and are mutually exclusive.** Asking for a different chat
  model unloads the current one first (a "swap"). You never have to manage this — it happens on demand.
- The **CPU embedder (`qwen-embed`) is always resident** and runs concurrently with any chat model.
  Embedding a query never evicts your chat model.
- The **GPU embedder (`qwen-embed-gpu`) is a chat-group member** — using it *does* swap the chat
  model out for the duration, then the next chat request swaps it back. Use it only for heavy
  batch indexing.

**Context window:** all chat models accept up to **131072 tokens** of context.

---

## Chatting (text)

Standard OpenAI Chat Completions. Base path: `/v1/chat/completions`.

```bash
curl -s localhost:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "qwen-38-27b",
  "messages": [
    {"role":"user","content":"Explain a hash map to a 10-year-old."}
  ],
  "max_tokens": 2048
}' | jq -r '.choices[0].message.content'
```

- **System prompt:** a default one is injected automatically if you don't send one. To set your
  own, just include a `{"role":"system","content":"..."}` message first and it's used as-is.
- **Temperature, top_p, stop, etc.:** standard OpenAI params pass straight through.

### These models think before they answer

The chat models produce **reasoning** and then a **final answer**, returned in two separate fields:

```bash
jq -r '.choices[0].message.reasoning_content'   # the thinking (chain-of-thought)
jq -r '.choices[0].message.content'             # the actual answer
```

Most of the time you only want `.content`. `reasoning_content` is there if you want to inspect how
it got there.

### max_tokens: the one gotcha that bites everyone

`max_tokens` covers **thinking + answer combined.** If it's too small, the model spends the whole
budget thinking and `content` comes back **empty** (everything landed in `reasoning_content`).

| model | use at least |
|---|---|
| `qwen-38-27b`, `qwen-35b`, `qwen-35b-batch` | **2048** |
| `qwen-38-flash-next` | **4096** (it thinks long, and variably) |

If you ever get an empty `content`, this is almost always why — raise `max_tokens`.

### Streaming

Add `"stream": true` for token-by-token Server-Sent Events, exactly like OpenAI:

```bash
curl -sN localhost:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model":"qwen-35b","stream":true,"max_tokens":2048,
  "messages":[{"role":"user","content":"Count to five slowly."}]}'
```

### Tool / function calling

Pass an OpenAI `tools` array as usual. One convenience built in: when `tools` are present the
server **raises a too-small `max_tokens` to a safe floor automatically** (512 for most models, 4096
for flash-next) so a tool call can't be truncated mid-argument. You can still set a larger value.

---

## Vision (images & video) — `qwen-38-27b` and `qwen-38-flash-next`

`qwen-38-27b` reads images and video natively. Use the OpenAI `image_url` content part — either a
`data:` URI (local file) or an `http(s)` URL.

```bash
curl -s localhost:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model":"qwen38",
  "messages":[{"role":"user","content":[
    {"type":"text","text":"What is in this image?"},
    {"type":"image_url","image_url":{"url":"data:image/png;base64,'"$(base64 -w0 pic.png)"'"}}
  ]}],
  "max_tokens":2048
}' | jq -r '.choices[0].message.content'
```

Both `qwen-38-27b` and `qwen-38-flash-next` accept images. Sending an image to `qwen-35b` will not work.

---

## Embeddings & RAG

OpenAI `/v1/embeddings`. Output is **1024-dimensional, L2-normalized** (unit length), using
last-token pooling. Two placements of the *same* model — choose by workload:

| model id | where | coexists with chat? | use for |
|---|---|---|---|
| `qwen-embed` | CPU | **yes** — always available | live per-query embeds while you chat |
| `qwen-embed-gpu` | 4090 | **no** — evicts the chat model while running | **bulk corpus indexing** |

```bash
# single input
curl -s localhost:8000/v1/embeddings -H 'Content-Type: application/json' -d '{
  "model":"qwen-embed","input":"how do I sort a list in python"
}' | jq '.data[0].embedding | length'      # -> 1024

# batch of inputs (send an array)
curl -s localhost:8000/v1/embeddings -H 'Content-Type: application/json' -d '{
  "model":"qwen-embed","input":["first chunk","second chunk","third chunk"]
}' | jq '.data | length'
```

**Building a retrieval (RAG) system:**
- **Index the corpus once** with `qwen-embed-gpu` (fast, but swaps the chat model out while it runs
  — fine for a one-time or offline job):
  ```bash
  curl -s localhost:8000/v1/embeddings -H 'Content-Type: application/json' \
    -d '{"model":"embed-gpu","input":["doc chunk 1","doc chunk 2","..."]}'
  ```
- **At query time**, embed each query with `qwen-embed` (CPU) — it runs alongside your chat model,
  so retrieval and generation don't queue behind each other.
- **Query prefix (matters for retrieval quality):** Qwen3-Embedding works best with an instruction
  prefix on the **query only** (not the documents). Prepend:
  ```
  Instruct: Given a search query, retrieve relevant passages
  Query: <your query text>
  ```
- Max input is **8192 tokens** per embed; keep chunks well under that.

---

## Point real tools at it

Anything OpenAI-compatible works. Base URL is `http://localhost:8000/v1`, any key string.

**Environment variables (works with most tools):**
```bash
export OPENAI_BASE_URL=http://localhost:8000/v1
export OPENAI_API_KEY=local
```

**Python (official OpenAI SDK):**
```python
from openai import OpenAI
c = OpenAI(base_url="http://localhost:8000/v1", api_key="local")

r = c.chat.completions.create(
    model="qwen-35b",
    messages=[{"role": "user", "content": "hi"}],
    max_tokens=2048,
)
print(r.choices[0].message.content)

e = c.embeddings.create(model="qwen-embed", input="text to embed")
print(len(e.data[0].embedding))   # 1024
```

**Aider:**
```bash
aider --openai-api-base http://localhost:8000/v1 --openai-api-key local --model openai/qwen-35b
```

**Anything else** (LangChain, LlamaIndex, Continue, curl, etc.): give it the base URL
`http://localhost:8000/v1`, any API key, and one of the model ids above.

---

## Handy operations

You never *have* to do these — models load on demand — but they're useful.

```bash
# What's running right now? (an empty "running" list is normal — nothing is loaded until asked)
curl -s localhost:8000/healthz | jq

# List the models you can request
curl -s localhost:8000/v1/models | jq -r '.data[].id'

# Pre-warm a model so the first real request is instant (blocks until it's resident)
curl -s localhost:8000/load/flash-next | jq       # GET /load/<model or alias>  (~150 s cold — pre-warm)

# Free the GPUs (unload whatever chat model is resident)
curl -s -X POST localhost:8000/unload | jq

# Watch VRAM live
watch -n1 nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
```

---

## What's normal (and looks alarming but isn't)

- **`"running": []` / both GPUs at 0 MiB** — the correct **resting state.** Nothing is loaded until
  you ask for a model. Not a failure.
- **A long pause on the first request to a model** — it's loading. Swap-in costs roughly:
  **27B ~15–20 s · 35B ~16–25 s · flash-next ~150 s cold** (the biggest model — batch its work). Cold
  (first time since boot) is slower than warm.
- **The first request after any load is ~15% slower** — one-time graph warmup, not a regression.
- **Switching between chat models is slow the first time each** — that's the swap. Staying on one
  model is fast.

---

## Troubleshooting

| symptom | cause / fix |
|---|---|
| `content` is empty, `reasoning_content` is full | `max_tokens` too small — raise it (≥2048, ≥4096 for flash-next). |
| Request hangs a long time then responds | Model was cold-loading or the request was long. Normal up to the swap costs above. |
| Image request ignored / errors | Images only work on `qwen-38-27b`. Check the `model` field. |
| Embeddings request is slow in bulk | Use `qwen-embed-gpu` for large batches (it swaps chat out); use `qwen-embed` for single queries. |
| "Is the server even up?" | `curl -s localhost:8000/healthz` → `{"router":"ok",...}` means yes. |
| Connection refused on :8000 | The service may be down. This needs an operator — see below. |

**Is the server actually up?**
```bash
curl -s localhost:8000/healthz | jq
# {"router":"ok","backend":{"swap":200,"running":[...]}}  -> healthy
```

If you get **connection refused** or `"router":"degraded"`, the underlying services need attention
from whoever administers this box (they are systemd units, `qwen-router` and `qwen-swap`). That's an
operator task, not something you fix as a user.

---

## Quick reference card

```
Endpoint      http://localhost:8000            (OpenAI-compatible, no auth)
Chat          POST /v1/chat/completions
Embeddings    POST /v1/embeddings
List models   GET  /v1/models
Health        GET  /healthz
Pre-warm      GET  /load/<model>
Unload        POST /unload

Chat models   qwen-38-27b   27B dense + VISION   (code/RAG/images)
              qwen-35b      35B MoE, fast        (default)
              qwen-35b-batch 35B, 4 slots        (many small requests)
              qwen-38-flash-next 125B MoE+vision (biggest; ~150s cold swap)
Embedders     qwen-embed      CPU, always-on     (query embeds)
              qwen-embed-gpu  4090, on-demand     (bulk indexing)

Remember      max_tokens >= 2048 (>= 4096 for flash-next)
              answer is in .choices[0].message.content
              thinking is in .choices[0].message.reasoning_content
              context window: 131072 tokens
              embeddings: 1024-dim, normalized, max 8192 input tokens
```
