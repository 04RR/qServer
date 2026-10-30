#!/usr/bin/env bash
# Ornith-1.5-9B (ornith-ai) — DENSE 9B reasoning+agentic/coding model, TEXT-ONLY here.
#
# ARCH: general.architecture = `qwen35` — the SAME tag our Qwen3.8-27B carries (verified from both GGUFs:
#   33 blocks, GQA head_count 16 / kv 4, head_dim 256, full_attention_interval 4 = Qwen3.5 hybrid, mRoPE,
#   native ctx 262144). So it runs on the EXISTING production build (657e011) that already serves the 27B
#   on qwen35 — no rebuild. Dense => no MTP head in the GGUF => no speculative decode.
#
# FASTEST-INFERENCE PLACEMENT (the #1 past learning): the whole model lives on the 4090 (GPU0) ALONE.
#   At Q6_K (~7.6 GiB) + q8_0 KV (small: the hybrid keeps a KV cache only on the full-attention layers),
#   it fits the 24.5 GiB 4090 with headroom. We therefore expose ONLY the 4090 (CUDA_VISIBLE_DEVICES) and
#   do NOT tensor-split or spill to the 5060 Ti — every byte parked on the slow card (448 vs 1008 GB/s) plus
#   a PCIe hop per layer is a straight tax with no parallelism to win back (blocks are sequential). One fast
#   card, no splits = the fastest this box can decode a 9B.
#
# VISION: OFF by design (user: "no vision needed"); the mmproj is not downloaded and --mmproj is not passed.
#   Re-enable later by fetching mmproj-Ornith-1.5-9B-BF16.gguf and adding --mmproj (F16 projector is fine on
#   the 4090 — the Blackwell #24399 Q8_0 exception only bites the 5060 Ti, which we don't use here).
set -euo pipefail
source "$HOME/ai/qwen36/gpus.sh"   # GPU UUIDs: pin by UUID, not index
export PATH=/usr/local/cuda-12.8/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:${LD_LIBRARY_PATH:-}

ROOT=~/ai/qwen36
BIN="$ROOT/llama.cpp/build/bin/llama-server"          # always absolute — the production build (qwen35-capable)
MODEL="${MODEL:-$ROOT/models/Ornith-1.5-9B-Q6_K.gguf}" # ornith-ai/Ornith-1.5-9B-GGUF, Q6_K; env-overridable

# --- tunables (overridable via env) ---
CTX="${CTX:-131072}"                # native is 262144; 131072 is the house default (YaRN left OFF — the card
                                    #   warns static YaRN hurts normal-length quality; enable only if a run truly
                                    #   needs >262144, via --rope-scaling yarn --yarn-orig-ctx 262144).
KV="${KV:-q8_0}"                    # LOCKED q8_0 (q4_0 silently guts long-range retrieval on this stack).
NGL="${NGL:-99}"                    # all layers on the single visible GPU (the 4090).
PORT="${PORT:-8084}"                # backend port (8087 is the kr_web Docker container; 8080/82/83/85/86/88/89 taken).

# SAMPLING: Ornith card's GENERAL thinking profile (temp 1.0 WITH presence_penalty 1.5 — they go together).
#   For PRECISE CODING the card specifies temp 0.6 / presence_penalty 0.0 (same top_p/top_k/min_p); flip both
#   via TEMP/PRES env, never temp alone. Thinking is forced ON at the router regardless (FORCE_THINKING).
TEMP="${TEMP:-1.0}"; PRES="${PRES:-1.5}"

# PIN TO THE 4090 ONLY (=GPU0). Only one device is visible => -ngl 99 puts everything on it; no -sm/-ts/spill.
exec env CUDA_VISIBLE_DEVICES="$GPU_4090" "$BIN" \
  -m "$MODEL" \
  --alias ornith \
  -ngl "$NGL" -c "$CTX" -fa on -np 1 \
  --cache-type-k "$KV" --cache-type-v "$KV" \
  --no-context-shift --ctx-checkpoints 32 \
  --jinja \
  --reasoning-preserve \
  --temp "$TEMP" --top-p 0.95 --top-k 20 --min-p 0.0 \
  --presence-penalty "$PRES" --repeat-penalty 1.0 \
  --host 127.0.0.1 --port "$PORT"
# NOTE: no --mmproj (vision off). No -sm/-ts/-ot (single GPU). No --spec-type (dense: no MTP head).
