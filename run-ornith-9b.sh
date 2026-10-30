#!/usr/bin/env bash
# Ornith-1.5-9B (ornith-ai) — DENSE 9B reasoning+agentic/coding model, TEXT-ONLY here.
#
# ARCH: general.architecture = `qwen35` — the SAME tag our Qwen3.8-27B carries (verified from both GGUFs:
#   33 blocks, GQA head_count 16 / kv 4, head_dim 256, full_attention_interval 4 = Qwen3.5 hybrid, mRoPE,
#   native ctx 262144). So it runs on the EXISTING production build (657e011) that already serves the 27B
#   on qwen35 — no rebuild.
#
# MTP: the GGUF ships an embedded next-token (nextn) head on the last block — blk.32.nextn.{eh_proj,enorm,
#   hnorm,shared_head_norm} — the SAME self-speculation head the 27B/35B carry. So MTP works natively via
#   `--spec-type draft-mtp` (no -md; the embedded head is used), vocab-matched by construction, ~zero extra
#   VRAM. A 9B fully on the 4090 is memory-bandwidth-bound, the regime where MTP pays (verify N draft tokens
#   per single weight-read) — unlike flash-next (CPU-compute-bound) where MTP was a net loss. MTP requires
#   -np 1 (we use it). BENCHMARKED on this box (standalone, Q6_K, 4090-only, temp 0, 256 tok):
#     off -> 95 t/s both; NMAX=3 -> code 169 t/s (~1.8x, acc 0.78) / prose 130 t/s (~1.37x, acc 0.49).
#   NMAX=3 is the knee: 4-5 add a sliver of code speed but hurt prose (drafts wasted; acc decays to 0.34).
#   Toggle with MTP=off; re-sweep SPEC_NMAX if the workload is code-only (4) vs prose-heavy (2).
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

# MTP self-speculative decode via the embedded nextn head (default on). SPEC_NMAX = draft depth per step.
# MTP=off => --spec-type none. Set from the NMAX sweep (draft_n>0 + acceptance + t/s vs off).
MTP="${MTP:-on}"
SPEC_NMAX="${SPEC_NMAX:-3}"
SPEC_ARGS=(--spec-type draft-mtp --spec-draft-n-max "$SPEC_NMAX")
[ "$MTP" = "off" ] && SPEC_ARGS=(--spec-type none)

# SAMPLING: Ornith card's GENERAL thinking profile (temp 1.0 WITH presence_penalty 1.5 — they go together).
#   For PRECISE CODING the card specifies temp 0.6 / presence_penalty 0.0 (same top_p/top_k/min_p); flip both
#   via TEMP/PRES env, never temp alone. Thinking is forced ON at the router regardless (FORCE_THINKING).
TEMP="${TEMP:-1.0}"; PRES="${PRES:-1.5}"

# PIN TO THE 4090 ONLY (=GPU0). Only one device is visible => -ngl 99 puts everything on it; no -sm/-ts/spill.
exec env CUDA_VISIBLE_DEVICES="$GPU_4090" "$BIN" \
  -m "$MODEL" \
  --alias ornith \
  -ngl "$NGL" -c "$CTX" -fa on -np 1 \
  "${SPEC_ARGS[@]}" \
  --cache-type-k "$KV" --cache-type-v "$KV" \
  --no-context-shift --ctx-checkpoints 32 \
  --jinja \
  --reasoning-preserve \
  --temp "$TEMP" --top-p 0.95 --top-k 20 --min-p 0.0 \
  --presence-penalty "$PRES" --repeat-penalty 1.0 \
  --host 127.0.0.1 --port "$PORT"
# NOTE: no --mmproj (vision off). No -sm/-ts/-ot (single GPU). MTP via the embedded nextn head (SPEC_ARGS).
