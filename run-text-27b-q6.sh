#!/usr/bin/env bash
# 27B dense, QUALITY-FIRST — UD-Q6_K_XL. Spans BOTH GPUs (24.2 GiB weights won't fit the 4090 alone).
# This is a SECOND 27B entry ALONGSIDE run-text-mtp.sh (Q4_K_XL / GPU0-only), not a replacement.
# The Q4 entry stays live as the fallback; this one is the higher-quality option.
#
# SPLIT: --split-mode layer (whole layers per GPU) + --tensor-split 3,1 (proportion, ~75/25).
#   -sm layer is set EXPLICITLY so it can never fall back to row/tensor split, which would send
#   activations across PCIe twice per layer. This is a DENSE model, so we do NOT use -ot: per-tensor
#   placement across devices has the same PCIe-thrash cost. Layer split keeps each layer's math on
#   one device; only the residual stream crosses the bus, once per GPU boundary.
# Every other flag (ctx, q8_0 KV, flash-attn, MTP n=4, v19 template, sampling) is CARRIED UNCHANGED
# from run-text-mtp.sh so the two 27B entries differ ONLY in quant, device span, and port.
set -euo pipefail

export PATH=/usr/local/cuda-12.8/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:${LD_LIBRARY_PATH:-}

ROOT=~/ai/qwen36
BIN="$ROOT/llama.cpp/build/bin/llama-server"          # always absolute — never PATH-resolved
MODEL="$ROOT/models/Qwen3.6-27B-UD-Q6_K_XL.gguf"
TMPL="$ROOT/templates/qwen3.6-v19.jinja"

# --- tunables (overridable via env) ---
CTX="${CTX:-131072}"                # >=100K hard requirement (carried unchanged)
KV="${KV:-q8_0}"                    # LOCKED q8_0 everywhere; never q4_0 (guts long-range retrieval)
SPEC_NMAX="${SPEC_NMAX:-4}"         # carried from Q4 entry (code peaked at n=4)
NGL="${NGL:-99}"
TS="${TS:-3,1}"                     # tensor-split PROPORTION across GPU0,GPU1. MEASURED best: 3,1 is
                                    #   speed-optimal (minimises the slow 5060 Ti's share) AND leaves
                                    #   >=1 GiB free on each (GPU0 ~2.2G / GPU1 ~5.9G, GPU1 resident
                                    #   ~10.4G). If GPU0 ever OOMs, shift MORE onto GPU1 (2,1) — NOT
                                    #   toward GPU0; GPU0 is the binding card.
MTP="${MTP:-on}"                    # on|off

SPEC_ARGS=(--spec-type draft-mtp --spec-draft-n-max "$SPEC_NMAX")
[ "$MTP" = "off" ] && SPEC_ARGS=(--spec-type none)

# NOTE: NO CUDA_VISIBLE_DEVICES pin here — this entry deliberately needs BOTH GPUs. -sm layer + -ts
# distributes whole layers across them; -mg 0 keeps output/KV-for-final on the 4090.
exec "$BIN" \
  -m "$MODEL" \
  --alias qwen36-q6 \
  -ngl "$NGL" -c "$CTX" -fa on -np 1 \
  -sm layer --tensor-split "$TS" -mg 0 \
  "${SPEC_ARGS[@]}" \
  --cache-type-k "$KV" --cache-type-v "$KV" \
  --no-context-shift --ctx-checkpoints 32 \
  --jinja --chat-template-file "$TMPL" \
  --reasoning-preserve \
  --temp 0.6 --top-p 0.95 --top-k 20 --min-p 0.0 \
  --presence-penalty 0.0 --repeat-penalty 1.0 \
  --host 0.0.0.0 --port 8081
# chat-template: v19, same as the Q4 entry. NO --mmproj (keeps the MTP draft path accelerating).
