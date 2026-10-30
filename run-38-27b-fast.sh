#!/usr/bin/env bash
# Qwen3.8-27B — FAST quant: ISTA-DASLab GSQ-RCO IQ3_S-mtp (non-uniform mixed-precision, ~3.5 bpw, 11.8 GB,
# task-lossless per the card: AIME25 100.0 / LCB-v6 85.71 == BF16, GPQA-D within 0.51). SAME base model as
# qwen-38-27b (Qwen3.8-27B, arch `qwen35`) — runs on THIS stack's build, no rebuild. TEXT-ONLY here.
#
# WHY A SECOND 27B ENTRY: the production qwen-38-27b is Q6_K (22.9 GiB) and spans BOTH GPUs. This quant is
# HALF that, so the WHOLE MODEL FITS ON THE 4090 ALONE — no --split-mode layer, no --tensor-split, no PCIe
# hop per layer. Single fast card (1008 GB/s) + the embedded MTP head = the fastest 27B this box can decode.
#
# PLACEMENT: default SINGLE-GPU (expose only the 4090). Set TS=<a,b> to test the dual-GPU alternative
# (then both GPUs are exposed, -sm layer --tensor-split TS -mg 0) — used by the sweep to compare, not shipped.
set -euo pipefail
source "$HOME/ai/qwen36/gpus.sh"
export PATH=/usr/local/cuda-12.8/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:${LD_LIBRARY_PATH:-}

ROOT=~/ai/qwen36
BIN="$ROOT/llama.cpp/build/bin/llama-server"
MODEL="${MODEL:-$ROOT/models/Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf}"

CTX="${CTX:-131072}"                # full 131072 FITS on the 4090 alone (~18.0 of 24.5 GiB w/ MTP n=3) —
                                    #   no context sacrifice; no step-down needed. Native train ctx is 262144.
KV="${KV:-q8_0}"                    # LOCKED q8_0 (q4_0 breaks long-range retrieval on this stack).
NGL="${NGL:-99}"
PORT="${PORT:-8084}"                # freed slot (8087 is kr_web; 8080/82/83/85/86/88/89 taken).

# MTP self-speculative decode via the embedded nextn head (default on). BENCHMARKED (single-GPU, 131072,
# q8_0 KV, temp 0, 256 tok): off -> 59 t/s both; n=3 -> code 124 (2.08x, acc 0.91) / prose 102 (1.72x, acc
# 0.67). n=3 is the knee (prose peak); n=4/5 add code (135/139) but decay prose (95/88). Single-GPU is
# ~1.45x faster than dual-GPU TS=4,1 @131072 (code 124 vs 86) — keeping it all on the 4090 is the big lever.
MTP="${MTP:-on}"
SPEC_NMAX="${SPEC_NMAX:-3}"         # 4 for code-heavy, 2 for prose-heavy; MTP=off to disable.
SPEC_ARGS=(--spec-type draft-mtp --spec-draft-n-max "$SPEC_NMAX")
[ "$MTP" = "off" ] && SPEC_ARGS=(--spec-type none)

# Prefill ubatch: bigger = faster prompt processing (we have 4090 headroom). Unset = build default (2048/512).
UBA=(); [ -n "${UB:-}" ] && UBA=(-b "$UB" -ub "$UB")

# PLACEMENT: single-GPU (4090 only) unless TS is set (dual, for the sweep's comparison).
if [ -n "${TS:-}" ]; then
  DEV="$GPU_BOTH";  PLACE=(-sm layer --tensor-split "$TS" -mg 0)
else
  DEV="$GPU_4090";  PLACE=()          # only the 4090 visible => -ngl 99 keeps everything on it, no split
fi

exec env CUDA_VISIBLE_DEVICES="$DEV" "$BIN" \
  -m "$MODEL" \
  --alias qwen38-fast \
  -ngl "$NGL" -c "$CTX" -fa on -np 1 \
  "${PLACE[@]}" \
  "${UBA[@]}" \
  "${SPEC_ARGS[@]}" \
  --cache-type-k "$KV" --cache-type-v "$KV" \
  --no-context-shift --ctx-checkpoints 32 \
  --jinja \
  --reasoning-preserve \
  --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0.0 \
  --presence-penalty 0.0 --repeat-penalty 1.0 \
  --host 127.0.0.1 --port "$PORT"
# NOTE: no --mmproj (vision off). MTP via embedded nextn head. Single-GPU by default (no -sm/-ts).
