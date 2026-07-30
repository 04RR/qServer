#!/usr/bin/env bash
# Server C — Qwen3.6-35B-A3B (MoE), text only, MTP, spans 4090 (GPU0) + 5060 Ti (GPU1).
# Placement model: EVERYTHING on CUDA0 except a minimal spill of routed-expert tensors to CUDA1.
#
# WHY MIN-SPILL (do not "fill" GPU1 — it is overflow storage, not a second engine):
#   Transformer blocks are strictly sequential, so total decode = t_GPU0 + t_GPU1, never max().
#   There is no parallelism to harvest. The 4090 reads ~1008 GB/s vs the 5060 Ti's ~448 GB/s,
#   so every byte parked on GPU1 is a straight tax, and each spilled layer also costs a PCIe
#   round-trip + sync (no NVLink). Spill the FEWEST layers that let the budget close.
#
# WHY THESE 6 LAYERS: the UD quant bumped blk.34/38 to Q6_K (498 MiB) and blk.39 to 562 MiB.
#   Spilling a fat layer moves more bytes/token to the slow card for zero benefit, so the spill
#   pool is plain-464 MiB layers only. blk.40 (MTP block) stays whole on CUDA0 — the draft path
#   round-trips enough without a cross-GPU hop in the verify step.
#
# #24399 (Blackwell sm_120 Q8_0 device exception) is structurally dead here, BY CONSTRUCTION:
#   routed experts are exclusively Q4_K/Q5_K/Q6_K; every Q8_0 tensor (attn_qkv, attn_gate,
#   ssm_out, shexp, token_embd, output) is non-expert and stays on CUDA0. The regex matching only
#   `_exps` is what holds the gate — so verify residency at boot, don't trust the regex.
export PATH=/usr/local/cuda-12.8/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:${LD_LIBRARY_PATH:-}

ROOT=~/ai/qwen36
BIN="$ROOT/llama.cpp/build/bin/llama-server"          # always absolute — never PATH-resolved
MODEL="$ROOT/models/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf"
TMPL="$ROOT/templates/qwen3.6-v19.jinja"              # embedded 35B template is STALE (has `| safe`
                                                      # on the tool-call path) — always pass v19.
CTX="${CTX:-131072}"
KV="${KV:-q8_0}"                    # LOCKED: q4_0 broke long-range retrieval on the 27B (2/5 vs 5/5).
SPEC_NMAX="${SPEC_NMAX:-3}"         # default; re-swept in M4 (MoE + cross-GPU draft may not pay)
NGL="${NGL:-99}"
MTP="${MTP:-on}"                    # on|off
PORT="${PORT:-8082}"

# M4 sweep knob: which layers' routed experts spill to CUDA1. Climb from min-spill (6 -> 10 -> 16).
# Plain-464 pool only; never 34/38/39 (Q6_K-bumped) and never 40 (MTP block).
SPILL="${SPILL:-28|29|30|31|32|33}"
OT_ARGS=(-ot "blk\.(${SPILL})\.ffn_(gate|down|up)_exps\.weight=CUDA1")
[ "$SPILL" = "none" ] && OT_ARGS=()

# TS: tensor-split ratio. Default "1,0" = 100% of layers weighted to GPU0, so -ot is the ONLY
# thing that moves tensors (and both CUDA backends stay initialized so the CUDA1 target resolves).
# Set TS=<a,b> with SPILL=none to measure the naive whole-layer tensor-split alternative, which
# also drags attention + KV onto the slow card instead of only expert weights.
TS="${TS:-1,0}"

SPEC_ARGS=(--spec-type draft-mtp --spec-draft-n-max "$SPEC_NMAX")
[ "$MTP" = "off" ] && SPEC_ARGS=(--spec-type none)

# -ts 1,0 : split-mode layer with 100% weight on GPU0. This keeps BOTH CUDA backends initialized
# (so the -ot CUDA1 target resolves) while leaving default placement entirely on the 4090.
# `-sm none` would risk never creating the CUDA1 backend at all.
exec env CUDA_VISIBLE_DEVICES=0,1 "$BIN" \
  -m "$MODEL" --alias qwen36-35b \
  -ngl "$NGL" -c "$CTX" -fa on -np 1 \
  -sm layer -ts "$TS" -mg 0 \
  "${OT_ARGS[@]}" \
  "${SPEC_ARGS[@]}" \
  --cache-type-k "$KV" --cache-type-v "$KV" \
  --no-context-shift --ctx-checkpoints 32 \
  --jinja --chat-template-file "$TMPL" \
  --reasoning-preserve \
  --temp 0.6 --top-p 0.95 --top-k 20 --min-p 0.0 \
  --presence-penalty 0.0 --repeat-penalty 1.0 \
  --host 0.0.0.0 --port "$PORT"
# NOTE: no --mmproj (vision retired). No -ot ...=CPU, no --cpu-moe/--n-cpu-moe: everything in VRAM.
