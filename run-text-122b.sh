#!/usr/bin/env bash
# Server D — Qwen3.5-122B-A10B (MoE), text only, MTP, TRI-TIER: 4090 + 5060 Ti + system RAM.
# EXPERIMENT (not a production ship): bar is >=10 t/s without being overly quantized.
#
# PLACEMENT: fill GPU0 -> fill GPU1 -> remainder to CPU.
# The 35B's "min-spill" principle was never "minimise the second GPU" -- it was "MINIMISE THE
# SLOWEST TIER IN PLAY". On the 35B the slow tier was GPU1 (its alternative was the faster 4090),
# so min-spill starved GPU1. Here the slow tier is the CPU (DDR5 ~85 GB/s AND i-quant dequant
# compute), and GPU1 at 448 GB/s is the *fast* alternative to it. Same principle, inverted
# direction: FILL GPU1 TO THE BRIM to starve the CPU slice.
#
# WHY THE CPU TIER IS THE RISK: it holds IQ2_S / IQ4_XS routed experts. i-quants use codebook
# lookups, not K-quants' simple bit-unpacking, so the CPU tier is slower than the DDR5-vs-VRAM
# bandwidth ratio alone predicts -- an extra dequant-compute factor landing exactly on these types.
#
# #24399 (Blackwell sm_120 Q8_0): structurally dead. Routed experts are IQ2_S/IQ4_XS/IQ3_S/Q6_K/
# Q2_K/Q4_K -- ZERO Q8_0/Q8_K. Every Q8_0 (MTP head, 2 attn, 3 shexp) is non-expert -> CUDA0.
# CPU and CUDA0 are both safe for Q8_0; only CUDA1 is the hazard. Verify residency at boot.
export PATH=/usr/local/cuda-12.8/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:${LD_LIBRARY_PATH:-}

ROOT=~/ai/qwen36
BIN="$ROOT/llama.cpp/build/bin/llama-server"
MODEL="$ROOT/models/UD-IQ3_S/Qwen3.5-122B-A10B-UD-IQ3_S-00001-of-00003.gguf"   # part 1; llama.cpp
                                                                              # finds 2/3 by name
TMPL="$ROOT/templates/qwen3.6-v19.jinja"     # embedded template is checked at M2; v19 unless proven fine
CTX="${CTX:-131072}"
KV="${KV:-q8_0}"                  # LOCKED: q4_0 broke long-range retrieval on the 27B (2/5 vs 5/5)
# M4 SWEEP RESULT: 3, NOT the card's 6.
#   The card's 6 peaks short-code (40.7 vs 36.9 t/s, +10%) but makes CREATIVE 27% SLOWER THAN MTP
#   OFF (18.1 vs 24.8), with acceptance collapsing 0.47 -> 0.24 (tripping the <40% gate).
#   WHY, and it is tier-specific: on the all-VRAM 35B, creative at acceptance 0.46 still GAINED
#   1.25x, because a rejected draft on a bandwidth-bound GPU is ~free (the weights were read
#   anyway). Here a rejected draft costs FULL CPU COMPUTE -- the CPU tier is compute-bound, so
#   wasted drafts are wasted work. Deep drafting only pays where acceptance is high.
#   n=3 is the only depth that is never harmful: +1.51x code, +1.44x long-code, neutral creative.
# NOTE: n>=7 will NOT BOOT -- deeper draft buffers OOM against GPU0's 602 MiB headroom
#   ("failed to create MTP context" is the symptom; cudaMalloc OOM is the cause). Raising n_max
#   past 6 requires rebalancing an expert layer off GPU0 first.
SPEC_NMAX="${SPEC_NMAX:-3}"
NGL="${NGL:-99}"
MTP="${MTP:-on}"
PORT="${PORT:-8084}"
TS="${TS:-1,0}"                   # all layers weighted to GPU0; -ot does the real placement, and
                                  # both CUDA backends stay initialised so CUDA1 resolves.

# ---- tri-tier expert placement (M4 sweep knobs) ----
# 49 blocks (0..48). Per-layer routed-expert bytes: 900 MiB x47, blk.46=1290 (fat), blk.48=936 (MTP).
# Everything NOT matched below stays on CUDA0 (attention, KV, shared expert, router, MTP head,
# and the highest-numbered expert layers -- including fat blk.46 and MTP blk.48, which we
# deliberately keep on the fast card).
CPU_LAYERS="${CPU_LAYERS:-0-16}"    # 17 layers -> ~15300 MiB in RAM  (the slowest tier: smallest slice)
GPU1_LAYERS="${GPU1_LAYERS:-17-32}" # 16 layers -> ~14400 MiB on the 5060 Ti (filled, not min-spilled)

expand() { # "0-16" or "0-3,7,9-11" -> "0|1|2|...|16"
  local out=() part lo hi
  IFS=',' read -ra parts <<< "$1"
  for part in "${parts[@]}"; do
    if [[ "$part" == *-* ]]; then
      lo="${part%%-*}"; hi="${part##*-}"
      for ((i=lo; i<=hi; i++)); do out+=("$i"); done
    else out+=("$part"); fi
  done
  local IFS='|'; echo "${out[*]}"
}

# CRITICAL: -ot must be ONE comma-separated argument, NOT repeated flags.
#   "DEPRECATED: argument '-ot' specified multiple times, use comma-separated values instead
#    (only last value will be used)"
# Two -ot flags silently collapse to the last one: the CPU rule would be DROPPED, layers 0-16 would
# default to CUDA0, and GPU0 would need ~29 GiB of experts on a 24 GiB card -> OOM. The warning is
# easy to miss in a 48 GiB boot. The regexes contain '|' but no commas, so comma-joining is safe.
OT_RULES=()
[ "$CPU_LAYERS"  != "none" ] && OT_RULES+=("blk\.($(expand "$CPU_LAYERS"))\.ffn_(gate|down|up)_exps\.weight=CPU")
[ "$GPU1_LAYERS" != "none" ] && OT_RULES+=("blk\.($(expand "$GPU1_LAYERS"))\.ffn_(gate|down|up)_exps\.weight=CUDA1")
OT_ARGS=()
if [ ${#OT_RULES[@]} -gt 0 ]; then
  joined=$(IFS=,; echo "${OT_RULES[*]}")
  OT_ARGS=(-ot "$joined")
fi

SPEC_ARGS=(--spec-type draft-mtp --spec-draft-n-max "$SPEC_NMAX")
[ "$MTP" = "off" ] && SPEC_ARGS=(--spec-type none)

exec env CUDA_VISIBLE_DEVICES=0,1 "$BIN" \
  -m "$MODEL" --alias qwen35-122b \
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
# NOTE: no --mlock / --no-mmap (would pin the full 48 GiB in host RAM).
# The CPU tier here is deliberate expert placement via -ot, NOT a partial layer offload.
