#!/usr/bin/env bash
# Qwen3.8-27B — DENSE, NATIVE VISION-LANGUAGE, QUALITY-FIRST. Replaces the whole 3.6-27B dense line
# (Q4 fast + Q6 quality + the retired 3.6 vision path) with ONE multimodal server.
#
# ARCH: model_type qwen3_5 (SAME family as the 3.5/3.6 stack, per the user + config.json:
#   Qwen3_5ForConditionalGeneration). llama.cpp maps it to the `qwen3vl` dense graph and this build
#   (657e011) already ships that graph + clip_graph_qwen3vl (vision) + the gated_delta_net CUDA
#   kernels the hybrid linear-attention layers need. The 122B (qwen3_5 MoE) already runs on it, so
#   the text path is proven; VISION on this exact build is verified by gates38.sh, not assumed.
#
# SPLIT: 22.9 GiB of Q6_K weights won't fit the 4090 (24.5 GiB) alone once KV + vision/compute buffers
#   are added, so it spans BOTH GPUs. --split-mode layer (whole layers per GPU) + --tensor-split
#   (proportion). -sm layer is EXPLICIT so it can never fall back to row/tensor split (activations
#   across PCIe twice/layer). DENSE model => NO -ot (per-tensor placement has the same thrash cost).
#   -mg 0 keeps output + the clip/mmproj projector on the 4090.
#
# MTP + VISION IN ONE SERVER: MTP head is embedded (config: mtp_num_hidden_layers 1, unsloth_fixed_mtp).
#   The retired 3.6 vision server ran WITHOUT MTP (slot-position corruption risk with images, #22867/
#   #23371). Here we load BOTH and let gates38.sh decide empirically: it checks text requests still
#   draft (draft_n>0) AND an image request answers correctly. If a future llama.cpp regresses that
#   coexistence, set MTP=off (below) to fall back to the correctness-first vision-only mode.
set -euo pipefail

export PATH=/usr/local/cuda-12.8/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:${LD_LIBRARY_PATH:-}

ROOT=~/ai/qwen36
BIN="$ROOT/llama.cpp/build/bin/llama-server"          # always absolute — never PATH-resolved
MODEL="${MODEL:-$ROOT/models/Qwen3.8-27B-Q6_K.gguf}"  # unsloth/Qwen3.8-27B-GGUF, byte-verified; env-overridable
MMPROJ="${MMPROJ:-$ROOT/models/mmproj-Qwen3.8-F16.gguf}" # F16, NOT Q8_0 -> avoids Blackwell #24399 on the 5060 Ti

# --- tunables (overridable via env) ---
CTX="${CTX:-131072}"                # native is 262144; 131072 keeps VRAM headroom for vision buffers.
KV="${KV:-q8_0}"                    # LOCKED q8_0 everywhere; never q4_0 (guts long-range retrieval).
SPEC_NMAX="${SPEC_NMAX:-4}"         # carried from the 3.6-27B (code peaked at n=4).
NGL="${NGL:-99}"
TS="${TS:-3,1}"                     # tensor-split PROPORTION across GPU0,GPU1 (~75/25). Starting from
                                    #   the 3.6-Q6 measured optimum (minimises the slow 5060 Ti share).
                                    #   gates38.sh [1] verifies >=1 GiB free on EACH; if GPU0 OOMs,
                                    #   shift MORE onto GPU1 (2,1) — never toward GPU0 (binding card).
MTP="${MTP:-on}"                    # on|off. off = correctness-first vision (no draft) if MTP+vision regress.
VISION="${VISION:-on}"              # on|off. off drops --mmproj (text-only; frees the projector VRAM).
PORT="${PORT:-8080}"               # backend port. 8080 = the primary-dense slot (replaces qwen-27b).
                                    #   Overridable so a standalone gate can use a non-colliding port.

# --- EXPERIMENTAL: ngram-mod drafter (draftless LCG-hash table over the token stream) --------------
# NGRAM=on stacks a draftless recall drafter WITH MTP via the comma-list --spec-type draft-mtp,ngram-mod.
# Precedence: ngram-mod PREEMPTS MTP whenever its ~16 MB pool has a hit (chained single-token lookups →
# variable-length draft, stops at first miss); MTP (--spec-draft-n-max) is the FALLBACK depth on a miss.
# Lossless: the target verifies the whole block and accepts only its own samples. Default OFF (unchanged).
NGRAM="${NGRAM:-off}"               # off|on
NG_MATCH="${NG_MATCH:-24}"          # --spec-ngram-mod-n-match: lookup n-gram length (<16 warns: poor quality)
NG_MIN="${NG_MIN:-24}"             # --spec-ngram-mod-n-min: minimum drafted tokens
NG_MAX="${NG_MAX:-86}"             # --spec-ngram-mod-n-max: maximum drafted tokens (variable, stops at miss)
SEED="${SEED:-1234}"               # fixed seed — measurement reproducibility (acc spread 0.836..0.891 across
                                    #   unseeded runs swamps small effects). SEED=-1 restores production variety.

SPEC_ARGS=(--spec-type draft-mtp --spec-draft-n-max "$SPEC_NMAX")
[ "$MTP" = "off" ] && SPEC_ARGS=(--spec-type none)
# ngram-mod stacks only when MTP is on (it preempts, MTP is the fallback). MTP=off keeps --spec-type none.
if [ "$NGRAM" = "on" ] && [ "$MTP" != "off" ]; then
  SPEC_ARGS=(--spec-type draft-mtp,ngram-mod
             --spec-draft-n-max "$SPEC_NMAX"
             --spec-ngram-mod-n-match "$NG_MATCH"
             --spec-ngram-mod-n-min  "$NG_MIN"
             --spec-ngram-mod-n-max  "$NG_MAX")
fi

MMPROJ_ARGS=(--mmproj "$MMPROJ")
[ "$VISION" = "off" ] && MMPROJ_ARGS=()

# PIN TO BOTH GPUs explicitly (=0,1) — same self-documenting pin the sibling scripts use so
# qwen-swap.service can stay pin-agnostic. -sm layer + -ts spread whole layers across GPU0+GPU1.
# NOTE: no --chat-template-file — 3.8 ships its own embedded template (developer-role + improved tool
# calling); the 3.6 v19 hand-fix does not apply. gates38.sh [4]/[5] verify thinking+tools parse.
exec env CUDA_VISIBLE_DEVICES=0,1 "$BIN" \
  -m "$MODEL" \
  "${MMPROJ_ARGS[@]}" \
  --alias qwen38 \
  -ngl "$NGL" -c "$CTX" -fa on -np 1 \
  -sm layer --tensor-split "$TS" -mg 0 \
  "${SPEC_ARGS[@]}" \
  --cache-type-k "$KV" --cache-type-v "$KV" \
  --no-context-shift --ctx-checkpoints 32 \
  --jinja \
  --reasoning-preserve \
  --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0.0 \
  --presence-penalty 0.0 --repeat-penalty 1.0 \
  --seed "$SEED" \
  --host 0.0.0.0 --port "$PORT"
