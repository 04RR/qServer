#!/usr/bin/env bash
# Qwen3.8-Flash-Next on MAINLINE llama.cpp — eval launcher (copy of the live fork launcher).
# ISOLATED: BIN = ~/ai/flashnext-main build; default MTP head = the SELF-CONTAINED Q8_0 (mainline
# cannot borrow tensors from the `shared-` head). Default PORT 8095. Does NOT touch the live stack.
#
# Keeps every hard rule: -sm layer only (never row/tensor), -ot ONE comma arg, GPUs pinned by UUID,
# f16 KV default, mmap on (NO --load-mode none), NEVER GGML_CUDA_ENABLE_UNIFIED_MEMORY, no --mlock.
# New-build knobs added for the sweep:
#   UB             -> -b/-ub (prefill ubatch; unset = build default 2048/512)
#   SPEC_UB        -> --spec-draft-n-ubatch (MTP draft ubatch)
#   MOE_CACHE      -> --moe-cache-mib (GPU expert cache, #29887/#30112/#30637)
#   DRAFT_SAMPLING -> --spec-draft-sampling (greedy|probabilistic)
#   GGML_CUDA_REGISTER_HOST (env) -> pinned host memory for CPU-expert copies
#   CKPT           -> -ctxcp (ctx checkpoints per slot; reuse a prefilled 100K haystack)
#   T              -> -t     (CPU generation threads; unset = build default -1)
set -euo pipefail
source "$HOME/ai/qwen36/gpus.sh"   # GPU UUIDs: pin by UUID, not index
export PATH=/usr/local/cuda-12.8/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:${LD_LIBRARY_PATH:-}

ROOT=~/ai/flashnext-main
BIN="${BIN:-$ROOT/llama.cpp/build/bin/llama-server}"
FMODELS=~/ai/flashnext/models   # reuse the already-downloaded GGUFs (shards + mmproj + heads)

MODEL="${MODEL:-$FMODELS/UD-IQ4_XS/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf}"
MMPROJ="${MMPROJ:-$FMODELS/mmproj-F16.gguf}"
MTP="${MTP:-$FMODELS/MTP/mtp-Qwen3.8-Flash-Next-Q8_0.gguf}"   # SELF-CONTAINED head (mainline)
CTX="${CTX:-131072}"
NMAX="${NMAX:-2}"
NCM="${NCM:-31}"
KVK="${KVK:-f16}"; KVV="${KVV:-f16}"
TS="${TS:-40,8}"
MG="${MG:-0}"
PORT="${PORT:-8095}"
NP="${NP:-1}"
PLE_OT="${PLE_OT:-per_layer_token_embd=CPU}"
VISION="${VISION:-on}"

MM=();   [ "$VISION" = "on" ] && MM=(--mmproj "$MMPROJ")
TSA=();  [ -n "$TS" ] && TSA=(--tensor-split "$TS")
WARM=(); [ "${NO_WARMUP:-1}" = "1" ] && WARM=(--no-warmup)
UBA=();  [ -n "${UB:-}" ] && UBA=(-b "$UB" -ub "$UB")
MCA=();  [ -n "${MOE_CACHE:-}" ] && MCA=(--moe-cache-mib "$MOE_CACHE")
LM=();   [ -n "${LOAD_MODE:-}" ] && LM=(-lm "$LOAD_MODE")
LVA=();  [ -n "${LV:-}" ] && LVA=(-lv "$LV")
CKA=();  [ -n "${CKPT:-}" ] && CKA=(-ctxcp "$CKPT")
TA=();   [ -n "${T:-}" ] && TA=(-t "$T")

# MTP speculative draft (default on). SPEC_OFF=1 = baseline / batch lane.
SPEC=(-md "$MTP" --spec-type draft-mtp --spec-draft-n-max "$NMAX")
[ -n "${SPEC_UB:-}" ]        && SPEC+=(--spec-draft-n-ubatch "$SPEC_UB")
[ -n "${DRAFT_SAMPLING:-}" ] && SPEC+=(--spec-draft-sampling "$DRAFT_SAMPLING")
[ "${SPEC_OFF:-0}" = "1" ] && SPEC=()

echo "[run-main] BIN=$BIN"
echo "[run-main] CTX=$CTX NCM=$NCM TS=${TS:-default} MG=$MG KV=$KVK/$KVV UB=${UB:-default} SPEC_UB=${SPEC_UB:-} MOE_CACHE=${MOE_CACHE:-none} VISION=$VISION NP=$NP PORT=$PORT CKPT=${CKPT:-default} T=${T:-default(-1)} MTP=$([ "${SPEC_OFF:-0}" = 1 ] && echo off || echo "on(n=$NMAX ${DRAFT_SAMPLING:-greedy})") REGHOST=${GGML_CUDA_REGISTER_HOST:-0}"
echo "[run-main] -ot $PLE_OT"

exec env CUDA_VISIBLE_DEVICES="$GPU_BOTH" GGML_CUDA_REGISTER_HOST="${GGML_CUDA_REGISTER_HOST:-0}" "$BIN" \
  -m "$MODEL" \
  "${MM[@]}" \
  "${SPEC[@]}" \
  -c "$CTX" --cache-type-k "$KVK" --cache-type-v "$KVV" \
  "${UBA[@]}" \
  -ngl 99 -sm layer "${TSA[@]}" -mg "$MG" -fa on \
  -ot "$PLE_OT" \
  --n-cpu-moe "$NCM" \
  "${MCA[@]}" "${WARM[@]}" "${LM[@]}" "${LVA[@]}" "${CKA[@]}" "${TA[@]}" \
  --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0.0 --presence-penalty 0.0 --jinja \
  --host 127.0.0.1 --port "$PORT" --parallel "$NP"
