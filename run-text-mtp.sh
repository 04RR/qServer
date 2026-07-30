#!/usr/bin/env bash
# Server A — TEXT + MTP, RTX 4090 (CUDA0). Max speed, >=100K context. NO mmproj (keeps MTP alive).
set -euo pipefail

export PATH=/usr/local/cuda-12.8/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:${LD_LIBRARY_PATH:-}

ROOT=~/ai/qwen36
BIN="$ROOT/llama.cpp/build/bin/llama-server"          # always absolute — never PATH-resolved
MODEL="$ROOT/models/Qwen3.6-27B-UD-Q4_K_XL.gguf"
TMPL="$ROOT/templates/qwen3.6-v19.jinja"

# --- tunables (overridable via env; see bench.sh / M4) ---
CTX="${CTX:-131072}"                # >=100K hard requirement
KV="${KV:-q8_0}"                    # M4: q8_0 chosen (no retrieval degradation at 100K; headroom is there)
SPEC_NMAX="${SPEC_NMAX:-4}"         # M4 sweep: code peaks at n=4 (n=5 regresses); n=2/3 slower on code
NGL="${NGL:-99}"
MTP="${MTP:-on}"                    # on|off — set off to benchmark the no-speculation baseline

SPEC_ARGS=(--spec-type draft-mtp --spec-draft-n-max "$SPEC_NMAX")
[ "$MTP" = "off" ] && SPEC_ARGS=(--spec-type none)

exec env CUDA_VISIBLE_DEVICES=0 "$BIN" \
  -m "$MODEL" \
  --alias qwen36-text \
  -ngl "$NGL" -c "$CTX" -fa on -np 1 \
  "${SPEC_ARGS[@]}" \
  --cache-type-k "$KV" --cache-type-v "$KV" \
  --no-context-shift --ctx-checkpoints 32 \
  --jinja --chat-template-file "$TMPL" \
  --reasoning-preserve \
  --temp 0.6 --top-p 0.95 --top-k 20 --min-p 0.0 \
  --presence-penalty 0.0 --repeat-penalty 1.0 \
  --host 0.0.0.0 --port 8080
# NOTE: no --mmproj here — that omission is what keeps the MTP draft path accelerating.
# chat-template-kwargs: NO space after the colon (Qwen silently ignores it otherwise).
