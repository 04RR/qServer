#!/usr/bin/env bash
# Server B — VISION, RTX 5060 Ti (CUDA1). mmproj loaded, NO MTP. Correctness > speed.
set -euo pipefail

export PATH=/usr/local/cuda-12.8/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:${LD_LIBRARY_PATH:-}

ROOT=~/ai/qwen36
BIN="$ROOT/llama.cpp/build/bin/llama-server"          # always absolute
MODEL="$ROOT/models/Qwen3.6-27B-UD-Q3_K_XL.gguf"      # non-MTP repo; only quant clearing the 1.5 GiB VRAM floor
MMPROJ="$ROOT/models/mmproj-F16.gguf"                 # F16 (no Q8_0 -> avoids Blackwell #24399)
TMPL="$ROOT/templates/qwen3.6-v19.jinja"

# --- tunables ---
CTX="${CTX:-32768}"                # 32K target; drop to 24576 if compute buffers spike at boot
KV="${KV:-q4_0}"                   # q4_0 to fit the 16 GB card
NGL="${NGL:-99}"

exec env CUDA_VISIBLE_DEVICES=1 "$BIN" \
  -m "$MODEL" \
  --mmproj "$MMPROJ" \
  --alias qwen36-vision \
  -ngl "$NGL" -c "$CTX" -fa on -np 1 \
  --cache-type-k "$KV" --cache-type-v "$KV" \
  --image-min-tokens 1024 \
  --no-context-shift \
  --jinja --chat-template-file "$TMPL" \
  --reasoning-preserve \
  --temp 0.6 --top-p 0.95 --top-k 20 --min-p 0.0 \
  --host 0.0.0.0 --port 8081
# NO --spec-type here: MTP gives no gain with vision and risks slot-position corruption/OOM (#22867/#23371).
