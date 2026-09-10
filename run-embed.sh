#!/usr/bin/env bash
# Qwen3-Embedding-0.6B — embedding server, CONCURRENT with the chat models (own non-exclusive swap group).
# CPU by default (NGL=0, CUDA_VISIBLE_DEVICES="" -> zero VRAM footprint) so it coexists with ALL chat
# models, including the 122B which needs the whole GPU. Serves OpenAI /v1/embeddings on :8086.
set -euo pipefail
ROOT=~/ai/qwen36
BIN="$ROOT/llama.cpp/build/bin/llama-server"
MODEL="${MODEL:-$ROOT/models/Qwen3-Embedding-0.6B-f16.gguf}"   # f16: embeddings are quantization-sensitive
CTX="${CTX:-8192}"                 # max input tokens per embed (RAG chunks are far smaller)
NGL="${NGL:-0}"                    # 0 = CPU (coexists with everything). >0 = GPU (faster bulk index, but
                                   #   then it competes with the 122B for VRAM — keep it 0 unless you know).
THREADS="${THREADS:-8}"            # modest, so query-embed doesn't starve a resident chat model's CPU work
POOLING="${POOLING:-last}"         # Qwen3-Embedding uses LAST-token pooling
PORT="${PORT:-8086}"               # dedicated port (chat models use 8080/8082/8084)

# CPU: hide all GPUs so the CUDA build allocates ZERO VRAM. GPU: pin to the spare 5060 Ti (GPU1).
if [ "$NGL" = "0" ]; then PIN=(env CUDA_VISIBLE_DEVICES=); else PIN=(env CUDA_VISIBLE_DEVICES="${DEV:-1}"); fi

# -b/-ub == CTX so an input up to CTX tokens is processed in ONE ubatch (required for correct pooling).
exec "${PIN[@]}" "$BIN" \
  -m "$MODEL" \
  --alias qwen-embed \
  --embedding --pooling "$POOLING" --embd-normalize 2 \
  -ngl "$NGL" -c "$CTX" -b "$CTX" -ub "$CTX" \
  -t "$THREADS" \
  --host 0.0.0.0 --port "$PORT"
