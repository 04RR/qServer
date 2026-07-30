#!/usr/bin/env bash
# Router on :8000 -> llama-swap :9000 -> llama-server (qwen-27b | qwen-35b).
# Vision is retired, so the image-content routing is gone. The router's remaining job is the
# tool-budget guard (llama-swap's filters.setParams is an unconditional set, not a conditional
# floor, so it cannot express it -- see proxy.py) plus mandatory system-prompt injection.
set -euo pipefail
cd "$(dirname "$0")"
export SWAP_BACKEND="${SWAP_BACKEND:-http://127.0.0.1:9000}"
export INJECT_SYSTEM="${INJECT_SYSTEM:-1}"
export TOOL_MIN_TOKENS="${TOOL_MIN_TOKENS:-512}"
exec .venv/bin/python -m uvicorn proxy:app --host 0.0.0.0 --port 8000 --no-access-log
