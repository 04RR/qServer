#!/usr/bin/env bash
# Benchmark a running llama-server: tok/s (pp + tg), TTFT, MTP draft acceptance, VRAM.
# Usage: ./bench.sh [PORT] [GPU_INDEX]   (default: 8080 0  = Server A text)
#   A/B MTP: run Server A once with SPEC_NMAX=2/3 and once via a no-spec launch, compare.
set -uo pipefail
PORT="${1:-8080}"
GPU="${2:-0}"
HOST="http://127.0.0.1:$PORT"

command -v jq >/dev/null || { echo "need jq"; exit 1; }

vram() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$GPU" | tr -d ' '; }

run_one() {
  local label="$1" prompt="$2" npred="$3"
  local body resp
  body=$(jq -n --arg p "$prompt" --argjson n "$npred" \
    '{prompt:$p, n_predict:$n, temperature:0.6, top_p:0.95, top_k:20, cache_prompt:true, stream:false}')
  resp=$(curl -s "$HOST/completion" -H 'Content-Type: application/json' -d "$body")
  echo "── $label ──"
  echo "$resp" | jq -r '
    .timings as $t |
    "  prompt_n        : \($t.prompt_n)   @ \(($t.prompt_per_second // 0)|floor) t/s (pp)",
    "  predicted_n     : \($t.predicted_n)   @ \(($t.predicted_per_second // 0)|floor) t/s (tg)",
    "  ttft (prompt_ms): \(($t.prompt_ms // 0)|floor) ms",
    (if $t.draft_n != null then
       "  draft_n         : \($t.draft_n)",
       "  draft_accepted  : \($t.draft_n_accepted)  (\(((($t.draft_n_accepted // 0)*100)/(($t.draft_n // 1)))|floor)% acceptance)"
     else "  draft stats     : (none in timings; check server log for MTP acceptance)" end)
  ' 2>/dev/null || { echo "  raw timings:"; echo "$resp" | jq '.timings'; }
}

echo "=== bench $HOST  (GPU$GPU VRAM before: $(vram) MiB) ==="
# warm the cache / short code
run_one "short-code"   "Write a Python function that returns the nth Fibonacci number iteratively." 200
# long-form / low-predictability (MTP tends to help less here)
run_one "creative"     "Write an original 200-word story about a lighthouse keeper who collects fog." 300
# reasoning
run_one "reasoning"    "A bat and ball cost 1.10 total. The bat costs 1.00 more than the ball. How much is the ball? Think step by step." 400
echo "=== GPU$GPU VRAM after: $(vram) MiB ==="
echo
echo "For per-run MTP acceptance also grep the server log for 'draft' / 'accept'."
