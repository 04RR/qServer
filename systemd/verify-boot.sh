#!/usr/bin/env bash
# Run AFTER `wsl --shutdown` + reopening WSL, to prove the systemd stack came up correctly.
# (systemd "degraded" is fine -- it means some *other* unit failed, not ours; we check ours explicitly.)
#
# REWRITTEN for the llama-swap architecture. The old version asserted "GPU0 > 18000 MiB AND
# GPU1 > 10000 MiB" because both servers were always resident. That assertion is now WRONG in
# two ways: (1) vision is retired so GPU1 is idle unless the 35B is loaded, and (2) models are
# load-on-demand (ttl:0), so at boot NO model is loaded and BOTH GPUs are legitimately at ~0.
# An idle GPU is now the correct boot state, not a failure. We prove the stack by *exercising*
# a swap instead of by looking at resident VRAM.
set -uo pipefail
export PATH=$HOME/.local/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:${LD_LIBRARY_PATH:-}
fail=0

echo "== system state =="
systemctl is-system-running || true   # 'degraded' is acceptable; we verify our units below

echo; echo "== our units =="
for u in qwen-swap qwen-router; do
  st=$(systemctl is-active "$u" 2>/dev/null)
  printf "  %-14s %s\n" "$u" "$st"
  [ "$st" = "active" ] || { echo "    !! not active:"; systemctl --no-pager status "$u" | tail -5; fail=1; }
done

echo; echo "== retired units must stay retired (else they steal port 8080 + GPU0 at boot) =="
for u in qwen-text qwen-vision; do
  en=$(systemctl is-enabled "$u" 2>&1)
  printf "  %-14s enabled=%s\n" "$u" "$en"
  case "$en" in disabled|masked|not-found) ;; *) echo "    !! $u is $en — it will fight llama-swap for port 8080/GPU0"; fail=1;; esac
done

echo; echo "== endpoint =="
# RETRY, do not assume. `systemctl restart` returns when ExecStart is SPAWNED (Type=simple), not
# when uvicorn has bound :8000 -- running this script immediately after a restart raced ahead of
# the bind by ~1s and reported a false FAIL (and then a false "no models", since that check hit
# the same dead socket). Give it 30s to come up.
up=0
for i in $(seq 1 30); do
  curl -sf -m5 http://127.0.0.1:8000/healthz >/dev/null 2>&1 && { up=1; break; }
  sleep 1
done
if [ "$up" = 1 ]; then curl -sf -m5 http://127.0.0.1:8000/healthz; echo
else echo "  !! :8000 not answering after 30s"; fail=1; fi

echo; echo "== models registered =="
ids=$(curl -sf -m5 http://127.0.0.1:8000/v1/models | jq -r '.data[].id' 2>/dev/null)   # one id per line
echo "  $(echo "$ids" | paste -sd, -)"
# grep -qx (whole-line exact), NOT -q (substring): 'qwen-27b' is a substring of 'qwen-27b-q6', so -q
# would report the Q6 present even if only the Q4 were registered. Match exact ids, newline-separated.
for m in qwen-27b qwen-27b-q6 qwen-35b qwen-122b; do
  echo "$ids" | grep -qx "$m" || { echo "  !! missing model: $m"; fail=1; }
done

echo; echo "== end-to-end: 27B (via legacy alias 'qwen36') =="
a=$(curl -sf -m600 http://127.0.0.1:8000/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"qwen36","messages":[{"role":"user","content":"one word: capital of France"}],"max_tokens":2048}' \
  | jq -r '.choices[0].message.content // ""' 2>/dev/null)
echo "  answer: $(echo "$a" | tr -d '\n' | head -c 40)"
echo "$a" | grep -qi paris || { echo "  !! unexpected answer"; fail=1; }
r=$(curl -sf -m5 http://127.0.0.1:8000/healthz | jq -r '.backend.running|join(",")')
v0=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0|tr -d ' ')
v1=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 1|tr -d ' ')
echo "  running=$r  GPU0=${v0} GPU1=${v1}"
[ "$r" = "qwen-27b" ] || { echo "  !! alias qwen36 did not resolve to qwen-27b"; fail=1; }
# the 27B must NOT touch GPU1 -- if it does, pinning was lost
[ "$v1" -lt 1000 ] 2>/dev/null || { echo "  !! 27B is using GPU1 (${v1} MiB) — pinning lost"; fail=1; }

echo; echo "== end-to-end: swap to 35B, and PROVE the 27B was evicted first =="
a2=$(curl -sf -m900 http://127.0.0.1:8000/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"qwen-35b","messages":[{"role":"user","content":"one word: capital of Japan"}],"max_tokens":2048}' \
  | jq -r '.choices[0].message.content // ""' 2>/dev/null)
echo "  answer: $(echo "$a2" | tr -d '\n' | head -c 40)"
echo "$a2" | grep -qi tokyo || { echo "  !! unexpected answer"; fail=1; }
r2=$(curl -sf -m5 http://127.0.0.1:8000/healthz | jq -r '.backend.running|join(",")')
v0b=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0|tr -d ' ')
v1b=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 1|tr -d ' ')
echo "  running=$r2  GPU0=${v0b} GPU1=${v1b}"
[ "$r2" = "qwen-35b" ] || { echo "  !! swap did not land on qwen-35b"; fail=1; }
# exactly one model may be resident. If the 27B had NOT been evicted, GPU0 would be ~23G + the
# 35B's 21G = impossible; the load would have OOM'd. Reaching min-spill residency proves eviction.
if [ "$v0b" -gt 18000 ] && [ "$v1b" -gt 2000 ] && [ "$v1b" -lt 6000 ] 2>/dev/null; then
  echo "  OK: 35B min-spill residency (GPU0=${v0b} GPU1=${v1b}) — 27B was fully evicted first"
else
  echo "  !! 35B residency off-plan: GPU0=${v0b} GPU1=${v1b} (expect ~21000/~3000)"; fail=1
fi

echo; echo "== end-to-end: swap to the 122B (TRI-TIER: GPU0 + GPU1 + system RAM) =="
# ~40-50s to load 48 GiB across three tiers; longer from cold page cache.
a3=$(curl -sf -m900 http://127.0.0.1:8000/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"qwen-122b","messages":[{"role":"user","content":"one word: capital of Italy"}],"max_tokens":4096}' \
  | jq -r '.choices[0].message.content // ""' 2>/dev/null)
echo "  answer: $(echo "$a3" | tr -d '\n' | head -c 40)"
echo "$a3" | grep -qi rome || { echo "  !! unexpected answer"; fail=1; }
r3=$(curl -sf -m5 http://127.0.0.1:8000/healthz | jq -r '.backend.running|join(",")')
v0c=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0|tr -d ' ')
v1c=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 1|tr -d ' ')
rss=$(awk '/VmRSS/{print int($2/1024)}' /proc/$(pgrep -f "bin/llama-server"|head -1)/status 2>/dev/null)
echo "  running=$r3  GPU0=${v0c} GPU1=${v1c} RSS=${rss} MiB"
[ "$r3" = "qwen-122b" ] || { echo "  !! swap did not land on qwen-122b"; fail=1; }
# Tri-tier residency is only reachable if the PREVIOUS model was fully evicted first (23 GiB of
# GPU0 + both cards + a ~15 GiB RAM slice cannot coexist with it). So this doubles as the
# eviction assertion -- we prove the stack by exercising it, not by reading resident VRAM.
if [ "$v0c" -gt 20000 ] && [ "$v1c" -gt 12000 ] && [ "$rss" -gt 8000 ] 2>/dev/null; then
  echo "  OK: 122B tri-tier residency (GPU0=${v0c} GPU1=${v1c} RAM=${rss}) — prior model fully evicted"
else
  echo "  !! 122B residency off-plan: GPU0=${v0c} GPU1=${v1c} RSS=${rss} (expect ~23900/~14650/~17000)"; fail=1
fi

echo
[ "$fail" -eq 0 ] && echo "BOOT VERIFY: PASS" || { echo "BOOT VERIFY: FAIL"; exit 1; }
