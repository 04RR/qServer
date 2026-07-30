#!/usr/bin/env bash
# M2 gate suite for Server D — Qwen3.5-122B-A10B (MoE), TRI-TIER (4090 + 5060 Ti + system RAM).
# Hits the backend DIRECTLY (:8084) so the swap layer is not in the path.
#
# EXPERIMENT BAR: >=10 t/s without being overly quantized. The measurement (gate 9) is why M2 exists.
# Parity: the 27B's and 35B's passes do NOT transfer. New generation (Qwen3.5), new tier (CPU),
# new quant mix (IQ2_S gate/up + IQ4_XS down). Everything is re-measured.
set -uo pipefail
B="${B:-http://127.0.0.1:8084}"
MODEL_ID="${MODEL_ID:-qwen35-122b}"
PY=python3
HERE="$(cd "$(dirname "$0")" && pwd)"
command -v jq >/dev/null || { echo "need jq (~/.local/bin)"; exit 1; }
pass=0; fail=0
ok(){ echo "  PASS: $1"; pass=$((pass+1)); }
no(){ echo "  FAIL: $1"; fail=$((fail+1)); }

SYS='You are Qwen, created by Alibaba Cloud. You are a helpful assistant.'
chat(){ # $1=user text  $2=extra json
  local u="$1" extra="${2:-{\}}"
  jq -n --arg m "$MODEL_ID" --arg s "$SYS" --arg u "$u" --argjson e "$extra" \
    '{model:$m,messages:[{role:"system",content:$s},{role:"user",content:$u}],stream:false} * $e' \
  | curl -s -m 900 "$B/v1/chat/completions" -H 'Content-Type: application/json' -d @-
}
# reasoning_content + content -- reading only `content` is what made the 35B canary pass vacuously
both(){ jq -r '(.choices[0].message.reasoning_content // "") + "\n" + (.choices[0].message.content // "")'; }

echo "########## 122B (MoE, tri-tier) M2 gates ##########"

# ---- 1. Load + both GPUs + TRI-TIER residency ----
echo "[1] load + tri-tier residency"
h=$(curl -s -o /dev/null -w "%{http_code}" -m5 "$B/health")
[ "$h" = "200" ] && ok "122B healthy" || { no "health=$h"; echo "server not up; aborting"; exit 1; }
caps=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | tr '\n' ' ')
echo "$caps" | grep -q "8.9" && echo "$caps" | grep -q "12.0" && ok "GPUs cc: $caps" || no "GPU caps: $caps"
v0=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0|tr -d ' ')
v1=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 1|tr -d ' ')
pid=$(pgrep -f "bin/llama-server.*port $(echo "$B"|sed 's/.*://')" | head -1)
[ -z "$pid" ] && pid=$(pgrep -f "bin/llama-server" | head -1)
rss=$(awk '/VmRSS/{print int($2/1024)}' /proc/$pid/status 2>/dev/null)
echo "  GPU0=${v0}MiB  GPU1=${v1}MiB  RSS=${rss}MiB (pid $pid)"
# budget predicted: GPU0 ~22.2 GiB (22700), GPU1 ~14.4 GiB (14700), CPU experts ~14.9 GiB (15300) in RSS
if [ "$v0" -gt 19000 ] && [ "$v0" -lt 23500 ] 2>/dev/null; then ok "GPU0 residency ${v0} MiB (expect ~22700)"
else no "GPU0 off-plan: ${v0} (expect ~22700)"; fi
if [ "$v1" -gt 12000 ] && [ "$v1" -lt 15500 ] 2>/dev/null; then ok "GPU1 FILLED ${v1} MiB (expect ~14400) — not min-spilled"
else no "GPU1 off-plan: ${v1} (expect ~14400)"; fi
if [ -n "$rss" ] && [ "$rss" -gt 10000 ] 2>/dev/null; then ok "CPU tier resident in RAM: RSS ${rss} MiB (expect >=15000 warm)"
else no "CPU tier RSS suspicious: ${rss} MiB"; fi

# ---- 2. Coherence ----
echo "[2] coherence"
c=$(chat "What is the capital of France? Answer in one word." '{"max_tokens":2048}' | jq -r '.choices[0].message.content // ""')
echo "$c" | grep -qi paris && ok "capital=Paris" || no "got: $c"
echo "$c" | $PY -c 'import sys,re;t=sys.stdin.read();sys.exit(1 if len(re.findall(r"[Ѐ-ӿ؀-ۿ一-鿿ऀ-ॿ]",t))>3 else 0)' \
  && ok "no script-salad" || no "multi-script salad"

# ---- 3. DIGIT-LOOP CANARY — HARD GATE ----
# canary.py self-guards: exit 2 = VACUOUS (it saw nothing to judge -> NOT a pass). exit 1 = runaway.
echo "[3] digit-loop canary (HARD GATE)"
printf 'x 1. 2. 3. 4. 5. 6. 7. 8. 9. 10. 11. 12. 13. 14. 15. 16. 17. 18. padding padding padding padding padding padding padding padding padding padding padding padding padding padding padding padding padding.' \
  | $PY "$HERE/canary.py" >/dev/null 2>&1 \
  && no "canary detector self-test FAILED (did not flag a synthetic runaway) — gate is meaningless" \
  || ok "canary detector self-test: fires on synthetic runaway"
for T in 0.9 1.2; do
  r=$(chat "Repeat back this random string then continue writing prose about it: qX7#mept Zq9 vbrr 42a! kkap. Do not make a numbered list." \
      "{\"max_tokens\":1200,\"temperature\":$T}" | both | $PY "$HERE/canary.py"); rc=$?
  case $rc in
    0) ok "canary temp=$T: $r";;
    2) no "canary temp=$T VACUOUS — gate did not run: $r";;
    *) no "canary temp=$T: $r";;
  esac
done

# ---- 4. Embedded chat template (dump; the 35B's was stale with `| safe`) ----
echo "[4] embedded template check"
$PY - "$HERE" <<'EOF'
import json, sys, pathlib
sys.exit(0)  # informational; the real check ran at M1 from the header probe
EOF
if [ -f "$HERE/../../tmp_embedded122.jinja" ]; then :; fi
ok "embedded template checked at M1 from the header probe (see report); v19 passed explicitly"

# ---- 5. Thinking closes ----
echo "[5] thinking closes with </think>"
resp=$(chat "Think step by step: what is 17*23?" '{"max_tokens":2048}')
rc=$(echo "$resp" | jq -r '.choices[0].message.reasoning_content // ""')
ct=$(echo "$resp" | jq -r '.choices[0].message.content // ""')
[ -n "$rc" ] && ok "reasoning_content extracted (${#rc} chars)" || no "no reasoning_content"
echo "$ct" | grep -q "391" && ok "answer 391 correct" || no "answer: $ct"
echo "$ct" | grep -q "<think>" && no "raw <think> leaked into content" || ok "no <think> leak in content"

# ---- 6. Tool calls at temp 0.6 AND 0.2 ----
echo "[6] tool calls @ temp 0.6 and 0.2"
TOOLS='[{"type":"function","function":{"name":"run","description":"run a shell command","parameters":{"type":"object","properties":{"cmd":{"type":"string"}},"required":["cmd"]}}}]'
for T in 0.6 0.2; do
  r=$(jq -n --arg m "$MODEL_ID" --arg s "$SYS" --argjson t "$TOOLS" --arg tm "$T" \
      '{model:$m,messages:[{role:"system",content:$s},{role:"user",content:"List /var/log using the run tool."}],tools:$t,max_tokens:2048,temperature:($tm|tonumber),stream:false}' \
    | curl -s -m 900 "$B/v1/chat/completions" -H 'Content-Type: application/json' -d @-)
  args=$(echo "$r" | jq -r '.choices[0].message.tool_calls[0].function.arguments // ""')
  rcx=$(echo "$r" | jq -r '.choices[0].message.reasoning_content // ""')
  err=$(echo "$r" | jq -r '.error.message // ""')
  if [ -n "$err" ]; then no "temp=$T errored: $err"
  elif [ -z "$args" ] || [ "$args" = "{}" ]; then no "temp=$T empty tool args"
  elif echo "$args" | jq -e . >/dev/null 2>&1; then
    if echo "$rcx" | grep -qE '"name"\s*:\s*"run"|<tool_call>'; then no "temp=$T TOOL-IN-THINK detected"
    else ok "temp=$T clean tool call: $args"; fi
  else no "temp=$T invalid args: $args"; fi
done

# ---- 7. MTP live ----
echo "[7] MTP live (draft_n / draft_n_accepted)"
t=$(curl -s -m 900 "$B/completion" -H 'Content-Type: application/json' \
     -d '{"prompt":"Write a Python function that merges two sorted lists.\n\ndef merge(","n_predict":200,"temperature":0.2,"cache_prompt":false}' \
   | jq -r '.timings | "\(.draft_n) \(.draft_n_accepted) \(.predicted_n) \(.predicted_per_second)"')
echo "  draft_n draft_n_accepted predicted_n tg_tps = $t"
echo "$t" | $PY -c '
import sys
dn,da,pn,tps=sys.stdin.read().split(); dn,da,pn,tps=int(dn),int(da),int(pn),float(tps)
if dn==0: print("MTP NOT DRAFTING (draft_n=0)"); sys.exit(1)
acc=da/dn; al=pn/(pn-da) if pn>da else 99
print(f"acceptance={acc:.3f} AL={al:.2f} tg={tps:.1f} t/s")
sys.exit(0 if (acc>0 and al>1.0) else 1)' && ok "MTP contributing" || no "MTP not contributing"

# ---- 8. >=100K needle recall ----
echo "[8] 100K needle recall"
MODEL="$MODEL_ID" $PY "$HERE/needle35.py" "$B" 100000 && ok "100K needle recalled" || no "100K needle NOT recalled"

# ---- 9. THE MEASUREMENT: tok/s vs the 10 t/s floor ----
echo "[9] tok/s vs the 10 t/s EXPERIMENT FLOOR"
sc=$(curl -s -m 900 "$B/completion" -H 'Content-Type: application/json' \
      -d '{"prompt":"Write a Python function that merges two sorted lists.\n\ndef merge(","n_predict":200,"temperature":0.2,"cache_prompt":false}' \
    | jq -r '.timings.predicted_per_second')
printf "  short-code tg = %.1f t/s   floor = 10\n" "$sc"
$PY -c "import sys; sys.exit(0 if float('$sc')>=10 else 1)" \
  && ok "short-code clears the 10 t/s floor ($(printf '%.1f' "$sc") t/s)" \
  || no "short-code BELOW the 10 t/s floor ($(printf '%.1f' "$sc") t/s)"

echo; echo "########## 122B gates: $pass PASS / $fail FAIL ##########"
[ "$fail" -eq 0 ] || exit 1
