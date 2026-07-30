#!/usr/bin/env bash
# M2 gate suite for Server C — Qwen3.6-35B-A3B MoE, cross-GPU expert offload.
# Hits the 35B backend DIRECTLY (:8082) so the swap/router layer is not in the path.
#
# Parity note: the 27B's passes DO NOT transfer. Different repo (separate conversion), different
# arch (MoE), and the MTP verify loop now re-decodes through a cross-GPU expert path the dense
# 27B never exercised. Every gate below is re-measured on this model.
set -uo pipefail
B="${B:-http://127.0.0.1:8082}"
PY=python3
command -v jq >/dev/null || { echo "need jq (~/.local/bin)"; exit 1; }
pass=0; fail=0
ok(){ echo "  PASS: $1"; pass=$((pass+1)); }
no(){ echo "  FAIL: $1"; fail=$((fail+1)); }

SYS='You are Qwen, created by Alibaba Cloud. You are a helpful assistant.'
chat(){ # $1=user text  $2=extra json
  local u="$1" extra="${2:-{\}}"
  jq -n --arg s "$SYS" --arg u "$u" --argjson e "$extra" \
    '{model:"qwen36-35b",messages:[{role:"system",content:$s},{role:"user",content:$u}],stream:false} * $e' \
  | curl -s -m 300 "$B/v1/chat/completions" -H 'Content-Type: application/json' -d @-
}

echo "########## 35B (MoE) M2 gates ##########"

# ---- 1. Load + both GPUs enumerate + cross-GPU residency ----
echo "[1] load + dual-GPU residency"
h=$(curl -s -o /dev/null -w "%{http_code}" -m5 "$B/health")
[ "$h" = "200" ] && ok "35B healthy" || { no "health=$h"; echo "server not up; aborting"; exit 1; }
caps=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | tr '\n' ' ')
echo "$caps" | grep -q "8.9" && echo "$caps" | grep -q "12.0" && ok "GPUs cc: $caps" || no "GPU caps: $caps"
v0=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0|tr -d ' ')
v1=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 1|tr -d ' ')
echo "  GPU0=${v0}MiB  GPU1=${v1}MiB"
# min-spill expectation: GPU0 heavily loaded (~21G), GPU1 lightly (~3.3G). If GPU1 is ~0 the -ot
# never took; if GPU1 is huge, the split is wrong (or -ts distributed layers behind our back).
if [ "$v0" -gt 18000 ] && [ "$v1" -gt 2000 ] && [ "$v1" -lt 6000 ] 2>/dev/null; then
  ok "min-spill residency: GPU0=${v0} GPU1=${v1} (expect ~21000/~3300)"
else
  no "residency off-plan: GPU0=${v0} GPU1=${v1} (expect ~21000/~3300) — -ot may not have applied"
fi

# ---- 2. Coherence ----
echo "[2] coherence"
c=$(chat "What is the capital of France? Answer in one word." '{"max_tokens":2048}' | jq -r '.choices[0].message.content // ""')
echo "$c" | grep -qi paris && ok "capital=Paris" || no "got: $c"
echo "$c" | $PY -c 'import sys,re;t=sys.stdin.read();sys.exit(1 if len(re.findall(r"[Ѐ-ӿ؀-ۿ一-鿿ऀ-ॿ]",t))>3 else 0)' \
  && ok "no script-salad" || no "multi-script salad"

# ---- 3. DIGIT-LOOP CANARY — HARD GATE (cross-GPU MTP verify correctness) ----
# This is the one genuinely untested path in the build: the MTP head's id_last/rollback logic now
# re-decodes through experts living on a different device. Force draft REJECTIONS with high-entropy
# text; a broken verify loop shows up as an ascending-integer runaway or single-token spam.
echo "[3] digit-loop canary (HARD GATE)"
# CRITICAL: with --reasoning-preserve, generation lands in reasoning_content and `content` can be
# EMPTY (finish_reason=length while predicted_n=600). Reading only `content` made this gate pass
# vacuously on a 1-char string. Analyze reasoning_content + content together — the reasoning is
# where the bulk of tokens are generated, so it is exactly where a broken MTP verify loop shows.
canary(){ # $1=temp
  chat "Repeat back this random string then continue writing prose about it: qX7#mept Zq9 vbrr 42a! kkap. Do not make a numbered list." \
       "{\"max_tokens\":1200,\"temperature\":$1}" \
  | jq -r '(.choices[0].message.reasoning_content // "") + "\n" + (.choices[0].message.content // "")'
}
# NOTE: the detector lives in canary.py, NOT inline. Inline `$PY -c '...'` broke: the Python string
# literals 'ok'/'RUNAWAY' terminated bash's single-quoted -c argument, so the detector died with
# NameError and the gate reported FAIL without ever testing the model. A detector that cannot fire
# is worse than no detector — canary.py is self-tested against a synthetic runaway (see below).
CANARY="${CANARY:-$(dirname "$0")/canary.py}"
# self-test the detector before trusting any PASS from it
printf 'x 1. 2. 3. 4. 5. 6. 7. 8. 9. 10. 11. 12. 13. 14. 15. 16. 17.' | $PY "$CANARY" >/dev/null 2>&1 \
  && { no "canary detector self-test FAILED (did not flag a synthetic runaway) — gate is meaningless"; } \
  || ok "canary detector self-test: fires on synthetic runaway"
for T in 0.9 1.2; do
  r=$(canary $T | $PY "$CANARY"); rc=$?
  [ $rc -eq 0 ] && ok "canary temp=$T: $r" || no "canary temp=$T: $r"
done

# ---- 4. Thinking closes ----
echo "[4] thinking closes with </think>"
resp=$(chat "Think step by step: what is 17*23?" '{"max_tokens":2048}')
rc=$(echo "$resp" | jq -r '.choices[0].message.reasoning_content // ""')
ct=$(echo "$resp" | jq -r '.choices[0].message.content // ""')
[ -n "$rc" ] && ok "reasoning_content extracted (${#rc} chars)" || no "no reasoning_content"
echo "$ct" | grep -q "391" && ok "answer 391 correct" || no "answer: $ct"
echo "$ct" | grep -q "<think>" && no "raw <think> leaked into content" || ok "no <think> leak in content"

# ---- 5. Tool calls at temp 0.6 AND 0.2 (tool-in-think re-test; 27B pass does not transfer) ----
echo "[5] tool calls @ temp 0.6 and 0.2"
TOOLS='[{"type":"function","function":{"name":"run","description":"run a shell command","parameters":{"type":"object","properties":{"cmd":{"type":"string"}},"required":["cmd"]}}}]'
for T in 0.6 0.2; do
  r=$(jq -n --arg s "$SYS" --argjson t "$TOOLS" --arg tm "$T" \
      '{model:"qwen36-35b",messages:[{role:"system",content:$s},{role:"user",content:"List /var/log using the run tool."}],tools:$t,max_tokens:2048,temperature:($tm|tonumber),stream:false}' \
    | curl -s -m 300 "$B/v1/chat/completions" -H 'Content-Type: application/json' -d @-)
  args=$(echo "$r" | jq -r '.choices[0].message.tool_calls[0].function.arguments // ""')
  rc=$(echo "$r" | jq -r '.choices[0].message.reasoning_content // ""')
  err=$(echo "$r" | jq -r '.error.message // ""')
  if [ -n "$err" ]; then no "temp=$T errored: $err"
  elif [ -z "$args" ] || [ "$args" = "{}" ]; then no "temp=$T empty tool args"
  elif echo "$args" | jq -e . >/dev/null 2>&1; then
    # tool-in-think = the model emitted the call inside its reasoning instead of as a tool_call
    if echo "$rc" | grep -qE '"name"\s*:\s*"run"|<tool_call>'; then no "temp=$T TOOL-IN-THINK detected"
    else ok "temp=$T clean tool call: $args"; fi
  else no "temp=$T invalid args: $args"; fi
done

# ---- 6. MTP live: acceptance > 0 and AL > 1 ----
echo "[6] MTP live (draft_n / draft_n_accepted)"
t=$(curl -s -m 300 "$B/completion" -H 'Content-Type: application/json' \
     -d '{"prompt":"Write a Python function that merges two sorted lists.\n\ndef merge(","n_predict":220,"temperature":0.2,"cache_prompt":false}' \
   | jq -r '.timings | "\(.draft_n) \(.draft_n_accepted) \(.predicted_n) \(.predicted_per_second)"')
echo "  draft_n draft_n_accepted predicted_n tg_tps = $t"
echo "$t" | $PY -c '
import sys
dn,da,pn,tps=sys.stdin.read().split()
dn,da,pn,tps=int(dn),int(da),int(pn),float(tps)
if dn==0: print("MTP NOT DRAFTING (draft_n=0)"); sys.exit(1)
acc=da/dn; al=pn/(pn-da) if pn>da else 99
print(f"acceptance={acc:.3f} AL={al:.2f} tg={tps:.1f} t/s")
sys.exit(0 if (acc>0 and al>1.0) else 1)' && ok "MTP contributing" || no "MTP not contributing"

# ---- 7. >=100K needle recall ----
# Self-calibrating: the 27B's hardcoded 10500-sentence filler tokenizes to ~188K here (~18
# tok/sentence vs the 27B's ~10) and 400s against n_ctx. needle35.py calibrates via /tokenize.
echo "[7] 100K needle recall"
$PY "$(dirname "$0")/needle35.py" "$B" 100000 && ok "100K needle recalled" || no "100K needle NOT recalled"

# ---- 8. Cross-GPU sustained decode: no device exception (#24399 surface check) ----
echo "[8] sustained cross-GPU decode (no device exception)"
errs=0
for i in 1 2 3 4 5 6; do
  r=$(chat "Write a detailed 300-word explanation of how a B-tree insert rebalances." '{"max_tokens":700,"temperature":0.7}')
  e=$(echo "$r" | jq -r '.error.message // ""')
  [ -n "$e" ] && { errs=$((errs+1)); echo "    err: $e"; }
done
h2=$(curl -s -o /dev/null -w "%{http_code}" -m5 "$B/health")
v0b=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0|tr -d ' ')
v1b=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 1|tr -d ' ')
[ "$errs" -eq 0 ] && [ "$h2" = "200" ] && ok "6/6 sustained gens clean, server alive, VRAM ${v0b}/${v1b}" \
  || no "$errs/6 errored (health=$h2)"

echo; echo "########## 35B gates: $pass PASS / $fail FAIL ##########"
[ "$fail" -eq 0 ] || exit 1
