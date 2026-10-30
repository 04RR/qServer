#!/usr/bin/env bash
# Smoke gate for Ornith-1.5-9B (ornith-ai), arch `qwen35`, dense, TEXT-ONLY, WHOLE MODEL ON THE 4090.
# Hits the backend DIRECTLY (:8084) so the swap/router layer is not in the path.
#
# What's DIFFERENT from gates35.sh (why this is its own gate):
#  - SINGLE-GPU residency: the fastest-config invariant is that NOTHING spills to the 5060 Ti. #1 asserts
#    GPU0 loaded AND GPU1 ~0 (the OPPOSITE of the 35B's min-spill check). A non-zero GPU1 means the pin/
#    placement regressed and we're paying the cross-GPU tax this model was configured to avoid.
#  - NO MTP: Ornith is dense with no draft head in the GGUF, so there is no draft_n gate (that's 35B-only).
set -uo pipefail
B="${B:-http://127.0.0.1:8084}"
EXPECT_MTP="${EXPECT_MTP:-1}"   # 1 = assert MTP is drafting (default launcher). Set 0 for an MTP=off run.
PY=python3
command -v jq >/dev/null || { echo "need jq"; exit 1; }
pass=0; fail=0
ok(){ echo "  PASS: $1"; pass=$((pass+1)); }
no(){ echo "  FAIL: $1"; fail=$((fail+1)); }

SYS='You are Qwen, created by Alibaba Cloud. You are a helpful assistant.'
chat(){ # $1=user text  $2=extra json
  local u="$1" extra="${2:-{\}}"
  jq -n --arg s "$SYS" --arg u "$u" --argjson e "$extra" \
    '{model:"ornith",messages:[{role:"system",content:$s},{role:"user",content:$u}],stream:false} * $e' \
  | curl -s -m 300 "$B/v1/chat/completions" -H 'Content-Type: application/json' -d @-
}

echo "########## Ornith-1.5-9B (qwen35 dense, 4090-only) gates ##########"

# ---- 1. Load + SINGLE-GPU residency (whole model on the 4090, nothing on the 5060 Ti) ----
echo "[1] load + single-GPU residency (4090 only)"
h=$(curl -s -o /dev/null -w "%{http_code}" -m5 "$B/health")
[ "$h" = "200" ] && ok "Ornith healthy" || { no "health=$h"; echo "server not up; aborting"; exit 1; }
v0=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0|tr -d ' ')
v1=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 1|tr -d ' ')
echo "  GPU0=${v0}MiB  GPU1=${v1}MiB"
# Q6_K (~7.6 GiB) + q8_0 KV lives entirely on GPU0; GPU1 must stay ~idle (allow a little for other tenants,
# but a real spill would be GiB, not MiB). If GPU1 is high the pin/placement regressed.
if [ "$v0" -gt 6000 ] && [ "$v1" -lt 1000 ] 2>/dev/null; then
  ok "single-GPU residency: GPU0=${v0} GPU1=${v1} (expect all on the 4090, GPU1 ~0)"
else
  no "residency off-plan: GPU0=${v0} GPU1=${v1} (expected GPU0>6000, GPU1<1000 — did it spill to the 5060 Ti?)"
fi

# ---- 2. Coherence ----
echo "[2] coherence"
c=$(chat "What is the capital of France? Answer in one word." '{"max_tokens":2048}' | jq -r '.choices[0].message.content // ""')
echo "$c" | grep -qi paris && ok "capital=Paris" || no "got: $c"
echo "$c" | $PY -c 'import sys,re;t=sys.stdin.read();sys.exit(1 if len(re.findall(r"[Ѐ-ӿ؀-ۿ一-鿿ऀ-ॿ]",t))>3 else 0)' \
  && ok "no script-salad" || no "multi-script salad"

# ---- 3. Digit-loop / repetition canary (reuses the stack's self-tested detector) ----
echo "[3] digit-loop canary"
canary(){ chat "Repeat back this random string then continue writing prose about it: qX7#mept Zq9 vbrr 42a! kkap. Do not make a numbered list." \
       "{\"max_tokens\":1200,\"temperature\":$1}" \
  | jq -r '(.choices[0].message.reasoning_content // "") + "\n" + (.choices[0].message.content // "")'; }
CANARY="${CANARY:-$(dirname "$0")/canary.py}"
printf 'x 1. 2. 3. 4. 5. 6. 7. 8. 9. 10. 11. 12. 13. 14. 15. 16. 17. 18. padding padding padding padding padding padding padding padding padding padding padding padding padding padding padding padding padding.' \
  | $PY "$CANARY" >/dev/null 2>&1
stc=$?
[ "$stc" -eq 1 ] && ok "canary detector self-test fires (exit 1)" \
  || no "canary self-test did NOT fire (exit $stc; need 1) — gate meaningless"
for T in 0.9 1.2; do
  r=$(canary $T | $PY "$CANARY"); rc=$?
  [ $rc -eq 0 ] && ok "canary temp=$T: $r" || no "canary temp=$T: $r"
done

# ---- 4. Thinking closes + reasoning_content surfaced ----
echo "[4] thinking closes with </think>"
resp=$(chat "Think step by step: what is 17*23?" '{"max_tokens":2048}')
rc=$(echo "$resp" | jq -r '.choices[0].message.reasoning_content // ""')
ct=$(echo "$resp" | jq -r '.choices[0].message.content // ""')
[ -n "$rc" ] && ok "reasoning_content extracted (${#rc} chars)" || no "no reasoning_content (template may not parse <think>)"
echo "$ct" | grep -q "391" && ok "answer 391 correct" || no "answer: $ct"
echo "$ct" | grep -q "<think>" && no "raw <think> leaked into content" || ok "no <think> leak in content"

# ---- 5. Tool calls @ temp 0.6 and 0.2 (clean, not tool-in-think) ----
echo "[5] tool calls @ temp 0.6 and 0.2"
TOOLS='[{"type":"function","function":{"name":"run","description":"run a shell command","parameters":{"type":"object","properties":{"cmd":{"type":"string"}},"required":["cmd"]}}}]'
for T in 0.6 0.2; do
  r=$(jq -n --arg s "$SYS" --argjson t "$TOOLS" --arg tm "$T" \
      '{model:"ornith",messages:[{role:"system",content:$s},{role:"user",content:"List /var/log using the run tool."}],tools:$t,max_tokens:2048,temperature:($tm|tonumber),stream:false}' \
    | curl -s -m 300 "$B/v1/chat/completions" -H 'Content-Type: application/json' -d @-)
  args=$(echo "$r" | jq -r '.choices[0].message.tool_calls[0].function.arguments // ""')
  rc=$(echo "$r" | jq -r '.choices[0].message.reasoning_content // ""')
  err=$(echo "$r" | jq -r '.error.message // ""')
  if [ -n "$err" ]; then no "temp=$T errored: $err"
  elif [ -z "$args" ] || [ "$args" = "{}" ]; then no "temp=$T empty tool args"
  elif echo "$args" | jq -e . >/dev/null 2>&1; then
    if echo "$rc" | grep -qE '"name"\s*:\s*"run"|<tool_call>'; then no "temp=$T TOOL-IN-THINK detected"
    else ok "temp=$T clean tool call: $args"; fi
  else no "temp=$T invalid args: $args"; fi
done

# ---- 6. Sustained decode: clean + report decode t/s ----
echo "[6] sustained decode (6 gens clean) + decode t/s"
errs=0
for i in 1 2 3 4 5 6; do
  r=$(chat "Write a detailed 300-word explanation of how a B-tree insert rebalances." '{"max_tokens":700,"temperature":0.7}')
  e=$(echo "$r" | jq -r '.error.message // ""')
  [ -n "$e" ] && { errs=$((errs+1)); echo "    err: $e"; }
done
tps=$(curl -s -m 300 "$B/completion" -H 'Content-Type: application/json' \
      -d '{"prompt":"Write a Python function that merges two sorted lists.\n\ndef merge(","n_predict":220,"temperature":0.2,"cache_prompt":false}' \
    | jq -r '.timings.predicted_per_second // 0')
h2=$(curl -s -o /dev/null -w "%{http_code}" -m5 "$B/health")
printf -v tpsr '%.1f' "$tps" 2>/dev/null || tpsr="$tps"
[ "$errs" -eq 0 ] && [ "$h2" = "200" ] && ok "6/6 gens clean, server alive, decode ~${tpsr} t/s" \
  || no "$errs/6 errored (health=$h2)"

# ---- 7. MTP live (embedded nextn head drafting): draft_n>0, acceptance>0, AL>1 ----
# Ornith's GGUF ships a nextn/MTP head; the launcher runs --spec-type draft-mtp (NMAX=3). On a 4090-only
# 9B (bandwidth-bound) MTP ~1.8x code / ~1.37x prose — so a draft_n of 0 means MTP silently stopped
# drafting (regression), which this gate must catch. (Set EXPECT_MTP=0 to allow an MTP=off run.)
echo "[7] MTP live (draft_n / acceptance)"
t=$(curl -s -m 300 "$B/completion" -H 'Content-Type: application/json' \
     -d '{"prompt":"Write a Python function that merges two sorted lists.\n\ndef merge(","n_predict":220,"temperature":0.2,"cache_prompt":false}' \
   | jq -r '.timings | "\(.draft_n // 0) \(.draft_n_accepted // 0) \(.predicted_n // 0) \(.predicted_per_second // 0)"')
echo "  draft_n draft_n_accepted predicted_n tg_tps = $t"
if [ "$EXPECT_MTP" = "1" ]; then
  echo "$t" | $PY -c '
import sys
dn,da,pn,tps=sys.stdin.read().split()
dn,da,pn,tps=int(dn),int(da),int(pn),float(tps)
if dn==0: print("MTP NOT DRAFTING (draft_n=0)"); sys.exit(1)
acc=da/dn; al=pn/(pn-da) if pn>da else 99
print(f"acceptance={acc:.3f} AL={al:.2f} tg={tps:.1f} t/s")
sys.exit(0 if (acc>0 and al>1.0) else 1)' && ok "MTP contributing" || no "MTP not contributing (expected draft-mtp live)"
else
  sk_dn=$(echo "$t" | awk '{print $1}')
  [ "$sk_dn" = "0" ] && ok "MTP off as expected (draft_n=0)" || no "EXPECT_MTP=0 but draft_n=$sk_dn (MTP still on?)"
fi

echo; echo "########## Ornith gates: $pass PASS / $fail FAIL ##########"
[ "$fail" -eq 0 ] || exit 1
