#!/usr/bin/env bash
# Gate suite for the 27B QUALITY entry — Qwen3.6-27B UD-Q6_K_XL, layer-split across BOTH GPUs.
# Hits the backend DIRECTLY (:8081) so the swap/router layer is not in the path (same convention as
# gates35.sh :8082 / gates122.sh :8084). Works whether the server is launched standalone OR loaded
# by llama-swap — either way the backend listens on :8081.
#
# WHY A SEPARATE GATE (not just reusing regress.sh): this entry exercises a path neither 27B nor the
# MoE models did — the DENSE MTP draft/verify loop re-decoding across a --tensor-split layer boundary
# (residual stream crossing PCIe mid-model). A broken cross-device verify shows up as a digit-loop.
# Modeled on regress.sh's 27B coverage (digit-loop, tool-in-think, prefix-cache), repointed here.
set -uo pipefail
B="${B:-http://127.0.0.1:8081}"
PY=python3
command -v jq >/dev/null || { echo "need jq (~/.local/bin)"; exit 1; }
pass=0; fail=0
ok(){ echo "  PASS: $1"; pass=$((pass+1)); }
no(){ echo "  FAIL: $1"; fail=$((fail+1)); }

SYS='You are Qwen, created by Alibaba Cloud. You are a helpful assistant.'
chat(){ # $1=user text  $2=extra json
  local u="$1" extra="${2:-{\}}"
  jq -n --arg s "$SYS" --arg u "$u" --argjson e "$extra" \
    '{model:"qwen36-q6",messages:[{role:"system",content:$s},{role:"user",content:$u}],stream:false} * $e' \
  | curl -s -m 300 "$B/v1/chat/completions" -H 'Content-Type: application/json' -d @-
}

echo "########## 27B Q6_K_XL (dense, dual-GPU layer split) gates ##########"

# ---- 1. Load + DUAL-GPU residency ----
# The real safety property is >=1 GiB FREE ON EACH CARD (verified live: 2.2 GiB GPU0 / 5.9 GiB GPU1
# at TS=3,1). The mission's "GPU1 in 6-9 GiB" was an estimate assuming ~7.5 GiB on GPU1; measured
# reality is ~10.4 GiB because GPU0 (24.5 GiB) is the BINDING constraint -- 3,1 already leaves GPU0
# at ~2.2 GiB free, so GPU1 necessarily carries ~10 GiB. 3,1 is also SPEED-OPTIMAL: it minimises the
# slow 5060 Ti's layer share (same min-spill logic as the 35B). So the band here is the measured
# envelope, not the estimate. The failure this catches: split silently didn't apply (one card ~0),
# or a card starved below 1 GiB free.
echo "[1] load + dual-GPU residency (resident on both, >=1 GiB free each)"
h=$(curl -s -o /dev/null -w "%{http_code}" -m5 "$B/health")
[ "$h" = "200" ] && ok "Q6 healthy" || { no "health=$h"; echo "server not up; aborting"; exit 1; }
caps=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | tr '\n' ' ')
echo "$caps" | grep -q "8.9" && echo "$caps" | grep -q "12.0" && ok "GPUs cc: $caps" || no "GPU caps: $caps"
T0=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits -i 0|tr -d ' ')
T1=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits -i 1|tr -d ' ')
v0=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0|tr -d ' ')
v1=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 1|tr -d ' ')
free0=$((T0-v0)); free1=$((T1-v1))
echo "  GPU0=${v0}/${T0}MiB (free ${free0})  GPU1=${v1}/${T1}MiB (free ${free1})"
# dense weights must be resident on BOTH cards (neither ~0 => split applied), >=1 GiB free on EACH.
if [ "$v0" -gt 18000 ] && [ "$v1" -ge 6000 ] && [ "$v1" -le 12000 ] \
   && [ "$free0" -ge 1024 ] && [ "$free1" -ge 1024 ] 2>/dev/null; then
  ok "dual-GPU residency ok: GPU0=${v0} GPU1=${v1}, free ${free0}/${free1} (>=1024 each)"
else
  no "residency off-plan: GPU0=${v0} GPU1=${v1} free ${free0}/${free1} — if GPU0 free<1G shift TS toward GPU1 (e.g. 2,1); if a card is ~0 the split didn't apply"
fi

# ---- 2. Coherence (no multi-script salad) ----
echo "[2] coherence"
c=$(chat "What is the capital of France? Answer in one word." '{"max_tokens":2048}' | jq -r '.choices[0].message.content // ""')
echo "$c" | grep -qi paris && ok "capital=Paris" || no "got: $c"
echo "$c" | $PY -c 'import sys,re;t=sys.stdin.read();sys.exit(1 if len(re.findall(r"[Ѐ-ӿ؀-ۿ一-鿿ऀ-ॿ]",t))>3 else 0)' \
  && ok "no script-salad" || no "multi-script salad"

# ---- 3. DIGIT-LOOP CANARY — HARD GATE (dense MTP verify across the layer-split boundary) ----
echo "[3] digit-loop canary (HARD GATE — MTP verify across --tensor-split)"
canary(){ # $1=temp
  chat "Repeat back this random string then continue writing prose about it: qX7#mept Zq9 vbrr 42a! kkap. Do not make a numbered list." \
       "{\"max_tokens\":1200,\"temperature\":$1}" \
  | jq -r '(.choices[0].message.reasoning_content // "") + "\n" + (.choices[0].message.content // "")'
}
CANARY="${CANARY:-$(dirname "$0")/canary.py}"
# self-test the detector before trusting any PASS from it (a detector that cannot fire is worse than none)
printf 'x 1. 2. 3. 4. 5. 6. 7. 8. 9. 10. 11. 12. 13. 14. 15. 16. 17.' | $PY "$CANARY" >/dev/null 2>&1 \
  && { no "canary detector self-test FAILED (did not flag a synthetic runaway) — gate is meaningless"; } \
  || ok "canary detector self-test: fires on synthetic runaway"
for T in 0.9 1.2; do
  r=$(canary $T | $PY "$CANARY"); rc=$?
  [ $rc -eq 0 ] && ok "canary temp=$T: $r" || no "canary temp=$T: $r"
done

# ---- 4. Thinking closes (no raw <think> leak into content) ----
echo "[4] thinking closes with </think>"
resp=$(chat "Think step by step: what is 17*23?" '{"max_tokens":2048}')
rc=$(echo "$resp" | jq -r '.choices[0].message.reasoning_content // ""')
ct=$(echo "$resp" | jq -r '.choices[0].message.content // ""')
[ -n "$rc" ] && ok "reasoning_content extracted (${#rc} chars)" || no "no reasoning_content"
echo "$ct" | grep -q "391" && ok "answer 391 correct" || no "answer: $ct"
echo "$ct" | grep -q "<think>" && no "raw <think> leaked into content" || ok "no <think> leak in content"

# ---- 5. Tool calls at temp 0.6 AND 0.2 (tool-in-think re-test) ----
echo "[5] tool calls @ temp 0.6 and 0.2 (no tool-in-think)"
TOOLS='[{"type":"function","function":{"name":"run","description":"run a shell command","parameters":{"type":"object","properties":{"cmd":{"type":"string"}},"required":["cmd"]}}}]'
for T in 0.6 0.2; do
  r=$(jq -n --arg s "$SYS" --argjson t "$TOOLS" --arg tm "$T" \
      '{model:"qwen36-q6",messages:[{role:"system",content:$s},{role:"user",content:"List /var/log using the run tool."}],tools:$t,max_tokens:2048,temperature:($tm|tonumber),stream:false}' \
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

# ---- 6. MTP live across the split: draft_n>0 and AL>1 (the flagged risk) ----
# The real question for this entry: does draft-mtp still accelerate when the model is layer-split
# across two devices? If the MTP head placement or cross-device verify silently disabled drafting,
# draft_n would be 0 here even though decode still works. This is the direct check.
echo "[6] MTP live across --tensor-split (draft_n / draft_n_accepted)"
t=$(curl -s -m 300 "$B/completion" -H 'Content-Type: application/json' \
     -d '{"prompt":"Write a Python function that merges two sorted lists.\n\ndef merge(","n_predict":220,"temperature":0.2,"cache_prompt":false}' \
   | jq -r '.timings | "\(.draft_n) \(.draft_n_accepted) \(.predicted_n) \(.predicted_per_second)"')
echo "  draft_n draft_n_accepted predicted_n tg_tps = $t"
echo "$t" | $PY -c '
import sys
dn,da,pn,tps=sys.stdin.read().split()
dn,da,pn,tps=int(dn),int(da),int(pn),float(tps)
if dn==0: print("MTP NOT DRAFTING across split (draft_n=0)"); sys.exit(1)
acc=da/dn; al=pn/(pn-da) if pn>da else 99
print(f"acceptance={acc:.3f} AL={al:.2f} tg={tps:.1f} t/s")
sys.exit(0 if (acc>0 and al>1.0) else 1)' && ok "MTP contributing across split" || no "MTP not contributing across split"

# ---- 7. Prefix-cache reuse (regress.sh [8], repointed) ----
echo "[7] prefix cache reuse"
PFX=$($PY -c 'print(("The quick brown fox jumps over the lazy dog. "*400))')
body=$(jq -n --arg p "$PFX Summarize the above in 5 words." '{prompt:$p,n_predict:8,cache_prompt:true,temperature:0.0,stream:false}')
t1=$(curl -s -m 120 "$B/completion" -H 'Content-Type: application/json' -d "$body" | jq -r '.timings.prompt_ms // 0')
t2=$(curl -s -m 120 "$B/completion" -H 'Content-Type: application/json' -d "$body" | jq -r '.timings.prompt_ms // 0')
echo "  prompt_ms first=$t1 second=$t2"
$PY -c "import sys; sys.exit(0 if float('$t2')<=max(0.35*float('$t1'),500) else 1)" \
  && ok "reprefill dropped ($t1 -> $t2 ms)" || no "no cache reuse ($t1 -> $t2 ms)"

# ---- 8. >=100K needle recall (the >=100K ctx requirement carried) ----
echo "[8] 100K needle recall"
NEEDLE="$(dirname "$0")/needle35.py"   # self-calibrates via /tokenize -> adapts to the 27B tokenizer
if [ -f "$NEEDLE" ]; then
  $PY "$NEEDLE" "$B" 100000 && ok "100K needle recalled" || no "100K needle NOT recalled"
else
  echo "  (needle35.py absent — skipping)"; fi

echo; echo "########## 27B Q6 gates: $pass PASS / $fail FAIL ##########"
[ "$fail" -eq 0 ] || exit 1
