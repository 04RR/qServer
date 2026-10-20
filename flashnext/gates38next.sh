#!/usr/bin/env bash
# M3 correctness gates for qwen-38-flash-next (qwen4exp 125B-A6B MoE + PLE + MTP), :8085.
# Modelled on gates122.sh (NOT gates35/gates27q6 — those carry a broken canary self-test).
# Hits the backend DIRECTLY. Assumes a RUNNING MTP-on server (run-38-flash-next.sh, NCM=31 TS=40,8).
#
# This script does the SINGLE-SERVER gates: 1,2,3,4,5,7,8,9,10,12.
# Gates that need an ALTERNATE server config are run by m3-driver.sh, not here:
#   #6  MTP output identity (draft on vs off)         -> needs a SPEC_OFF relaunch
#   #11 f16-vs-q8_0 KV needle differential            -> needs a KVK=q8_0 relaunch (+LLAMA_ATTN_ROT_DISABLE=1)
#   perplexity on a held-out slice (llama-perplexity)  -> separate binary run
set -uo pipefail
B="${B:-http://127.0.0.1:8085}"
MODEL_ID="${MODEL_ID:-flash-next}"
PY=python3
HERE="$(cd "$(dirname "$0")" && pwd)"
command -v jq >/dev/null || { echo "need jq (~/.local/bin)"; exit 1; }
pass=0; fail=0
ok(){ echo "  PASS: $1"; pass=$((pass+1)); }
no(){ echo "  FAIL: $1"; fail=$((fail+1)); }
# round2: EXPECT_MTP is DECLARED by the caller (from the entry's SPEC_OFF: SPEC_OFF=1 => MTP off =>
# EXPECT_MTP=0; otherwise EXPECT_MTP=1). We do NOT infer intent from the server. Instead we read the
# server's ACTUAL speculative state from /slots (/props omits it on this build) and assert it equals
# EXPECT_MTP. Only when EXPECT_MTP=0 AND the server confirms speculative=off do we SKIP #5/#9's draft
# parts. If /slots is unreachable or jq can't parse it -> FAIL (we never skip on an unverified state);
# if the actual state != EXPECT_MTP -> FAIL.
EXPECT_MTP="${EXPECT_MTP:-$([ "${SPEC_OFF:-0}" = "1" ] && echo 0 || echo 1)}"
# round2: EXPECT_VISION is DECLARED the same way (from the entry's VISION env: VISION=off => no mmproj =>
# EXPECT_VISION=0). The batch entry (qwen-38-flash-next-batch) runs text-only, so #8/#9 are N/A there and
# SKIP on declaration. Unlike EXPECT_MTP (where a silent MTP-on misconfig is dangerous, so we assert the
# server's actual /slots state), vision-off is benign — the worst case of a wrong declaration is skipping a
# check that would have passed — so we trust the declaration rather than probe for mmproj absence.
EXPECT_VISION="${EXPECT_VISION:-$([ "${VISION:-on}" = "off" ] && echo 0 || echo 1)}"
# Residency bands (#2) are PROFILE-specific. The two entries resident-size differently:
#   main  (NCM=31, f16 KV, np=1, vision on): GPU0 23220 / GPU1 14196 MiB  [measured as-shipped 2026-10-20]
#   batch (NCM=28, q8_0 KV, np=4, text-only): GPU0 18906 / GPU1 15568 MiB [measured as-shipped 2026-10-20]
# PROFILE=batch (set by regress.sh for the batch entry) selects the batch bands; default = main.
PROFILE="${PROFILE:-$([ "${NP:-1}" -gt 1 ] 2>/dev/null && echo batch || echo main)}"
if [ "$PROFILE" = "batch" ]; then
  G0_LO=16500; G0_HI=20500; G0_REF=18906; G1_LO=14200; G1_HI=16200; G1_REF=15568
else
  G0_LO=21500; G0_HI=24200; G0_REF=23220; G1_LO=13000; G1_HI=15500; G1_REF=14196
fi
skip=0; sk(){ echo "  SKIP: $1"; skip=$((skip+1)); }

# classify_slots <json> -> prints on|off|error (error = empty/unparseable/not-a-slot-array).
classify_slots(){
  local j="$1"
  [ -n "$j" ] || { echo error; return; }
  echo "$j" | jq -e 'type=="array" and length>0' >/dev/null 2>&1 || { echo error; return; }
  if echo "$j" | jq -e '(.[0].speculative==true) or ((.[0].params["speculative.types"] // "none")!="none")' >/dev/null 2>&1
    then echo on; else echo off; fi
}
# decide <expect 0|1> <slots-json> -> prints "RUN|SKIP|FAIL <reason>"; return 0 on RUN/SKIP, 1 on FAIL.
decide(){
  local expect="$1" j="$2" act actb
  act=$(classify_slots "$j")
  [ "$act" = "error" ] && { echo "FAIL /slots unreachable or unparseable — speculative state unverifiable"; return 1; }
  actb=$([ "$act" = "on" ] && echo 1 || echo 0)
  [ "$actb" != "$expect" ] && { echo "FAIL state mismatch — expected EXPECT_MTP=$expect, server speculative=$act"; return 1; }
  [ "$expect" = "0" ] && { echo "SKIP MTP off confirmed (speculative=$act) — draft gates N/A"; return 0; }
  echo "RUN MTP on confirmed (speculative=$act)"; return 0
}

# ---- SELFTEST (no server): expected on/off x actual on/off + /slots-down + malformed ----
if [ "${SELFTEST:-0}" = "1" ]; then
  echo "########## gates38next MTP-skip decision self-test ##########"
  ON='[{"id":0,"speculative":true,"params":{"speculative.types":"none"}}]'
  ON2='[{"id":0,"speculative":false,"params":{"speculative.types":"draft_model"}}]'   # on via types
  OFF='[{"id":0,"speculative":false,"params":{"speculative.types":"none"}}]'
  DOWN=''                                                                              # curl failed
  BAD='{"error":"server loading model"}'                                              # not a slot array
  st_pass=0; st_fail=0
  case_test(){ # <name> <expect> <json> <want-verb RUN|SKIP|FAIL>
    local name="$1" expect="$2" json="$3" want="$4" got rc verb
    got=$(decide "$expect" "$json"); rc=$?; verb=${got%% *}
    if [ "$verb" = "$want" ]; then echo "   st-PASS: $name -> $verb (rc=$rc) :: $got"; st_pass=$((st_pass+1))
    else echo "  st-FAIL: $name -> got '$verb' want '$want' (rc=$rc) :: $got"; st_fail=$((st_fail+1)); fi
    # cross-check return code agrees with verb
    if { [ "$want" = "FAIL" ] && [ "$rc" != "1" ]; } || { [ "$want" != "FAIL" ] && [ "$rc" != "0" ]; }; then
      echo "  st-FAIL: $name return-code $rc disagrees with verb $want"; st_fail=$((st_fail+1)); fi
  }
  case_test "expect-on  actual-on"        1 "$ON"  RUN
  case_test "expect-on  actual-on(types)" 1 "$ON2" RUN
  case_test "expect-on  actual-off"       1 "$OFF" FAIL
  case_test "expect-off actual-on"        0 "$ON"  FAIL
  case_test "expect-off actual-off"       0 "$OFF" SKIP
  case_test "expect-off slots-down"       0 "$DOWN" FAIL
  case_test "expect-on  slots-down"       1 "$DOWN" FAIL
  case_test "expect-off malformed"        0 "$BAD" FAIL
  echo "########## self-test: $st_pass PASS / $st_fail FAIL ##########"
  [ "$st_fail" -eq 0 ] && { echo "ALL SELFTEST CASES PASS"; exit 0; } || { echo "SELFTEST FAILURES"; exit 1; }
fi
SYS='You are Qwen, created by Alibaba Cloud. You are a helpful assistant.'
chat(){ local u="$1" extra="${2:-{\}}"
  jq -n --arg m "$MODEL_ID" --arg s "$SYS" --arg u "$u" --argjson e "$extra" \
    '{model:$m,messages:[{role:"system",content:$s},{role:"user",content:$u}],stream:false} * $e' \
  | curl -s -m 900 "$B/v1/chat/completions" -H 'Content-Type: application/json' -d @-; }
both(){ jq -r '(.choices[0].message.reasoning_content // "") + "\n" + (.choices[0].message.content // "")'; }

echo "########## qwen-38-flash-next (qwen4exp MoE+PLE+MTP) M3 gates ##########"

# ---- 1. Loads + responds ----
echo "[1] load + coherent response"
h=$(curl -s -o /dev/null -w "%{http_code}" -m5 "$B/health")
[ "$h" = "200" ] && ok "healthy" || { no "health=$h"; echo "server not up; aborting"; exit 1; }
c=$(chat "What is the capital of France? Answer in one word." '{"max_tokens":2048}' | jq -r '.choices[0].message.content // ""')
echo "$c" | grep -qi paris && ok "capital=Paris" || no "got: $c"
echo "$c" | $PY -c 'import sys,re;t=sys.stdin.read();sys.exit(1 if len(re.findall(r"[Ѐ-ӿ؀-ۿ一-鿿ऀ-ॿ]",t))>3 else 0)' \
  && ok "no script-salad" || no "multi-script salad"

# ---- 2. Residency within measured bands ----
# Brief wants PER-PID (nvidia-smi --query-compute-apps). WSL's nvidia-smi reports per-process memory as
# [N/A] (a WSL/driver limitation), so per-PID is unavailable here. Both GPUs are DEDICATED to this one
# llama-server (no display, no other CUDA process), so system-wide memory.used == this PID's usage. We
# assert that exactly one compute process is present (so system-wide is unambiguous), then band-check it.
echo "[2] residency (WSL: per-PID N/A -> system-wide, single dedicated process; PROFILE=$PROFILE)"
caps=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | tr '\n' ' ')
echo "$caps" | grep -q "8.9" && echo "$caps" | grep -q "12.0" && ok "GPUs cc: $caps" || no "GPU caps: $caps"
nproc_gpu=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | sort -u | grep -c . )  # WSL lists a PID once per GPU; dedupe
[ "$nproc_gpu" = "1" ] && ok "exactly 1 GPU compute process (system-wide == per-PID)" || no "$nproc_gpu GPU processes — system-wide residency ambiguous"
g0=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0 | tr -d ' ')
g1=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 1 | tr -d ' ')
echo "  VRAM used: GPU0=${g0}MiB GPU1=${g1}MiB (PROFILE=$PROFILE ref: $G0_REF / $G1_REF)"
if [ "$g0" -ge "$G0_LO" ] && [ "$g0" -le "$G0_HI" ] 2>/dev/null; then ok "GPU0 ${g0} (band $G0_LO-$G0_HI, ref $G0_REF)"; else no "GPU0 off-plan ${g0} (band $G0_LO-$G0_HI)"; fi
if [ "$g1" -ge "$G1_LO" ] && [ "$g1" -le "$G1_HI" ] 2>/dev/null; then ok "GPU1 ${g1} (band $G1_LO-$G1_HI, ref $G1_REF)"; else no "GPU1 off-plan ${g1} (band $G1_LO-$G1_HI)"; fi

# ---- 3. Canary detector self-test — require exit EXACTLY 1 (padded >200 chars) ----
echo "[3] canary self-test (exit==1 on synthetic runaway)"
printf 'x 1. 2. 3. 4. 5. 6. 7. 8. 9. 10. 11. 12. 13. 14. 15. 16. 17. 18. padding padding padding padding padding padding padding padding padding padding padding padding padding padding padding padding padding.' \
  | $PY "$HERE/../canary.py" >/dev/null 2>&1; stc=$?
[ "$stc" -eq 1 ] && ok "canary fires on synthetic runaway (exit 1)" \
  || no "canary self-test exit=$stc (need 1; 2=vacuous is a FAIL) — gate meaningless"

# ---- 4. Digit-loop canary on REAL output — NON-EMPTY assertion first ----
echo "[4] digit-loop canary on real output (non-empty first)"
for T in 1.0 1.2; do
  out=$(chat "Repeat back this random string then continue writing prose about it: qX7#mept Zq9 vbrr 42a! kkap. Do not make a numbered list." \
        "{\"max_tokens\":1200,\"temperature\":$T}" | both)
  if [ "$(printf '%s' "$out" | tr -d '[:space:]' | wc -c)" -lt 40 ]; then no "temp=$T EMPTY output — vacuous, not a pass"; continue; fi
  echo "$out" | $PY "$HERE/../canary.py" >/dev/null 2>&1; rc=$?
  case $rc in 0) ok "canary temp=$T clean";; 2) no "temp=$T VACUOUS";; *) no "temp=$T RUNAWAY";; esac
done

# ---- M. speculative state matches the declared EXPECT_MTP (sets MTP_SKIP for #5/#9) ----
echo "[M] speculative state vs declared EXPECT_MTP=$EXPECT_MTP (SPEC_OFF=${SPEC_OFF:-0})"
MTP_SKIP=0
_mdec=$(decide "$EXPECT_MTP" "$(curl -s -m5 "$B/slots" 2>/dev/null)"); _mrc=$?
_mverb=${_mdec%% *}; _mreason=${_mdec#* }
case "$_mverb" in
  RUN)  ok "$_mreason" ;;
  SKIP) ok "$_mreason"; MTP_SKIP=1 ;;
  FAIL) no "$_mreason" ;;
esac

# ---- 5. MTP liveness: draft acceptance present and > 0.4 ----
echo "[5] MTP live (draft_n>0, acceptance>0.4)"
if [ "$MTP_SKIP" = "1" ]; then sk "MTP off by config (EXPECT_MTP=0, server confirmed) — draft liveness N/A"; else
t=$(curl -s -m 900 "$B/completion" -H 'Content-Type: application/json' \
     -d '{"prompt":"Write a Python function that merges two sorted lists.\n\ndef merge(","n_predict":200,"temperature":0.2,"cache_prompt":false}' \
   | jq -r '.timings | "\(.draft_n) \(.draft_n_accepted) \(.predicted_n) \(.predicted_per_second)"')
echo "  draft_n draft_n_accepted predicted_n tg = $t"
echo "$t" | $PY -c '
import sys; dn,da,pn,tps=sys.stdin.read().split(); dn,da,pn,tps=int(dn),int(da),int(pn),float(tps)
if dn==0: print("MTP NOT DRAFTING (draft_n=0)"); sys.exit(1)
acc=da/dn; print(f"acceptance={acc:.3f} tg={tps:.1f} t/s")
sys.exit(0 if acc>0.4 else 1)' && ok "MTP live, acc>0.4" || no "MTP dead or acc<=0.4"
fi

# ---- 7. Tool call clean at temp 1.0, JSON args, no in-think call ----
echo "[7] tool call @ temp 1.0 (JSON args, no tool-in-think)"
TOOLS='[{"type":"function","function":{"name":"run","description":"run a shell command","parameters":{"type":"object","properties":{"cmd":{"type":"string"}},"required":["cmd"]}}}]'
r=$(jq -n --arg m "$MODEL_ID" --arg s "$SYS" --argjson t "$TOOLS" \
    '{model:$m,messages:[{role:"system",content:$s},{role:"user",content:"List /var/log using the run tool."}],tools:$t,max_tokens:2048,temperature:1.0,stream:false}' \
  | curl -s -m 900 "$B/v1/chat/completions" -H 'Content-Type: application/json' -d @-)
args=$(echo "$r" | jq -r '.choices[0].message.tool_calls[0].function.arguments // ""')
rcx=$(echo "$r" | jq -r '.choices[0].message.reasoning_content // ""')
err=$(echo "$r" | jq -r '.error.message // ""')
if [ -n "$err" ]; then no "temp=1.0 errored: $err"
elif [ -z "$args" ] || [ "$args" = "{}" ]; then no "temp=1.0 empty tool args"
elif echo "$args" | jq -e . >/dev/null 2>&1; then
  echo "$rcx" | grep -qE '"name"\s*:\s*"run"|<tool_call>' && no "temp=1.0 TOOL-IN-THINK" || ok "temp=1.0 clean tool call: $args"
else no "temp=1.0 invalid args: $args"; fi

# ---- 8. VISION — ground-truthed red PNG (mmproj pipeline end-to-end) ----
echo "[8] vision: name a ground-truthed red image"
if [ "$EXPECT_VISION" = "0" ]; then
  sk "vision off by config (EXPECT_VISION=0, VISION=${VISION:-on}) — mmproj not loaded on this entry"
  vall=""
else
IMG=/tmp/flashnext_gate_red.png
$PY - "$IMG" <<'EOF'
import sys, zlib, struct
def chunk(t,d): c=t+d; return struct.pack(">I",len(d))+c+struct.pack(">I",zlib.crc32(c)&0xffffffff)
W=H=224; raw=bytearray()
for y in range(H): raw.append(0); raw += bytes((220,30,30))*W
png=b"\x89PNG\r\n\x1a\n"+chunk(b"IHDR",struct.pack(">IIBBBBB",W,H,8,2,0,0,0))
png+=chunk(b"IDAT",zlib.compress(bytes(raw),9))+chunk(b"IEND",b"")
open(sys.argv[1],"wb").write(png)
EOF
B64=$(base64 -w0 "$IMG" 2>/dev/null || base64 "$IMG" | tr -d '\n')
vr=$(jq -n --arg m "$MODEL_ID" --arg s "$SYS" --arg u "What is the single dominant color of this image? Answer with just the color word." --arg d "data:image/png;base64,$B64" \
    '{model:$m,max_tokens:512,temperature:0.2,stream:false,messages:[{role:"system",content:$s},{role:"user",content:[{type:"text",text:$u},{type:"image_url",image_url:{url:$d}}]}]}' \
  | curl -s -m 300 "$B/v1/chat/completions" -H 'Content-Type: application/json' -d @-)
verr=$(echo "$vr" | jq -r '.error.message // ""')
vall=$(echo "$vr" | jq -r '(.choices[0].message.reasoning_content // "") + " " + (.choices[0].message.content // "")')
if [ -n "$verr" ]; then no "vision errored: $verr"
elif echo "$vall" | grep -qiE '\bred\b|crimson|scarlet'; then ok "vision reads red"
elif echo "$vall" | grep -qiE "can'?t see|no image|unable to (see|view)"; then no "vision NOT wired: $vall"
else no "vision wrong colour: $(echo "$vall"|tr '\n' ' '|cut -c1-80)"; fi
fi

# ---- 9. Coexistence: mmproj + PLE-offload + MTP ALL AT ONCE on one server ----
echo "[9] coexistence (vision + PLE-on-CPU + MTP live, same server)"
# PLE-on-CPU is inherent (-ot per_layer_token_embd=CPU at launch); prove vision AND MTP both live here.
v9=$(echo "$vall" | grep -qiE '\bred\b|crimson|scarlet' && echo yes || echo no)
if [ "$EXPECT_VISION" = "0" ]; then
  # Text-only entry: no mmproj, so the vision half of coexistence is N/A. PLE-on-CPU is inherent at launch;
  # MTP coexistence is covered by #M/#5. Nothing left to assert here.
  sk "coexistence N/A (vision off by config) — PLE offload inherent, MTP $([ "$MTP_SKIP" = "1" ] && echo off || echo on)"
elif [ "$MTP_SKIP" = "1" ]; then
  # MTP off: the draft-coexistence assertion is N/A -> explicit SKIP. Vision is already proven at #8,
  # PLE-on-CPU is inherent; but if vision broke at #8 this is still a real FAIL, not a skip.
  [ "$v9" = "yes" ] && sk "coexistence draft_n N/A (MTP off) — vision(red) verified at #8, PLE offload inherent" \
    || no "coexistence broken: vision=$v9 (MTP off)"
else
  d9=$(curl -s -m 900 "$B/completion" -H 'Content-Type: application/json' \
       -d '{"prompt":"def quicksort(a):","n_predict":120,"temperature":0.2,"cache_prompt":false}' | jq -r '.timings.draft_n // 0')
  [ "$v9" = "yes" ] && [ "$d9" -gt 0 ] 2>/dev/null && ok "vision(red) + MTP(draft_n=$d9) coexist with PLE on CPU" \
    || no "coexistence broken: vision=$v9 draft_n=$d9"
fi

# ---- 10. >=100K needle recall, FRESH sentinel ----
if [ "${SKIP_DEEP:-0}" = "1" ]; then
  echo "[10] 100K needle — SKIPPED (SKIP_DEEP=1; deep prefill, run separately)"
else
  echo "[10] 100K needle (fresh sentinel)"
  MODEL="$MODEL_ID" $PY "$HERE/needle-next.py" "$B" 100000 && ok "100K needle recalled" || no "100K needle NOT recalled"
fi

# ---- 12. reasoning_effort honored: (a) template render differs; (b) mean reasoning length high>low ----
# Two independent assertions (no direction-agnostic loophole):
#  12a DETERMINISTIC PLUMBING: /apply-template must render DIFFERENT prompts for reasoning_effort low vs
#      high. This proves the kwarg reaches the chat template (no generation, no sampling noise).
#  12b BEHAVIOURAL DIRECTION: mean reasoning_content length over 3 prompts at temp 0 must be HIGHER for
#      high than for low. This is the real product property. It is NOT loosened to "either direction":
#      if an entry (e.g. q8_0-KV batch) renders high SHORTER, that is a genuine FAIL to report, not hide.
echo "[12] reasoning_effort (a) template render differs  (b) mean reasoning length high>low (3 prompts, temp 0)"
_tmpl(){ curl -s -m30 "$B/apply-template" -H 'Content-Type: application/json' \
  -d "$(jq -n --arg s "$SYS" --arg e "$1" '{messages:[{role:"system",content:$s},{role:"user",content:"Hi"}],chat_template_kwargs:{reasoning_effort:$e}}')" \
  | jq -r '.prompt // ""'; }
tl=$(_tmpl low); th=$(_tmpl high)
if [ -n "$tl" ] && [ "$tl" != "$th" ]; then ok "12a /apply-template render differs low vs high (kwarg reaches template)"
else no "12a /apply-template render IDENTICAL (len low=${#tl} high=${#th}) — reasoning_effort not reaching the template"; fi
QS=("Prove that the square root of 2 is irrational, step by step." \
    "Explain, with the physics, why the daytime sky is blue." \
    "Design an algorithm for the k-th largest element in an unsorted array and analyse its complexity.")
sl=0; sh=0; n=0
for q in "${QS[@]}"; do
  a=$(chat "$q" '{"max_tokens":4096,"temperature":0,"chat_template_kwargs":{"reasoning_effort":"low"}}'  | jq -r '.choices[0].message.reasoning_content // "" | length')
  b=$(chat "$q" '{"max_tokens":4096,"temperature":0,"chat_template_kwargs":{"reasoning_effort":"high"}}' | jq -r '.choices[0].message.reasoning_content // "" | length')
  n=$((n+1)); sl=$((sl+a)); sh=$((sh+b)); echo "    prompt$n: low=$a high=$b"
done
ml=$((sl/n)); mh=$((sh/n))
echo "  mean reasoning chars over $n prompts: low=$ml high=$mh"
if [ "$mh" -gt "$ml" ] 2>/dev/null; then ok "12b mean reasoning high ($mh) > low ($ml)"
else no "12b mean reasoning high ($mh) NOT > low ($ml) — reasoning_effort not honored directionally on this entry"; fi

echo; echo "########## flash-next single-server gates: $pass PASS / $fail FAIL / $skip SKIP ##########"
echo "(gates 6 MTP-identity, 11 f16-vs-q8_0, perplexity: run via m3-driver.sh)"
[ "$fail" -eq 0 ] || exit 1
