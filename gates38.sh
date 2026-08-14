#!/usr/bin/env bash
# Gate suite for Qwen3.8-27B — DENSE qwen3_5 VISION-LANGUAGE model, layer-split across BOTH GPUs,
# MTP + mmproj loaded in ONE server. Hits the backend DIRECTLY (:8080) so the swap/router layer is
# not in the path (same convention as gates27q6 :8081 / gates35 :8082 / gates122 :8084). Works
# whether launched standalone OR loaded by llama-swap — either way the backend listens on :8080.
#
# WHAT THIS ENTRY EXERCISES THAT NOTHING ELSE DID:
#   1. VISION on this build: clip_graph_qwen3vl feeding the qwen3vl dense graph. gate [8] proves the
#      image pipeline end-to-end with a GROUND-TRUTHED solid-colour image (not a vibe check).
#   2. MTP draft/verify COEXISTING with --mmproj in one server (the 3.6 vision path ran WITHOUT MTP
#      over slot-position fears, #22867/#23371). gate [6] proves text requests still draft (draft_n>0);
#      if they don't, MTP+vision regressed on this build and the run script's MTP=off fallback applies.
#   3. The dense MTP verify loop re-decoding across a --tensor-split layer boundary (digit-loop canary).
set -uo pipefail
B="${B:-http://127.0.0.1:8080}"
PY=python3
HERE="$(cd "$(dirname "$0")" && pwd)"
command -v jq >/dev/null || { echo "need jq (~/.local/bin)"; exit 1; }
pass=0; fail=0
ok(){ echo "  PASS: $1"; pass=$((pass+1)); }
no(){ echo "  FAIL: $1"; fail=$((fail+1)); }

SYS='You are Qwen, created by Alibaba Cloud. You are a helpful assistant.'
chat(){ # $1=user text  $2=extra json
  local u="$1" extra="${2:-{\}}"
  jq -n --arg s "$SYS" --arg u "$u" --argjson e "$extra" \
    '{model:"qwen38",messages:[{role:"system",content:$s},{role:"user",content:$u}],stream:false} * $e' \
  | curl -s -m 300 "$B/v1/chat/completions" -H 'Content-Type: application/json' -d @-
}

echo "########## Qwen3.8-27B (dense qwen3_5, vision+MTP, dual-GPU layer split) gates ##########"

# ---- 1. Load + DUAL-GPU residency (weights on both cards, >=1 GiB free each) ----
echo "[1] load + dual-GPU residency (resident on both, >=1 GiB free each)"
h=$(curl -s -o /dev/null -w "%{http_code}" -m5 "$B/health")
[ "$h" = "200" ] && ok "3.8 healthy" || { no "health=$h"; echo "server not up; aborting"; exit 1; }
caps=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | tr '\n' ' ')
echo "$caps" | grep -q "8.9" && echo "$caps" | grep -q "12.0" && ok "GPUs cc: $caps" || no "GPU caps: $caps"
T0=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits -i 0|tr -d ' ')
T1=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits -i 1|tr -d ' ')
v0=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0|tr -d ' ')
v1=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 1|tr -d ' ')
free0=$((T0-v0)); free1=$((T1-v1))
echo "  GPU0=${v0}/${T0}MiB (free ${free0})  GPU1=${v1}/${T1}MiB (free ${free1})"
# weights (+mmproj) must be resident on BOTH cards (neither ~0 => split applied), >=1 GiB free on EACH.
if [ "$v0" -gt 18000 ] && [ "$v1" -ge 5000 ] && [ "$v1" -le 13000 ] \
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
CANARY="${CANARY:-$HERE/canary.py}"
# Self-test the detector before trusting any PASS: (1) input MUST exceed canary.py's MIN_LEN=200 or it
# exits 2 (VACUOUS) before the digit-loop logic runs; (2) require exit EXACTLY 1 (runaway), not merely
# non-zero — else a vacuous exit 2 reads as "fires". Both learned the hard way (see gates27q6/gates35).
printf 'x 1. 2. 3. 4. 5. 6. 7. 8. 9. 10. 11. 12. 13. 14. 15. 16. 17. 18. padding padding padding padding padding padding padding padding padding padding padding padding padding padding padding padding padding.' \
  | $PY "$CANARY" >/dev/null 2>&1
stc=$?
[ "$stc" -eq 1 ] && ok "canary detector self-test: fires on synthetic runaway (exit 1)" \
  || no "canary detector self-test did NOT fire (exit $stc; need 1=runaway — 2=vacuous is a FAIL) — gate is meaningless"
for T in 0.9 1.2; do
  r=$(canary $T | $PY "$CANARY"); rc=$?
  [ $rc -eq 0 ] && ok "canary temp=$T: $r" || no "canary temp=$T: $r"
done

# ---- 4. Thinking closes (embedded template parses reasoning; no raw <think> leak) ----
echo "[4] thinking closes with </think> (embedded 3.8 template)"
resp=$(chat "Think step by step: what is 17*23?" '{"max_tokens":2048}')
rc=$(echo "$resp" | jq -r '.choices[0].message.reasoning_content // ""')
ct=$(echo "$resp" | jq -r '.choices[0].message.content // ""')
[ -n "$rc" ] && ok "reasoning_content extracted (${#rc} chars)" || no "no reasoning_content (embedded template may not parse <think>)"
echo "$ct" | grep -q "391" && ok "answer 391 correct" || no "answer: $ct"
echo "$ct" | grep -q "<think>" && no "raw <think> leaked into content" || ok "no <think> leak in content"

# ---- 5. Tool calls at temp 0.6 AND 0.2 (embedded template tool parsing; no tool-in-think) ----
echo "[5] tool calls @ temp 0.6 and 0.2 (no tool-in-think)"
TOOLS='[{"type":"function","function":{"name":"run","description":"run a shell command","parameters":{"type":"object","properties":{"cmd":{"type":"string"}},"required":["cmd"]}}}]'
for T in 0.6 0.2; do
  r=$(jq -n --arg s "$SYS" --argjson t "$TOOLS" --arg tm "$T" \
      '{model:"qwen38",messages:[{role:"system",content:$s},{role:"user",content:"List /var/log using the run tool."}],tools:$t,max_tokens:2048,temperature:($tm|tonumber),stream:false}' \
    | curl -s -m 300 "$B/v1/chat/completions" -H 'Content-Type: application/json' -d @-)
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

# ---- 6. MTP live COEXISTING with --mmproj: draft_n>0 and AL>1 (text path still accelerates) ----
echo "[6] MTP live across split, mmproj loaded (draft_n / draft_n_accepted)"
t=$(curl -s -m 300 "$B/completion" -H 'Content-Type: application/json' \
     -d '{"prompt":"Write a Python function that merges two sorted lists.\n\ndef merge(","n_predict":220,"temperature":0.2,"cache_prompt":false}' \
   | jq -r '.timings | "\(.draft_n) \(.draft_n_accepted) \(.predicted_n) \(.predicted_per_second)"')
echo "  draft_n draft_n_accepted predicted_n tg_tps = $t"
echo "$t" | $PY -c '
import sys
dn,da,pn,tps=sys.stdin.read().split()
dn,da,pn,tps=int(dn),int(da),int(pn),float(tps)
if dn==0: print("MTP NOT DRAFTING with mmproj loaded (draft_n=0) -> set MTP=off or split servers"); sys.exit(1)
acc=da/dn; al=pn/(pn-da) if pn>da else 99
print(f"acceptance={acc:.3f} AL={al:.2f} tg={tps:.1f} t/s")
sys.exit(0 if (acc>0 and al>1.0) else 1)' && ok "MTP contributing (text path) with vision loaded" || no "MTP not contributing with mmproj loaded"

# ---- 7. Prefix-cache reuse ----
echo "[7] prefix cache reuse"
PFX=$($PY -c 'print(("The quick brown fox jumps over the lazy dog. "*400))')
body=$(jq -n --arg p "$PFX Summarize the above in 5 words." '{prompt:$p,n_predict:8,cache_prompt:true,temperature:0.0,stream:false}')
t1=$(curl -s -m 120 "$B/completion" -H 'Content-Type: application/json' -d "$body" | jq -r '.timings.prompt_ms // 0')
t2=$(curl -s -m 120 "$B/completion" -H 'Content-Type: application/json' -d "$body" | jq -r '.timings.prompt_ms // 0')
echo "  prompt_ms first=$t1 second=$t2"
$PY -c "import sys; sys.exit(0 if float('$t2')<=max(0.35*float('$t1'),500) else 1)" \
  && ok "reprefill dropped ($t1 -> $t2 ms)" || no "no cache reuse ($t1 -> $t2 ms)"

# ---- 8. VISION — HARD GATE: ground-truthed solid-colour image, model must name the colour ----
# Proves the mmproj/clip pipeline end-to-end (pixels -> vision tower -> LLM). Dependency-free PNG
# via python stdlib (zlib+struct); a solid RED 224x224 so a VLM answers "red" with high confidence.
# A broken/absent vision path shows up as "I can't see an image", an error, or the wrong colour.
echo "[8] vision: describe a ground-truthed image (mmproj pipeline)"
IMG=/tmp/qwen38_gate_red.png
$PY - "$IMG" <<'EOF'
import sys, zlib, struct
def chunk(t,d):
    c=t+d; return struct.pack(">I",len(d))+c+struct.pack(">I",zlib.crc32(c)&0xffffffff)
W=H=224
raw=bytearray()
for y in range(H):
    raw.append(0)                       # filter byte per scanline
    raw += bytes((220,30,30))*W         # solid red RGB
png=b"\x89PNG\r\n\x1a\n"+chunk(b"IHDR",struct.pack(">IIBBBBB",W,H,8,2,0,0,0))
png+=chunk(b"IDAT",zlib.compress(bytes(raw),9))+chunk(b"IEND",b"")
open(sys.argv[1],"wb").write(png)
EOF
if [ ! -s "$IMG" ]; then no "could not create test image"; else
  B64=$(base64 -w0 "$IMG" 2>/dev/null || base64 "$IMG" | tr -d '\n')
  vr=$(jq -n --arg s "$SYS" --arg u "What is the single dominant color of this image? Answer with just the color word." --arg d "data:image/png;base64,$B64" \
      '{model:"qwen38",max_tokens:512,temperature:0.2,stream:false,messages:[{role:"system",content:$s},{role:"user",content:[{type:"text",text:$u},{type:"image_url",image_url:{url:$d}}]}]}' \
    | curl -s -m 300 "$B/v1/chat/completions" -H 'Content-Type: application/json' -d @-)
  verr=$(echo "$vr" | jq -r '.error.message // ""')
  vans=$(echo "$vr" | jq -r '(.choices[0].message.content // "") ' )
  vall=$(echo "$vr" | jq -r '(.choices[0].message.reasoning_content // "") + " " + (.choices[0].message.content // "")')
  if [ -n "$verr" ]; then no "vision request errored: $verr (build may not accept image_url with this mmproj)"
  elif echo "$vall" | grep -qiE '\bred\b|crimson|scarlet'; then ok "vision works: model read the image as red — \"$(echo "$vans" | tr '\n' ' ' | cut -c1-60)\""
  elif echo "$vall" | grep -qiE "can'?t see|no image|unable to (see|view)|don'?t see"; then no "vision NOT wired: model says it cannot see the image — \"$vans\""
  else no "vision wrong colour (expected red): \"$(echo "$vall" | tr '\n' ' ' | cut -c1-80)\""; fi
fi

# ---- 9. >=100K needle recall ----
echo "[9] 100K needle recall"
NEEDLE="$HERE/needle35.py"   # self-calibrates via /tokenize -> adapts to the tokenizer
if [ -f "$NEEDLE" ]; then
  MODEL="qwen38" $PY "$NEEDLE" "$B" 100000 && ok "100K needle recalled" || no "100K needle NOT recalled"
else
  echo "  (needle35.py absent — skipping)"; fi

echo; echo "########## Qwen3.8-27B gates: $pass PASS / $fail FAIL ##########"
[ "$fail" -eq 0 ] || exit 1
