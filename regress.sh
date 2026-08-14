#!/usr/bin/env bash
# Qwen regression suite for the 27B (requirement #4 enforcement: CLEAN OUTPUT).
#
# Runs entirely against the router (:8000) -> llama-swap -> qwen-38-27b (via the legacy alias "qwen36").
# Under llama-swap, backend ports are LOAD-ON-DEMAND: :8080 exists only while qwen-38-27b (Qwen3.8-27B
# dense+vision) is loaded. The old separate 27B-Q4 (:8080) and 27B-Q6 (:8081) entries were REPLACED by
# the single Qwen3.8-27B multimodal server; :8081 is now unused. NB history: :8081 was vision's before
# the 3.6 vision path was retired, then briefly the 3.6-Q6 text server -- assuming a port is
# "permanently X" is a live hazard (soak.py once held "8081 == vision" and POSTed images to a text server).
# Tests [1] and [11] originally failed for architectural reasons rather than model reasons -- and this
# suite silently stopped guarding the 27B at all.
#   - vision test [11] and the image half of [12] are DELETED (vision retired, not broken)
#   - [0] is NEW: asserts every registered model HAS a live gate -- the assertion whose absence
#     let this suite rot unnoticed through two re-architectures.
#
# PASS/FAIL per test; nonzero exit if any hard test fails. Soak (10) runs only if SOAK_MIN>0.
set -uo pipefail

ROUTER="${ROUTER:-http://127.0.0.1:8000}"
SWAP="${SWAP:-http://127.0.0.1:9000}"
MODEL_ID="${MODEL_ID:-qwen36}"      # legacy alias -> qwen-38-27b (carried forward from the retired 3.6 27B)
# Tests 5/8/9 need DIRECT backend endpoints (/apply-template, /completion timings) that the chat
# API cannot express. The old TEXT=:8080 assumed an always-resident server; llama-swap instead
# exposes /upstream/<model_id> for exactly this, which works regardless of which port it chose.
# NB: the model must be LOADED first -- test [2] does that before any of these run.
TEXT="${TEXT:-$SWAP/upstream/qwen-38-27b}"
HERE="$(cd "$(dirname "$0")" && pwd)"
PY=python3
command -v jq >/dev/null || { echo "need jq"; exit 1; }
pass=0; fail=0; skip=0
ok(){ echo "  PASS: $1"; pass=$((pass+1)); }
no(){ echo "  FAIL: $1"; fail=$((fail+1)); }
sk(){ echo "  SKIP: $1"; skip=$((skip+1)); }

# helper: chat completion via router, returns the full JSON
chat(){ # $1=json messages array  $2=extra json (opt)
  local msgs="$1" extra="${2:-{\}}"
  jq -n --arg mid "$MODEL_ID" --argjson m "$msgs" --argjson e "$extra" \
     '{model:$mid,messages:$m,stream:false} * $e' \
    | curl -s -m 600 "$ROUTER/v1/chat/completions" -H 'Content-Type: application/json' -d @-
}

echo "################ Qwen regression (27B) ################"

# ---- 0. EVERY REGISTERED MODEL HAS A LIVE GATE ----
# WHY THIS EXISTS: the llama-swap re-architecture orphaned this very suite. gates35.sh and
# gates122.sh covered the newer models while regress.sh still pointed at a retired two-backend
# design -- so the 27B's canary/tool-in-think/thinking-closure assertions quietly stopped running
# and NOTHING noticed, because nothing asserted the mapping. This closes that hole in BOTH
# directions: a model with no gate fails, and a gate for a model that no longer exists fails.
echo "[0] every model has a live gate"
declare -A GATE_FOR=(
  [qwen-38-27b]="gates38.sh"  # Qwen3.8-27B dense+vision (dual-GPU layer split) — deep direct-backend gate
  [qwen-35b]="gates35.sh"
  [qwen-122b]="gates122.sh"
)
# NB: this file (regress.sh) is the ROUTER-integration suite for the primary model; gates38.sh is its
# mapped deep gate (analogous to gates35/gates122). Both are run; [0] only needs each model to map to one.
ids=$(curl -sf -m5 "$ROUTER/v1/models" | jq -r '.data[].id' 2>/dev/null | sort)
if [ -z "$ids" ]; then
  no "could not enumerate models from $ROUTER/v1/models"
else
  # direction 1: every registered model must have a gate file that exists and is executable
  for m in $ids; do
    g="${GATE_FOR[$m]:-}"
    if [ -z "$g" ]; then
      no "model '$m' is registered but has NO gate suite (add one to GATE_FOR)"
    elif [ ! -x "$HERE/$g" ]; then
      no "model '$m' maps to '$g' but it is missing or not executable"
    else
      ok "model '$m' -> $g (present, executable)"
    fi
  done
  # direction 2: every gate must correspond to a still-registered model (no dead gates)
  for m in "${!GATE_FOR[@]}"; do
    echo "$ids" | grep -qx "$m" || no "gate '${GATE_FOR[$m]}' targets model '$m' which is NOT registered (dead gate)"
  done
fi

# ---- 1. Load / health / GPU enumeration ----
# Under llama-swap the backend ports are NOT always-resident, so health is llama-swap answering +
# the 27B actually serving a request. An idle GPU is the correct resting state, not a failure.
echo "[1] load + health"
sh=$(curl -s -o /dev/null -w "%{http_code}" -m5 "$SWAP/v1/models")
rh=$(curl -s -o /dev/null -w "%{http_code}" -m5 "$ROUTER/healthz")
[ "$sh" = "200" ] && [ "$rh" = "200" ] && ok "llama-swap + router healthy ($sh/$rh)" || no "health swap=$sh router=$rh"
caps=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | tr '\n' ' ')
echo "$caps" | grep -q "8.9" && echo "$caps" | grep -q "12.0" && ok "GPUs cc: $caps" || no "GPU caps: $caps"

# ---- 2. Coherence (no multi-script salad) ----
echo "[2] coherence"
c=$(chat '[{"role":"user","content":"What is the capital of France? Answer in one word."}]' | jq -r '.choices[0].message.content // ""')
echo "$c" | grep -qi "paris" && ok "capital=Paris" || no "got: $c"
echo "$c" | $PY -c 'import sys,re; t=sys.stdin.read(); bad=len(re.findall(r"[Ѐ-ӿ؀-ۿ一-鿿ऀ-ॿ]",t)); sys.exit(1 if bad>3 else 0)' && ok "no script-salad" || no "multi-script salad detected"

# ---- 3. Digit-loop / MTP verify-loop canary ----
echo "[3] digit-loop canary (MTP correctness gate)"
out=$(chat '[{"role":"user","content":"Repeat back this random string then continue writing prose about it: qX7#mept Zq9 vbrr 42a! kkap. Do not make a numbered list."}]' '{"max_tokens":600,"temperature":0.9}' | jq -r '.choices[0].message.content // ""')
echo "$out" | $PY -c '
import sys,re
t=sys.stdin.read()
# ascending integer runaway like "1. 2. 3. ... "
nums=re.findall(r"\b(\d+)\b", t)
asc=1; mx=1
for a,b in zip(nums, nums[1:]):
    if int(b)==int(a)+1: asc+=1; mx=max(mx,asc)
    else: asc=1
# single-token spam
toks=t.split()
rep=1; mrep=1
for a,b in zip(toks, toks[1:]):
    if a==b: rep+=1; mrep=max(mrep,rep)
    else: rep=1
if mx>=15 or mrep>=30:
    print(f"RUNAWAY asc_run={mx} tok_rep={mrep}"); sys.exit(1)
print(f"ok asc_run={mx} tok_rep={mrep}"); sys.exit(0)
' && ok "no digit/token runaway" || no "digit-loop / repetition runaway (investigate MTP; fall back --spec-type none)"

# ---- 4. Thinking mode closes with </think> ----
echo "[4] thinking tags"
tk=$(chat '[{"role":"user","content":"Briefly, is 17 prime? Show reasoning."}]' '{"max_tokens":500}' | jq -r '.choices[0].message.content // .choices[0].message.reasoning_content // ""')
# note: llama-server may split reasoning into reasoning_content; check raw too
raw=$(chat '[{"role":"user","content":"Briefly, is 17 prime? Show your thinking."}]' '{"max_tokens":500}')
echo "$raw" | grep -q "</thinking>" && no "found </thinking> (should be </think>)" || ok "no bad </thinking> tag"

# ---- 5. Multi-turn preserve_thinking via /apply-template ----
echo "[5] preserve_thinking (template render)"
tpl=$(jq -n '{messages:[
  {"role":"user","content":"hi"},
  {"role":"assistant","content":"<think>user greeted</think>Hello!"},
  {"role":"user","content":"and now?"},
  {"role":"assistant","content":"<think>followup</think>Sure."},
  {"role":"user","content":"ok"}]}' | curl -s "$TEXT/apply-template" -H 'Content-Type: application/json' -d @- | jq -r '.prompt // ""')
if [ -n "$tpl" ]; then
  echo "$tpl" | grep -qE "<think>\s*</think>|<think/>" && no "empty <think/> injected into past turns" || ok "no empty <think/> in history"
else sk "no /apply-template output"; fi

# ---- 6. Tool calls at temp 0.6 and 0.2 ----
echo "[6] tool calls (temp 0.6 & 0.2)"
tools='[{"type":"function","function":{"name":"get_weather","description":"weather for a city","parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}}]'
for T in 0.6 0.2; do
  r=$(jq -n --argjson tl "$tools" --argjson temp "$T" \
    '{model:"qwen36",stream:false,temperature:$temp,tools:$tl,messages:[{"role":"user","content":"What is the weather in Paris? Use the tool."}]}' \
    | curl -s "$ROUTER/v1/chat/completions" -H 'Content-Type: application/json' -d @-)
  code=$(echo "$r" | jq -r 'if .error then "ERR" else "OK" end')
  args=$(echo "$r" | jq -r '.choices[0].message.tool_calls[0].function.arguments // ""')
  incthink=$(echo "$r" | jq -r '.choices[0].message.content // ""' | grep -c "<think>" || true)
  if [ "$code" = "ERR" ]; then no "temp $T: template/error: $(echo "$r"|jq -c .error)"
  elif [ -z "$args" ] || [ "$args" = "{}" ]; then no "temp $T: empty tool args"
  elif echo "$args" | grep -qi "paris"; then ok "temp $T: tool_call args ok ($args)"
  else no "temp $T: args missing city ($args)"; fi
done

# ---- 7. System-message contract (Qwen3.8: system MUST be first) ----
# The 3.8 embedded template hard-raises on a system/developer message that is not first (unlike the
# retired 3.6 v19, which tolerated mid-conversation injection). This is BY DESIGN, and the router only
# ever PREPENDS system (proxy.py: msgs.insert(0,...) when msg[0] isn't system/developer), so no real
# traffic through :8000 can trip it. Assert BOTH halves so a future template silently re-allowing
# mid-convo system (a behaviour change worth knowing) still fails here.
echo "[7] system-message contract (3.8: system must be first)"
sf=$(chat '[{"role":"system","content":"Be terse."},{"role":"user","content":"say ok"}]' '{"max_tokens":200}')
echo "$sf" | jq -e '.error' >/dev/null 2>&1 && no "system-first was rejected: $(echo "$sf"|jq -c .error.message)" || ok "system-first accepted"
mid=$(chat '[{"role":"user","content":"hi"},{"role":"assistant","content":"hello"},{"role":"system","content":"Be terse."},{"role":"user","content":"say ok"}]' '{"max_tokens":200}')
if echo "$mid" | jq -e '.error.message | test("must be at the beginning"; "i")' >/dev/null 2>&1; then
  ok "mid-convo system correctly rejected (3.8 contract: system must be first)"
elif echo "$mid" | jq -e '.error' >/dev/null 2>&1; then
  no "mid-convo system rejected but with an UNEXPECTED error: $(echo "$mid"|jq -c .error.message)"
else
  no "mid-convo system was ACCEPTED — 3.8 template contract changed (expected rejection); re-check template"
fi

# ---- 8. Prefix cache reuse (3K prefix twice) ----
echo "[8] prefix cache reuse"
PFX=$($PY -c 'print(("The quick brown fox jumps over the lazy dog. "*400))')
b=$(jq -n --arg p "$PFX Summarize the above in 5 words." '{prompt:$p,n_predict:8,cache_prompt:true,temperature:0.0,stream:false}')
t1=$(curl -s "$TEXT/completion" -H 'Content-Type: application/json' -d "$b" | jq -r '.timings.prompt_ms // 0')
t2=$(curl -s "$TEXT/completion" -H 'Content-Type: application/json' -d "$b" | jq -r '.timings.prompt_ms // 0')
echo "  prompt_ms first=$t1 second=$t2"
$PY -c "import sys; sys.exit(0 if float('$t2')<=max(0.35*float('$t1'),500) else 1)" && ok "reprefill dropped ($t1 -> $t2 ms)" || no "no cache reuse ($t1 -> $t2 ms)"

# ---- 9. Long-context 100K ----
echo "[9] 100K context"
# build+send in python (prompt is ~500KB -> exceeds shell ARG_MAX for jq)
r=$($PY - "$TEXT" <<'PYEOF'
import sys, json, urllib.request
base=sys.argv[1]
big="Fact: the secret code is TANGERINE-91. " + " ".join("Filler sentence number %d."%i for i in range(1,10500)) + " What is the secret code? Answer with just the code."  # ~107K tokens (<131072)
body=json.dumps({"prompt":big,"n_predict":16,"temperature":0.0,"cache_prompt":True,"stream":False}).encode()
req=urllib.request.Request(base+"/completion", data=body, headers={"Content-Type":"application/json"})
d=json.load(urllib.request.urlopen(req, timeout=600))
print(json.dumps({"prompt_n":d.get("timings",{}).get("prompt_n",0),"content":d.get("content","")}))
PYEOF
)
ntok=$(echo "$r" | jq -r '.prompt_n // 0'); ans=$(echo "$r" | jq -r '.content // ""')
echo "  prompt_n=$ntok answer=$ans"
[ "$ntok" -ge 90000 ] 2>/dev/null && echo "$ans" | grep -qi "TANGERINE-91" && ok "100K recall ok (prompt_n=$ntok)" || no "100K failed (prompt_n=$ntok ans=$ans)"

# ---- 10. Soak (opt-in) ----
echo "[10] soak"
if [ "${SOAK_MIN:-0}" -gt 0 ] 2>/dev/null; then
  echo "  running ${SOAK_MIN} min soak, sampling every 30s -> /tmp/qwen_soak.csv"
  end=$(( $(date +%s) + SOAK_MIN*60 )); echo "ts,rss_kb,vram0,vram1" > /tmp/qwen_soak.csv
  while [ "$(date +%s)" -lt "$end" ]; do
    chat '[{"role":"user","content":"Write a short haiku about entropy."}]' '{"max_tokens":80}' >/dev/null
    rss=$(ps --no-headers -o rss -C llama-server | paste -sd+ | bc 2>/dev/null || echo 0)
    v0=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0|tr -d ' ')
    v1=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 1|tr -d ' ')
    echo "$(date +%s),$rss,$v0,$v1" >> /tmp/qwen_soak.csv
  done
  $PY -c '
import csv
r=list(csv.DictReader(open("/tmp/qwen_soak.csv")))
if len(r)<4: print("  too few samples"); raise SystemExit
f=lambda k:[int(x[k]) for x in r]
import statistics
def climb(v):
    first=statistics.mean(v[:max(1,len(v)//4)]); last=statistics.mean(v[-max(1,len(v)//4):]); return last-first
print(f"  rss climb {climb(f(\"rss_kb\"))//1024} MiB, vram0 climb {climb(f(\"vram0\"))} MiB, vram1 climb {climb(f(\"vram1\"))} MiB")
import sys; sys.exit(1 if climb(f("vram0"))>800 or climb(f("rss_kb"))>2_000_000 else 0)
' && ok "no monotonic climb" || no "memory climb during soak"
else sk "soak (set SOAK_MIN>0 to run at M4)"; fi

# ---- 11. (RETIRED) vision ----
# Vision was retired with the llama-swap re-architecture: both GPUs now serve one text model at a
# time and there is no mmproj backend. This test is deleted rather than skipped -- a permanent SKIP
# is just noise that trains you to ignore the suite's output.

# ---- 12. Alias dispatch: "qwen36" must still reach the 27B ----
# Replaces the old image-vs-text routing test. The routing that matters now is by MODEL NAME
# (llama-swap's job), and the load-bearing property is that the legacy alias existing clients send
# still lands on the 27B -- NOT on a newer model. Silently repointing it would change behaviour
# under working clients.
echo "[12] alias dispatch (qwen36 -> qwen-38-27b)"
chat '[{"role":"user","content":"reply with the single word ping"}]' '{"max_tokens":2048}' >/dev/null
served=$(curl -sf -m5 "$ROUTER/healthz" | jq -r '.backend.running|join(",")' 2>/dev/null)
echo "  alias '$MODEL_ID' served-by=$served"
[ "$served" = "qwen-38-27b" ] && ok "alias '$MODEL_ID' -> qwen-38-27b" || no "alias '$MODEL_ID' landed on '$served' (expected qwen-38-27b)"

# ---- 13. Tool-call budget guard (regression for the M4 soak's 10 errors) ----
echo "[13] tool-call truncation guard"
GTOOLS='[{"type":"function","function":{"name":"run","description":"run a shell command","parameters":{"type":"object","properties":{"cmd":{"type":"string"}},"required":["cmd"]}}}]'
# small max_tokens + tools: DIRECT to text backend should truncate the tool call (finish=length / partial args);
# through the ROUTER, the guard raises max_tokens so the tool call completes with valid JSON args.
guard_check() {
  jq -n --argjson tl "$GTOOLS" '{model:"qwen36",stream:false,max_tokens:80,temperature:0.4,tools:$tl,messages:[{"role":"user","content":"Call the run tool to do a long detailed listing of /var/log."}]}' \
    | curl -s "$1/v1/chat/completions" -H 'Content-Type: application/json' -d @-
}
rr=$(guard_check "$ROUTER")
rargs=$(echo "$rr" | jq -r '.choices[0].message.tool_calls[0].function.arguments // ""')
rfin=$(echo "$rr" | jq -r '.choices[0].finish_reason // ""')
rerr=$(echo "$rr" | jq -r 'if .error then "ERR" else "ok" end')
echo "  router: finish=$rfin args=$rargs"
# valid JSON args (closing brace present) and not a 500 => guard worked
if [ "$rerr" = "ERR" ]; then no "router tool call errored: $(echo "$rr"|jq -c .error.message)"
elif echo "$rargs" | jq -e . >/dev/null 2>&1 && echo "$rargs" | grep -q "}"; then ok "router guard: tool args valid JSON ($rargs)"
else no "router guard failed: truncated/invalid args ($rargs, finish=$rfin)"; fi

echo "###################################################"
echo "PASS=$pass FAIL=$fail SKIP=$skip"
[ "$fail" -eq 0 ]
