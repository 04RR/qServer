#!/usr/bin/env bash
# Follow-up: 2e config (MTP off, f16, UB=3072, NCM31, TS40,8, vision on) with T=24, measure
# code-short decode WHILE a CPU embedder serves realistic RAG query traffic (one ~50-tok embed / 2s).
# Bounded + self-cleaning: preflight -> embedder(:8097,child) -> 2e server(:8095,child) -> qload bg
# -> bench2 code cell -> stop qload -> SIGTERM both (pidfiles only, never pkill -f).
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"                 # qServer flashnext/bench (this script's dir)
FLROOT="$(cd "$HERE/.." && pwd)"                      # qServer flashnext/ (the in-repo launcher lives here)
BUILD_ROOT="${BUILD_ROOT:-$HOME/ai/flashnext-main}"   # the mainline llama.cpp BUILD stays here (not copied in-repo); override with BUILD_ROOT=...
BDIR="${BDIR:-$HERE}"; mkdir -p "$BDIR"
MAIN="$FLROOT/run-38-flash-next-main.sh"
EMB="$HOME/ai/qwen36/run-embed.sh"
SRV_PID="$BDIR/t24q_srv.pid"; EMB_PID="$BDIR/t24q_emb.pid"
SRV_LOG="$BDIR/t24q_srv.log"; EMB_LOG="$BDIR/t24q_emb.log"; QL_LOG="$BDIR/t24q_qload.log"
STOP="$BDIR/.qload_stop"; OUT="$BDIR/t24q.json"
SPORT=8095; EPORT=8097
SBASE="http://127.0.0.1:$SPORT"; EBASE="http://127.0.0.1:$EPORT"
say(){ echo "[t24q] $*"; }

cleanup(){
  touch "$STOP" 2>/dev/null
  [ -n "${QLPID:-}" ] && kill -0 "$QLPID" 2>/dev/null && { kill -TERM "$QLPID" 2>/dev/null; sleep 1; }
  for pf in "$SRV_PID" "$EMB_PID"; do
    [ -f "$pf" ] || continue
    p=$(cat "$pf" 2>/dev/null)
    [ -n "$p" ] && kill -0 "$p" 2>/dev/null && { kill -TERM "$p" 2>/dev/null; for _ in $(seq 1 30); do kill -0 "$p" 2>/dev/null || break; sleep 1; done; kill -0 "$p" 2>/dev/null && kill -9 "$p" 2>/dev/null; }
    rm -f "$pf"
  done
  rm -f "$STOP"
}
trap cleanup EXIT INT TERM

wait_health(){ local base="$1" pid="$2" to="$3" t0 r=0; t0=$(date +%s)
  while [ $(( $(date +%s) - t0 )) -lt "$to" ]; do
    kill -0 "$pid" 2>/dev/null || { say "FAIL: pid $pid died"; return 4; }
    curl -sf --max-time 3 "$base/health" >/dev/null 2>&1 && { r=1; break; }
    sleep 2
  done
  [ "$r" = 1 ] || { say "FAIL: $base /health not up in ${to}s"; return 4; }
  say "health OK: $base in $(( $(date +%s) - t0 ))s"
}

# ---- preflight (hard rule) ----
run=$(curl -s --max-time 5 localhost:9000/running | tr -d ' ')
[ "$run" = '{"running":[]}' ] || { say "ABORT: llama-swap not idle: $run"; exit 3; }
read g0 g1 < <(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | paste -sd' ' -)
{ [ "${g0:-9999}" -ge 500 ] || [ "${g1:-9999}" -ge 500 ]; } && { say "ABORT: GPU busy ($g0/$g1)"; exit 3; }
ma=$(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo)
[ "$ma" -lt 40000 ] && { say "ABORT: MemAvailable ${ma} < 40000"; exit 3; }
say "preflight OK: running=[] GPUs ${g0}/${g1} MemAvailable ${ma}"

# ---- embedder (:8097, CPU, THREADS=8 = production), lean CTX (query embeds are tiny) ----
say "launch embedder (:$EPORT, CPU, THREADS=8, CTX=2048)"
PORT=$EPORT THREADS=8 CTX=2048 "$EMB" > "$EMB_LOG" 2>&1 &
EPID=$!; echo "$EPID" > "$EMB_PID"
wait_health "$EBASE" "$EPID" 120 || { say "embedder failed"; tail -20 "$EMB_LOG"; exit 4; }
# confirm embedder is CPU-only: GPUs still clean
read g0 g1 < <(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | paste -sd' ' -)
{ [ "$g0" -ge 500 ] || [ "$g1" -ge 500 ]; } && { say "ABORT: embedder took VRAM ($g0/$g1)"; exit 3; }
say "embedder up, GPUs still ${g0}/${g1}"

# ---- 2e server (:8095) with T=24 ----
say "launch 2e server (:$SPORT, SPEC_OFF=1 f16 UB=3072 T=24 NCM31 TS40,8 vision on)"
env BIN="$BUILD_ROOT/llama.cpp/build/bin/llama-server" \
    SPEC_OFF=1 KVK=f16 KVV=f16 UB=3072 T=24 CKPT=8 NO_WARMUP=1 \
    NCM=31 TS=40,8 MG=0 CTX=131072 VISION=on NP=1 PORT=$SPORT \
    "$MAIN" > "$SRV_LOG" 2>&1 &
SPID=$!; echo "$SPID" > "$SRV_PID"
wait_health "$SBASE" "$SPID" 300 || { say "2e server failed"; tail -30 "$SRV_LOG"; exit 4; }
read g0 g1 < <(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | paste -sd' ' -)
say "2e up; VRAM GPU0=$g0 GPU1=$g1 (free $((24564-g0))/$((16311-g1)))"

# ---- start realistic query load (one ~50-tok embed / 2s), runs until we stop it ----
say "start qload (one ~50-tok embed / 2s) vs embedder"
EMBED_BASE="$EBASE" INTERVAL=2.0 MAX_S=600 STOP_FILE="$STOP" OUT="$BDIR/t24q_qload.json" \
  python3 "$HERE/qload.py" > "$QL_LOG" 2>&1 &
QLPID=$!
sleep 6   # let a few embeds land so the load is established before we measure

# ---- measure code-short decode on :8095 UNDER the query load ----
say "bench2 code cell under load -> $OUT"
BASE="$SBASE" ARM="t24q" OUT="$OUT" PIDFILE="$SRV_PID" CELLS="code" NSEED=3 CTX=131072 \
  python3 "$HERE/bench2.py" || say "bench2 returned $?"

# ---- stop load, fold qload summary into result ----
touch "$STOP"; sleep 1
for _ in $(seq 1 20); do kill -0 "$QLPID" 2>/dev/null || break; sleep 1; done
QL=$(cat "$BDIR/t24q_qload.json" 2>/dev/null || echo '{}')
say "qload summary: $QL"
OUT="$OUT" QL="$QL" python3 - <<'PY' || true
import os, json
o=os.environ["OUT"]
try: d=json.load(open(o))
except Exception: d={}
try: d["qload"]=json.loads(os.environ["QL"])
except Exception: d["qload"]={}
json.dump(d, open(o,"w"), indent=2)
print("folded qload into", o)
PY
say "DONE -> $OUT (server log: $SRV_LOG)"
