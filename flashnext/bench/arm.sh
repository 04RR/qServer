#!/usr/bin/env bash
# Run ONE benchmark arm end-to-end, bounded and self-cleaning:
#   preflight gate -> launch server (child, pidfile) -> wait /health -> bench.py -> SIGTERM -> capture.
# The server is a CHILD of this script and is always killed before exit (no orphan, no long-lived bg server).
# Caller sets: ARM, LAUNCHER, PORT(=8095) and the server-config env (NCM/TS/UB/SPEC_UB/MOE_CACHE/KVK/KVV/
# SPEC_OFF/VISION/NP/NMAX/DRAFT_SAMPLING/GGML_CUDA_REGISTER_HOST/BIN). Kill ONLY by pidfile (never pkill -f).
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"   # qServer flashnext/bench (this script's dir; was ~/ai/flashnext-main)
ARM="${ARM:?set ARM}"
LAUNCHER="${LAUNCHER:?set LAUNCHER}"
PORT="${PORT:-8095}"
BASE="http://127.0.0.1:${PORT}"
BDIR="${BDIR:-$HERE}"   # bench outputs land alongside the scripts (override with BDIR=...)
PID="$BDIR/$ARM.pid"; SLOG="$BDIR/$ARM.server.log"; OUT="$BDIR/$ARM.json"
HEALTH_TIMEOUT="${HEALTH_TIMEOUT:-300}"
mkdir -p "$BDIR"

say(){ echo "[arm:$ARM] $*"; }

# ---- preflight gate (hard rule) ----
run=$(curl -s --max-time 5 localhost:9000/running | tr -d ' ')
if [ "$run" != '{"running":[]}' ]; then say "ABORT: llama-swap not idle: $run"; exit 3; fi
read g0 g1 < <(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | paste -sd' ' -)
if [ "${g0:-9999}" -ge 500 ] || [ "${g1:-9999}" -ge 500 ]; then say "ABORT: a GPU is busy ($g0/$g1 MiB)"; exit 3; fi
ma=$(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo)
if [ "$ma" -lt 40000 ]; then say "ABORT: MemAvailable ${ma} MiB < 40000 (let page cache drop)"; exit 3; fi
say "preflight OK: running=[] GPUs ${g0}/${g1} MiB, MemAvailable ${ma} MiB"

# ---- launch server (child), wait /health ----
say "launch: $LAUNCHER (PORT=$PORT)"
"$LAUNCHER" > "$SLOG" 2>&1 &
SPID=$!; echo "$SPID" > "$PID"
say "server pid=$SPID; waiting /health (<=${HEALTH_TIMEOUT}s)"
t0=$(date +%s); ready=0
while [ $(( $(date +%s) - t0 )) -lt "$HEALTH_TIMEOUT" ]; do
  if ! kill -0 "$SPID" 2>/dev/null; then say "FAIL: server died during load"; tail -30 "$SLOG"; rm -f "$PID"; exit 4; fi
  if curl -sf --max-time 3 "$BASE/health" >/dev/null 2>&1; then ready=1; break; fi
  sleep 2
done
hr=$(( $(date +%s) - t0 ))
if [ "$ready" != 1 ]; then say "FAIL: /health not up after ${hr}s"; tail -30 "$SLOG"; kill -TERM "$SPID" 2>/dev/null; exit 4; fi
say "health ready in ${hr}s"

# ---- bench ----
BENCH="${BENCH:-bench.py}"
say "benchmarking ($BENCH) -> $OUT"
BASE="$BASE" ARM="$ARM" OUT="$OUT" SPID="$SPID" PIDFILE="$PID" \
  python3 "$HERE/$BENCH" || say "$BENCH returned $?"

# ---- optional concurrent throughput bench (CONC_BENCH=1, for the batch lane / arm G) ----
if [ "${CONC_BENCH:-0}" = "1" ]; then
  say "concurrent bench (CONC=${CONC:-1,4,8}) -> $BDIR/$ARM.conc.json"
  BASE="$BASE" ARM="$ARM" OUT="$BDIR/$ARM.conc.json" CONC="${CONC:-1,4,8}" \
    python3 "$HERE/bench_conc.py" || say "bench_conc.py returned $?"
fi

# ---- optional correctness gate (GATE=1 runs the isolated-tree gates38next against this server) ----
if [ "${GATE:-0}" = "1" ]; then
  say "gate: gates38next.sh (SKIP_DEEP=${SKIP_DEEP:-0}) vs $BASE"
  SKIP_DEEP="${SKIP_DEEP:-0}" B="$BASE" MODEL_ID="flash-next" \
    bash "$HERE/../gates38next.sh" > "$BDIR/$ARM.gate.log" 2>&1
  say "gate exit=$? ; summary: $(grep -iE 'PASS / .*FAIL|single-server gates' "$BDIR/$ARM.gate.log" | tail -1)"
fi

# ---- optional concurrent-burst VRAM probe (BURST=1): 8x 8K-prompt burst, peak VRAM >=1GiB/GPU ----
if [ "${BURST:-0}" = "1" ]; then
  say "conc_burst (CONC=${BURST_CONC:-8}, PTOK=${BURST_PTOK:-8000}) -> $BDIR/$ARM.burst.json"
  BASE="$BASE" ARM="$ARM" OUT="$BDIR/$ARM.burst.json" CONC="${BURST_CONC:-8}" PTOK="${BURST_PTOK:-8000}" \
    python3 "$HERE/conc_burst.py" || say "conc_burst.py returned $?"
fi

# ---- optional vision VRAM probe (VISION_PROBE=1): peak VRAM w/ max-res image, alone + after 100K ----
if [ "${VISION_PROBE:-0}" = "1" ]; then
  say "vision_probe -> $BDIR/$ARM.vprobe.json"
  BASE="$BASE" ARM="$ARM" OUT="$BDIR/$ARM.vprobe.json" \
    python3 "$HERE/vision_probe.py" || say "vision_probe.py returned $?"
fi

# ---- optional 5-needle retrieval gate (NEEDLE5=1): recall at 100K and 125K (q8_0 gate, arm 3) ----
if [ "${NEEDLE5:-0}" = "1" ]; then
  for depth in 100000 125000; do
    say "needle5: $depth vs $BASE"
    MODEL=flash-next NCTX="${CTX:-131072}" python3 "$HERE/needle5.py" "$BASE" "$depth" "${PASS5:-4}" \
      > "$BDIR/$ARM.needle5_$depth.log" 2>&1
    n5=$?; echo "NEEDLE5_EXIT=$n5" >> "$BDIR/$ARM.needle5_$depth.log"
    say "needle5 $depth exit=$n5 ; $(grep -E 'recalled|ABORT|HTTP' "$BDIR/$ARM.needle5_$depth.log" | tail -1)"
  done
fi

# ---- optional standalone 100K needle (NEEDLE=1): deep-context correctness + a 2nd P/D-100K sample ----
if [ "${NEEDLE:-0}" = "1" ]; then
  say "needle: needle-next.py 100000 vs $BASE"
  MODEL=flash-next NCTX="${CTX:-131072}" python3 "$HERE/../needle-next.py" "$BASE" 100000 > "$BDIR/$ARM.needle.log" 2>&1
  nc=$?; echo "NEEDLE_EXIT=$nc" >> "$BDIR/$ARM.needle.log"
  say "needle exit=$nc ; $(tail -2 "$BDIR/$ARM.needle.log" | tr '\n' ' ')"
fi

# ---- stop server, capture MoE cache hit-rate + health time, merge ----
say "SIGTERM server"
kill -TERM "$SPID" 2>/dev/null
for _ in $(seq 1 30); do kill -0 "$SPID" 2>/dev/null || break; sleep 1; done
kill -0 "$SPID" 2>/dev/null && { say "escalate SIGKILL"; kill -9 "$SPID" 2>/dev/null; }
rm -f "$PID"

HR="$hr" SLOG="$SLOG" OUT="$OUT" ARMN="$ARM" python3 - <<'PY' || true
import json, os, re
out = os.environ["OUT"]
try:
    d = json.load(open(out))
except Exception:
    d = {}
d["health_ready_s"] = int(os.environ["HR"])
log = open(os.environ["SLOG"], errors="ignore").read()
hits = re.findall(r"(llama_moe_cache.*|moe[_ ]cache.*hit.*|.*hit rate.*)", log, re.I)
d["moe_cache_log"] = hits[-8:] if hits else []
# flag NVMe re-reads / implausible smpbo seen at init
d["smpbo_warn"] = bool(re.search(r"implausible sharedMemPerBlockOptin", log))
# capture gate summary if a gate ran
gpath = os.path.join(os.path.dirname(out), os.environ.get("ARMN", "") + ".gate.log")
try:
    gl = open(gpath, errors="ignore").read()
    m = re.search(r"(\d+)\s*PASS\s*/\s*(\d+)\s*FAIL", gl)
    if m:
        d["gate"] = {"pass": int(m.group(1)), "fail": int(m.group(2))}
    d["gate_lines"] = [l for l in gl.splitlines() if re.search(r"PASS:|FAIL:", l)]
except FileNotFoundError:
    pass
npath = os.path.join(os.path.dirname(out), os.environ.get("ARMN", "") + ".needle.log")
try:
    nl = open(npath, errors="ignore").read()
    me = re.search(r"NEEDLE_EXIT=(\d+)", nl)
    d["needle"] = {"recalled": (me.group(1) == "0") if me else None,
                   "lines": [l for l in nl.splitlines() if re.search(r"calibrated|prompt_tokens|answer=|NEEDLE_EXIT", l)]}
except FileNotFoundError:
    pass
json.dump(d, open(out, "w"), indent=2)
print("merged health_ready_s + moe_cache_log into", out)
PY

say "DONE (health ${hr}s). server log: $SLOG  result: $OUT"
