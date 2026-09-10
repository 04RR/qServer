#!/usr/bin/env bash
# Minimal gate for the Qwen3-Embedding-0.6B embedders. Hits /v1/embeddings directly (default :8086, or $B).
# Asserts: 1024-dim, L2-normalized (~1.0), and that semantically-close inputs are nearer than far ones.
# Works for BOTH the CPU embedder (qwen-embed, :8086) and the on-demand GPU one (qwen-embed-gpu, :8088):
#   CPU:  ./gates-embed.sh                                   (defaults below)
#   GPU:  M=qwen-embed-gpu ./gates-embed.sh http://127.0.0.1:8088
set -uo pipefail
B="${1:-http://127.0.0.1:8086}"
M="${M:-qwen-embed}"      # model id sent in the body; override to qwen-embed-gpu to gate the GPU sibling
PY=python3
pass=0; fail=0
ok(){ echo "  PASS: $1"; pass=$((pass+1)); }
no(){ echo "  FAIL: $1"; fail=$((fail+1)); }

echo "########## $M (Qwen3-Embedding-0.6B) gate @ $B ##########"
r=$(curl -s -m 120 "$B/v1/embeddings" -H 'Content-Type: application/json' \
     -d '{"model":"'"$M"'","input":["how do I sort a list in python","python list sorting tutorial","the weather in paris today"]}')
echo "$r" | $PY -c '
import sys,json,math
d=json.load(sys.stdin)
data=d.get("data",[])
if len(data)!=3: print("BAD n="+str(len(data))); sys.exit(1)
vs=[e["embedding"] for e in data]
dim=len(vs[0]); norm=math.sqrt(sum(x*x for x in vs[0]))
def cos(a,b): return sum(x*y for x,y in zip(a,b))
near=cos(vs[0],vs[1]); far=cos(vs[0],vs[2])
print(f"dim={dim} norm={norm:.4f} cos(near)={near:.4f} cos(far)={far:.4f}")
ok = dim==1024 and abs(norm-1.0)<0.01 and near>far
sys.exit(0 if ok else 2)
' && ok "1024-dim, normalized, semantic ordering (near>far)" || no "embedding shape/semantics wrong"

echo "########## $M gate: $pass PASS / $fail FAIL ##########"
[ "$fail" -eq 0 ]
