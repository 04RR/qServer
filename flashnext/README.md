# Flash-Next — in-repo launcher + gate

Canonical, version-controlled home for the `qwen-38-flash-next` / `qwen-38-flash-next-batch`
launcher and correctness gate, tracked in the qServer repo so a fresh checkout is self-contained.
`swap/config.yaml{,.q6km,.plain}` point both entries at `flashnext/run-38-flash-next-main.sh`, and
`regress.sh` maps both ids to `flashnext/gates38next.sh`.

## Files
- **`run-38-flash-next-main.sh`** — the launcher. Round-2 config (MTP off, f16 KV, `-ub 3072`, `-t 16`,
  `-ctxcp 8`, NCM=31, TS=40,8). All knobs are env-overridable; the swap config injects them per entry.
- **`gates38next.sh`** — single-server correctness gate with the **EXPECT_MTP** contract. `regress.sh`
  runs it with `EXPECT_MTP=0`; it asserts the server's `/slots` speculative state matches, SKIPs the
  draft gates (#5/#9) only on confirmed MTP-off, and FAILs on a mismatch or unreachable `/slots`.
  `SELFTEST=1 bash gates38next.sh` self-checks the decision logic (8 cases, no server).
- **`needle-next.py`** — deep-context (100K) needle, a dependency of gate #10.

## What lives OUTSIDE this repo (large; built/fetched separately)
- **Binary:** the launcher's `BIN` defaults to the MAINLINE llama.cpp build at
  `~/ai/flashnext-main/llama.cpp/build/bin/llama-server` (build 11736, commit 33d6135cb).
  Rebuild with `~/ai/flashnext-main/build-main.sh` (CUDA 12.8; arches 89;120).
- **Weights:** GGUFs (UD-IQ4_XS shards + `mmproj-F16.gguf`) at `~/ai/flashnext/models`.

## Rollback
The old MTP-on fork (`~/ai/flashnext`, its own `llama.cpp` build 10802, launcher
`run-38-flash-next.sh`) is kept untouched as the no-rebuild rollback target. See the
"Flash-Next rollback" section in `../USAGE.md`; on the rolled-back (MTP-on) config run the gate
with `EXPECT_MTP=1`.
