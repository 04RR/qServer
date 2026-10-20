# Flash-Next — in-repo launcher + gate

Canonical, version-controlled home for the `qwen-38-flash-next` / `qwen-38-flash-next-batch`
launcher and correctness gate, tracked in the qServer repo so a fresh checkout is self-contained.
`swap/config.yaml{,.q6km,.plain}` point both entries at `flashnext/run-38-flash-next-main.sh`, and
`regress.sh` maps both ids to `flashnext/gates38next.sh`.

## Files
- **`run-38-flash-next-main.sh`** — the launcher. Round-2 config (MTP off, f16 KV, `-ub 3072`, `-t 16`,
  `-ctxcp 8`, NCM=31, TS=40,8). All knobs are env-overridable; the swap config injects them per entry.
- **`gates38next.sh`** — single-server correctness gate with three declared contracts so one gate covers
  both config entries:
  - **`EXPECT_MTP`** (from `SPEC_OFF`): asserts the server's `/slots` speculative state matches; SKIPs the
    draft gates (#5/#9) only on confirmed MTP-off; FAILs on a mismatch or unreachable `/slots`.
  - **`EXPECT_VISION`** (from `VISION`; `off`→`0`): the text-only batch entry has no mmproj, so #8 (vision)
    and #9 (coexistence) SKIP. Declared, not asserted (a wrong declaration only skips a would-pass check).
  - **`PROFILE`** (`main`|`batch`): selects the #2 residency VRAM bands (the entries resident-size differently).
  `regress.sh` runs it `EXPECT_MTP=0` for the main entry and `EXPECT_MTP=0 EXPECT_VISION=0 PROFILE=batch` for
  the batch entry (also with `KVK=q8_0`). As-run on mainline 11736 (`SKIP_DEEP=1`): main **15/0/2**; batch
  **13 PASS / 0 FAIL / 3 SKIP / 1 KNOWN-FAIL** — the KNOWN-FAIL is `#12b` (`reasoning_effort` *plumbed* (12a)
  but not *directional* under q8_0 KV). On q8_0, `#12b` is a KNOWN-FAIL (does NOT fail the gate) while
  high/low ≥ 820 permille (recorded baseline 733→639 = 871); it reverts to a hard FAIL if the numbers degrade
  below that, or on any non-q8_0 KV (the exit criterion: an f16-KV batch re-test). Any OTHER failing check
  still exits non-zero. `SELFTEST=1 bash gates38next.sh` self-checks the MTP decision + `classify_12b` logic
  (16 cases, no server).
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
