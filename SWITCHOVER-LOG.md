# Flash-Next round-2 switchover log — 2026-10-20

Ship of `qwen-38-flash-next` (+ `-batch`) to the round-2 config: **MTP off**, mainline llama.cpp
build **11736** (commit 33d6135cb), f16 KV (main) / q8_0 KV (batch), `-ub 3072` (main) / `2048` (batch),
`-t 16`, `-ctxcp 8`, NCM=31/TS=40,8 (main) · NCM=28/TS=37,11 (batch). Launcher + gate live **in-repo**
at `flashnext/`. Source of truth: `~/ai/flashnext-main/RESULTS-round2.md`.

## Before → after (same harness, bench2.py)
Old fork (MTP n2, shared head) → round-2. The `+%` column is the 2e `-t 24` figure from RESULTS; the
**shipped `-t 16`** spot-check numbers (measured today through the live stack) are in the last column.

| metric | old fork | round-2 (2e, t24) | shipped t16 (measured 2026-10-20) |
|---|---|---|---|
| code decode (t/s) | 14.84 | 19.41 (+31%) | **16.62** (3 seeds, 2048-tok) |
| decode @100K (t/s) | 11.78 | 18.45 (+57%) | ~18.0 (gate needle) / ~16.6 sustained |
| P-8K prefill (t/s) | 209 | 590 | **672** |
| P-100K prefill (t/s) | 201 | 821 | ~820 (gate needle) |

Decode is CPU-expert-bound and ≈flat with context (MTP off); the shipped `-t 16` trades ~2 t/s vs the
isolated `-t 24` to leave the persistent CPU embedder its 8 cores (see the thread-contention finding in
RESULTS Arm 5). Prefill is thread-independent.

## In-repo launcher confirmed live
- Resident llama-server's parent = the post-restart llama-swap (PID 1194291) running
  `-config /home/rohit/ai/qwen36/swap/config.yaml`, whose flash-next `cmd` is
  `/home/rohit/ai/qwen36/flashnext/run-38-flash-next-main.sh`.
- `GET /running` reports `cmd: env ... /home/rohit/ai/qwen36/flashnext/run-38-flash-next-main.sh`.
- (The llama-server's own `/proc/<pid>/cmdline` shows the binary, because the launcher `exec`s it —
  the launcher path can't appear there; the parent + config + `/running` are the authoritative proof.)

## Gates (mainline 11736, committed gate md5 `b63a42d8`, both entries)
- **main** `EXPECT_MTP=0`:
  - `SKIP_DEEP=0`: **15 PASS / 0 FAIL / 2 SKIP** (deep needle ran: pp≈820, tg≈18 @100K, sentinel recalled).
  - `SKIP_DEEP=1`: **14 PASS / 0 FAIL / 2 SKIP**. SKIPs = #5, #9 (MTP off, `/slots`-confirmed).
  - `reasoning_effort` now **PASSES** (low 654 vs high 880, 25% diff) — mainline honors it; the old fork
    (10802) did not, so the round-1 "pre-existing FAIL" is resolved by the build, not suppressed.
- **batch** `EXPECT_MTP=0 EXPECT_VISION=0 PROFILE=batch SKIP_DEEP=1`: **13 PASS / 0 FAIL / 3 SKIP**.
  SKIPs = #5 (MTP off), #8/#9 (vision off, text-only). #12 PASSES (654 vs 465, 28% diff — honored even
  though high renders *shorter* under q8_0 KV; the check now tests effect-present, not strict low<high).
- Gate self-test (`SELFTEST=1`): **8 PASS / 0 FAIL** (MTP decision logic, no server).

### Gate contract added this ship (so one gate covers both entries)
- `EXPECT_VISION` (from `VISION`; off→0): text-only batch → #8/#9 SKIP. Declared, not asserted (benign).
- `PROFILE` (main|batch): #2 residency VRAM bands. Measured as-shipped: **main** GPU0 23220 / GPU1 14196;
  **batch** GPU0 **18906** (5.6 GiB free) / GPU1 **15568** (~740 free). The old config comment's
  "20724/15094" was stale — corrected in all three config files.
- #12 reworked to effect-present (low≠high ≥5%); still FAILs an ignored kwarg (identical→0%).

## Production spot-check (through the live stack)
- **main**: code decode 16.62 t/s, P-8K 672 t/s, VRAM [23220, 14196] — on-target for `-t 16`.
- **batch** (`bench_conc.py`, direct :8089): conc-1 12.8 t/s; conc-8 **23.6 t/s warm** (16.4 cold — the
  first run paid expert-mmap page-ins), 0 errors. vs round-2 `-t 24` ~32 (the `-t 16` choice costs ~26%).

## regress.sh (router-integration suite + [0] mapping)
**PASS=22 / FAIL=0 / SKIP=1** (soak skipped). `[0]` maps all **7** registered models to live gates,
including the batch entry's `EXPECT_MTP=0 EXPECT_VISION=0 PROFILE=batch flashnext/gates38next.sh`. The
27B is unchanged (coherence, digit-canary, thinking tags, tool calls, system-first contract, prefix-cache
reuse 10950→173 ms, 100K recall, alias dispatch, #13 router tool-call guard all PASS).

## In-flight swap behaviour (probe, report only)
Long stream on `qwen-35b-batch`, `qwen36` requested 6 s in. The **stream completed uninterrupted**
(3000 chunks, 27.8 s, clean `[DONE]`); the competing `qwen36` request **waited 42.5 s** (drained the
active stream within the 30 s `unloadTimeout`, then swapped in the 27B). ⇒ graceful drain, not kill.

## Unload
`curl :8000/unload` → `{"unloaded":true}`, `/running` empty, **GPU0 0 MiB / GPU1 0 MiB**.

## Open levers (not blockers; for a round 3 or operator choice)
1. **Batch threads.** Shipped `-t 16` (conservative: the CPU embedder is persistent). Standalone bulk
   indexing with the embedder idle would gain ~+36% conc-8 at `-t 24` (23.6 → ~32). One-line change
   (`T=16`→`T=24` in the batch `cmd`) + `systemctl restart qwen-swap`.
2. **Batch GPU0 headroom.** Batch leaves 5.6 GiB free on GPU0 while GPU1 is tight (~740 MiB); the split
   (NCM=28/TS=37,11) could shift more onto GPU0 to pull experts off CPU. Round-3 tuning.
3. **flashnext-main copies.** The pre-move `~/ai/flashnext-main/{run-38-flash-next-main.sh,
   gates38next.mtp-skip.sh}` are kept (not deleted, per standing instruction); the in-repo `flashnext/`
   copies are canonical. ~8 GB reclaimable in `~/ai/flashnext*` if ever desired (see RESULTS disk report).

## Rollback (config-only, no rebuild)
Revert both `cmd:` blocks in all three `swap/config.yaml*` to the old-fork launcher and restart:
```
# qwen-38-flash-next cmd:
env NO_WARMUP=1 NCM=31 TS=40,8 MG=0 CTX=131072 NMAX=2 VISION=on PORT=8085
/home/rohit/ai/flashnext/run-38-flash-next.sh
sudo systemctl restart qwen-swap qwen-router
```
The old fork tree (`~/ai/flashnext`, its build 10802, shared MTP head) is untouched → instant rollback.
On the rolled-back (MTP-on) config, run the gate with **`EXPECT_MTP=1`** (the [M] check then requires the
server to actually be drafting). Full recipe in `USAGE.md` → "Flash-Next rollback".
