# Qwen3.8-Flash-Next — mainline-llama.cpp upgrade + max-t/s tuning

**Date:** 2026-10-19 · **Box:** RTX 4090 (GPU0, 24 GiB) + RTX 5060 Ti (GPU1, 16 GiB, PCIe gen4 x4), i9-13900K, 48 GiB WSL cap.
**Quant held fixed:** UD-IQ4_XS (3 shards) + mmproj-F16. **Production NOT switched** (read-only brief).

## Build (Step 1)
- **mainline ggml-org/llama.cpp @ `33d6135cb`** = `llama-server 0.7.0-dev build 11736`, ggml 0.27.0 (the MoE-cache SLRU #30637 tip; includes qwen4exp MTP #29761, sparse FA, radix top-k, `--moe-cache-mib`, halved indexer buffer). Isolated at `~/ai/flashnext-main/`. CUDA 12.8, arch `89;120`, static, Release.
- **smpbo/#23385 patch NOT applied** — CUDA init showed no implausible `sharedMemPerBlockOptin` on driver 610.88; not needed.
- **`test-backend-ops` (correctness, CPU-vs-CUDA):** 5060 Ti/sm_120 **5468/5468 + 2488/2488 pass**, 4090 **7956/7956 pass** across MUL_MAT, MUL_MAT_ID, FLASH_ATTN_EXT, GATED_DELTA_NET, TOP_K. No sm_120 failure. ✅
- **MTP head:** mainline can't borrow tensors from the `shared-` head → downloaded the **self-contained `mtp-Qwen3.8-Flash-Next-Q8_0.gguf`** (4.14 GB).

## Headline results
1. **The upgrade itself is a free win** at identical flags (arm B vs A): decode-at-depth **+17 %** (D-100K 19.5→22.9), **MTP acceptance 0.65→0.86**, **GPU0 VRAM −1.2 GiB** (halved indexer buffer), load 51 s→20 s, faults halved. Correctness gate 15/16 (the 1 fail is the pre-existing `reasoning_effort`-per-request item, not a regression).
2. **Prefill was PCIe-bound — confirmed.** Raising the prefill ubatch (`-ub`) multiplies prompt-processing: **P-100K 210 → 620 (UB=2048) → 959 t/s (UB=4096)** = up to **4.4×**. This is the big lever for the **batch lane** and any prompt-heavy work.
3. **MoE GPU-expert cache (`--moe-cache-mib`) is NOT a win here** and *hurts* at concurrency (top-10-of-512 routing defeats it). Pinned host memory and probabilistic/NMAX=3 drafting: no measurable gain. Details below.

## All arms (median of 3 for D-short; single pass for P-100K/D-100K)

| arm | config | D-short | D-100K | P-8K | P-100K | MTP acc | VRAM G0/G1 | notes |
|---|---|---|---|---|---|---|---|---|
| **A** | old fork, MTP n2, UB512, f16, NCM31 TS40,8 | 11.5 | 19.47 | 221.8 | 210.0 | 0.646 | 21876/15288 | baseline |
| **B** | **new build**, same flags | 12.5 | **22.87** | 228.9 | 217.5 | **0.858** | **20640**/15302 | **upgrade win**; gate 15P/1F; load 20s |
| Cs2 | new, **MTP off, UB2048**, f16 | 16.9 | 18.3 | 631.4 | **620.4** | — | 22178/12978 | prefill ~3× |
| Cs4q | new, **MTP off, UB4096, q8_0** | 16.7 | 15.5 | 875.4 | **959.2** | — | 22652/15114 | prefill ~4.4×; G1 tight |
| C1 / C1q | MTP on + UB2048 (f16 / q8_0) | — | — | — | — | — | OOM | draft buf 3291 MiB → GPU1 OOM |
| Cs4(f16) | MTP off, UB4096, f16 | — | — | — | — | — | OOM | GPU0 +862 MiB over |
| Dref | new, MTP off, UB512, NCM31, no cache | 17.1 | — | 246.8 | — | — | 20490/11226 | **623 faults** (page cache warm) |
| Dmc37 | NCM37 + **cache 4096**, MTP off | 16.7 | — | 199.4 | — | — | 17758/11226 | 49 393 faults; prefill worse; cache no help |
| Dmc31 | NCM31 + cache 4096 | — | — | — | — | — | OOM | cache doesn't fit at prod NCM (f16) |
| **MainUB2** | **MTP on, UB1024, q8_0, TS42,6** | 11.9 | — | **372.5** | — | 0.873 | 22066/13594 | **+63 % prefill, MTP intact** (needs q8 gate) |
| E_pin | B + `GGML_CUDA_REGISTER_HOST=1` | 11.8 | — | 228.5 | — | 0.837 | 20640/15302 | no effect |
| F_nmax3 | B + NMAX=3 + probabilistic | 10.7 | — | 216.2 | — | 0.833 | 20640/15302 | no help; 1 empty run |

### Batch lane (concurrent aggregate decode t/s, short-code)
| arm | config | conc=1 | conc=4 | conc=8 | P-8K |
|---|---|---|---|---|---|
| **G0** | old fork, NCM28 TS37,11, UBdef, f16 | 15.8 | 29.1 | 26.6 | 236.9 |
| **G1** | **new, UB2048, q8_0** (MTP/vision off) | 16.8 | 25.5 | **27.5** | **668.5** |
| G2 | G1 + **cache 2048** | 17.9 | 19.4 | 19.5 | 661 | cache **hurts** concurrency |

## Interpretation
- **Decode is memory-bandwidth-bound** (reading ~35 GiB of CPU-side experts). Single-stream, the working set is served warm from the Linux page cache (Dref: only 623 major faults), so decode sits ~12–17 t/s and neither ubatch, pinned host, nor the MoE cache moves it. MTP (n=2, greedy, the self-contained head) is the one decode lever that helps, and the new build **raises its acceptance to 0.86** → that's why B's D-100K beats A.
- **Prefill is PCIe-bound** (each ubatch copies CPU experts to the 4090). `-ub` directly amortizes that copy → 3–4.4× once the ubatch is 2048–4096. This is almost pure upside for the batch/indexing lane (MTP off → GPU1 has room). On the interactive entry it's capped by VRAM: MTP's draft wants ~3.3 GiB on the 16 GB card at UB=2048, so the best that fits with MTP+vision is **UB=1024 + q8_0 + TS=42,6 → +63 % prefill** (MainUB2).
- **`--moe-cache-mib` is the wrong tool for this model.** With top-10-of-512 routing, consecutive tokens/concurrent sequences touch largely different experts, so the GPU cache gets few hits; at NP=4 it added eviction overhead and dropped aggregate from 27.5→19.5. This SHA also prints no hit-rate telemetry, so judgement is purely by t/s. Recommend leaving it at 0.

## Recommended configs (NOT applied)

### Winning launcher
`~/ai/flashnext-main/run-38-flash-next-main.sh` (copy of the live launcher; new BIN + self-contained head default + knobs `UB`, `SPEC_UB`, `MOE_CACHE`, `DRAFT_SAMPLING`). Same launcher serves both entries via env.

### MAIN entry `qwen-38-flash-next` — safe win (ship this)
New build, UB left at default, **f16 KV**, MTP on. This is arm B: measured gains with zero correctness risk (f16 100K needle passed in the gate).
```
env BIN=/home/rohit/ai/flashnext-main/llama.cpp/build/bin/llama-server \
    NO_WARMUP=1 NCM=31 TS=40,8 MG=0 CTX=131072 NMAX=2 VISION=on PORT=8085 \
    /home/rohit/ai/flashnext-main/run-38-flash-next-main.sh
```
*(Optional prefill-boost variant — +63 % prefill — `UB=1024 KVK=q8_0 KVV=q8_0 TS=42,6`: ship only after a q8_0 100K-needle gate, since the main entry is retrieval-sensitive.)*

### BATCH entry `qwen-38-flash-next-batch` — ship the ubatch win
```
env BIN=/home/rohit/ai/flashnext-main/llama.cpp/build/bin/llama-server \
    NP=4 SPEC_OFF=1 VISION=off NO_WARMUP=1 NCM=28 TS=37,11 MG=0 CTX=131072 \
    UB=2048 KVK=q8_0 KVV=q8_0 PORT=8089 \
    /home/rohit/ai/flashnext-main/run-38-flash-next-main.sh
```
Prefill 237→668 t/s (+182 %), concurrent decode ≈ unchanged. (UB=4096 pushes prefill to ~960 but GPU1 is near-full at NP=4; validate residency before using it. No `--moe-cache-mib`.)

### Proposed `swap/config.yaml` diff (both entries) — apply yourself
```diff
   "qwen-38-flash-next":
     cmd: >-
-      env NO_WARMUP=1 NCM=31 TS=40,8 MG=0 CTX=131072 NMAX=2 VISION=on PORT=8085
-      /home/rohit/ai/flashnext/run-38-flash-next.sh
+      env BIN=/home/rohit/ai/flashnext-main/llama.cpp/build/bin/llama-server
+      NO_WARMUP=1 NCM=31 TS=40,8 MG=0 CTX=131072 NMAX=2 VISION=on PORT=8085
+      /home/rohit/ai/flashnext-main/run-38-flash-next-main.sh

   "qwen-38-flash-next-batch":
     cmd: >-
-      env NP=4 SPEC_OFF=1 VISION=off NO_WARMUP=1 NCM=28 TS=37,11 MG=0 CTX=131072 PORT=8089
-      /home/rohit/ai/flashnext/run-38-flash-next.sh
+      env BIN=/home/rohit/ai/flashnext-main/llama.cpp/build/bin/llama-server
+      NP=4 SPEC_OFF=1 VISION=off NO_WARMUP=1 NCM=28 TS=37,11 MG=0 CTX=131072
+      UB=2048 KVK=q8_0 KVV=q8_0 PORT=8089
+      /home/rohit/ai/flashnext-main/run-38-flash-next-main.sh
```
Apply the same change to `swap/config.yaml.q6km` and `swap/config.yaml.plain` (keep all three in sync).

## Switch-over (you run these)
1. Edit the three config files as above.
2. `sudo systemctl restart qwen-swap qwen-router`
3. Pre-warm + gate the main entry:
   `curl -s localhost:8000/load/flash-next` then `B=http://127.0.0.1:8085 MODEL_ID=flash-next bash ~/ai/flashnext/gates38next.sh` → expect 15 PASS / 1 FAIL (the known reasoning_effort item).
4. `bash ~/ai/qwen36/regress.sh` → `[0]` must stay 7↔7 (every model mapped).
5. `curl -s localhost:8000/unload` ; then `wsl --shutdown`-reopen is optional — if done, `bash ~/ai/qwen36/systemd/verify-boot.sh`.

## Rollback
Revert the config `cmd` blocks to the old fork launcher + shared head and restart:
```
env NO_WARMUP=1 NCM=31 TS=40,8 MG=0 CTX=131072 NMAX=2 VISION=on PORT=8085
/home/rohit/ai/flashnext/run-38-flash-next.sh
```
`sudo systemctl restart qwen-swap qwen-router`. The old fork tree (`~/ai/flashnext`), its binary, and the shared head are untouched, so rollback is config-only (no rebuild).

## Could not test / caveats
- **q8_0 KV on the main entry:** MainUB2 (+63 % prefill) and the batch lane use q8_0; I ran the full 100K needle only on **f16** (gate B). Run a q8_0 100K-needle before shipping q8_0 on the retrieval-sensitive main entry. (M4 history says q8_0 is clean on this arch via #27941; not re-verified at 100K here.)
- **UB=4096 at NP=4** (batch) not residency-verified — GPU1 is near-full; validate before use.
- **MoE-cache hit rate:** this SHA emits no hit-rate line, so the cache was judged by t/s + pgmajfault only.
- Deep (262K) curves and a perplexity slice were out of scope (no corpus in repo; 262K needs an NCM/TS re-fit).
- Concurrent bench used a short-code prompt (decode-bound); the batch lane's real prompt-heavy workload benefits more from the prefill win than the concurrent-decode numbers suggest.
