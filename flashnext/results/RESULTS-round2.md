# Qwen3.8-Flash-Next — round 2: settle MTP, lock fastest main + batch

**Date:** 2026-10-20 · **Box:** RTX 4090 (GPU0, 24 GiB) + RTX 5060 Ti (GPU1, 16 GiB, PCIe4 x4), i9-13900K (8P+16E = 24 physical cores / 32 logical), 48 GiB WSL cap.
**Build:** mainline `ggml-org/llama.cpp @ 33d6135cb` (`llama-server build 11736`), CUDA 12.8, arch 89;120, static. Isolated at `~/ai/flashnext-main/`. **Quant fixed:** UD-IQ4_XS + mmproj-F16. **Production NOT switched.**

All numbers are through the round-2 harness: production chat template via `/apply-template`→`/completion`, **thinking ON**, temp 1.0 / top_p 0.95 / top_k 20, one discarded ~1K warm-up, 3 seeds/cell (median [min–max]), natural EOS (no `ignore_eos`), `n_predict` ceiling 1024 (every reported run generated the full 1024; none < 400 → none dropped). Decode cells: **code**-short, **prose**-short, **report-100K** (100K haystack + 1,500-word summary). Prefill cells: **P-8K**, **P-100K** (`cache_prompt:false`).

> **Why round 1's decode numbers were wrong:** round 1 measured decode on a bare `def merge(` completion with no chat template and thinking off, which inflated MTP acceptance to ~0.86 and understated t/s. Under the production template + thinking, realistic acceptance is **0.45–0.63**, and decode is 15–20 t/s. Round 2 supersedes round 1's D-short/D-100K.

## Context-checkpoint reuse — WORKS on qwen4exp ✅
Launched every arm with `-ctxcp 8` (`CKPT=8`). In report-100K, seed 1 prefilled the 100K haystack (`prompt_n = 100014`); **seeds 2 & 3 reused it (`prompt_n = 4`)** — so the 3 seeds share ONE 100K prefill per launch. RSS cost of the checkpoint snapshot: **~1.1 GiB (MTP off) / ~1.6 GiB (MTP on) / ~3.1 GiB (old fork)** VmRSS delta across the 100K prefill (includes the 100K KV, not only the checkpoint). Identical-prompt reuse via `cache_prompt` is what carries it; checkpoints additionally survive small suffix changes. No per-run re-prefill needed.

---

## Headline

1. **MTP OFF wins the main entry.** Under the real harness, MTP n2/n1/off geomeans are 16.3 / 17.0 / 16.8 — the best is within 5 % of off, so (per the decision rule) **drop MTP**: it frees ~4 GiB on GPU1 and unlocks a big prefill ubatch. Confirms the CPU-compute-bound hypothesis (acceptance only ~0.45–0.63 ⇒ verify's tripled CPU-expert work doesn't pay).
2. **`-t 24` is a free +10 % decode in isolation** (24 physical cores, no hyperthread oversubscription) — **but it collapses to 5.2 t/s under concurrent CPU embedding**, while `-t 16` holds at 16.7. The persistent CPU embedder makes **`-t 16` the right production default** (16 + the embedder's 8 = 24).
3. **Main winner = MTP off · f16 KV · `-ub 3072` · NCM31 · TS40,8 · vision on.** vs today's production (old fork, MTP n2) measured on the same harness: **decode +31–57 %, P-100K prefill 4.1×**, no q8_0 dependency.
4. **q8_0 KV is clean** (5/5 needle recall at 100K *and* 125K, equal to f16) — available as a max-prefill variant (P-100K ~1017) and used for the batch lane.
5. **Batch lane +18–23 % at conc 8, prefill +165 %** (new build, q8_0, `-ub 2048`, `-t 24`, NP4). `-ub 4096` OOMs GPU1 at NP4 — stay at 2048.

---

## Arm 1 — MTP A/B (new build, f16 KV, `-ub` default, NCM31 TS40,8, vision on, default threads)
Run order on2→off→on1→on2. Decision metric = geomean of the 3 decode cells.

| variant | code | prose | report-100K | **geomean** | acc | VRAM peak G0/G1 (free) |
|---|---|---|---|---|---|---|
| MTP n2 (pass a) | 16.14 [15.4–16.2] | 17.13 [16.1–18.6] | 15.53 | **16.25** | 0.45 | 21410/15416 (3154/895) |
| MTP n2 (pass b) | 15.89 | 17.85 | 15.26 | **16.30** | 0.45 | (3154/895) |
| MTP n1 | 17.90 [16.9–18.7] | 17.73 | 15.54 | **17.02** | 0.63 | (3252/911) |
| **MTP off** | 17.20 | 17.20 | 16.10 | **16.83** | — | 21214/11284 **(3350/5027)** |

- **Reproducible:** on2a vs on2b geomean 16.25 vs 16.30 = **0.3 %** (no noise flag).
- **Decision: MTP OFF.** Best variant (n1, 17.02) is within 5 % of off (16.83) → choose off. MTP off also drops GPU1 from ~15.3 GiB to ~11.2 GiB (**+4 GiB free**), which is what makes a big prefill ubatch fit.
- **Against the hypothesis:** decode is CPU-compute-bound (31/48 layers' experts on CPU, ~0.7 GiB read/token). Each verified draft token costs its own 10 CPU experts; at 0.45–0.63 acceptance the extra verified tokens don't cover the ~3× CPU work, so off ≳ on. MTP's only decode lever is acceptance, and it isn't high enough here.

## Arm 2 — main-entry configs (MTP off unless noted)

| id | config | code | prose | report-100K | P-8K | P-100K | peak G0/G1 (free) | q8 gate? |
|---|---|---|---|---|---|---|---|---|
| 2a | q8_0, `-ub 4096`, Tdef | 16.89 | 16.89 | 13.78 | **968** | **1017** | 22806/15270 (1758/1041) | needs gate |
| 2b | **MTP n1** q8_0 `-ub 2048` TS42,6 | 16.63 | 16.97 | 12.40 | 619 | 569 | 23332/14166 (1232/2145) | needs gate |
| 2c | f16, `-ub 2048`, Tdef | 17.80 | 17.87 | 16.62 | 682 | 667 | 22276/13078 (2288/3233) | no |
| 2d | f16, `-ub 4096`, TS37,11 | — | — | — | — | — | **OOM GPU1** (4.7 GiB ubatch buf) | — |
| **2e** | **f16, `-ub 3072`, `-t 24`** | **19.41** | **19.56** | **18.45** | 590 | **821** | 23290/14276 **(1274/2035)** | **no** |
| 2e′ | f16, `-ub 2048`, `-t 24` | 19.60 | 19.63 | 18.43 | 654 | 624 | 22276/13078 (2288/3233) | no |

- **`-ub 3072` is the sweet spot for f16:** same decode as `-ub 2048`, but **P-100K 821 vs 624 (+32 %)**, and it still clears ≥1 GiB on both GPUs. `-ub 4096` on f16 is a knife-edge (OOMs GPU0 at TS40,8, GPU1 at TS37,11 — the f16 ubatch buffer is ~4.7 GiB).
- **2b proved SPEC_UB is a dead end for the main entry:** `--spec-draft-n-ubatch 16` **is** applied (verified in the launcher echo + cmdline), but the 3291 MiB that blocks big `-ub` is the **draft model's weight+KV buffer on GPU1**, not a ubatch compute buffer — SPEC_UB can't shrink it. MTP n1 at UB2048 only fits by shifting TS to 42,6, and even then prefill is just 569. MTP-on cannot reach the UB3072 prefill.
- **q8_0 (2a) vs f16 (2e):** q8_0 buys +24 % P-100K (1017 vs 821) but costs **−25 % decode-at-depth** (13.78 vs 18.45 — q8 KV dequant at 100K) and needs the retrieval gate. For an interactive entry, 2e's decode wins; **2a is offered as a max-prefill variant.**

**Main winner: 2e** → `MTP off · f16 · -ub 3072 · NCM31 · TS40,8 · vision on`, threads per arm 4.

## Arm 3 — q8_0 retrieval gate (blocks shipping any q8_0 config)
5 fresh sentinels (not in repo history) at depths ~8/29/50/71/92 %, calibrated to the target:

| KV | 100K recall | 125K recall |
|---|---|---|
| f16 (reference) | **5/5** | **5/5** |
| **q8_0 (2a config)** | **5/5** | **5/5** |

**PASS** — q8_0 recall equals f16 and is ≥4/5 at both depths. q8_0 KV is safe for the retrieval-sensitive entry (consistent with the M4 #27941 history). `gates38next.sh SKIP_DEEP=0` on the q8_0 MTP-off config: **13 PASS / 3 FAIL**, where **2 fails are artifacts of running MTP off** (`#5 MTP dead`, `#9 coexistence draft_n=0` — the script assumes MTP on) and the 3rd is the **pre-existing** `reasoning_effort`-per-request item (unchanged from round 1). No genuine regression. → see the gate patch below.

### Proposed gates38next.sh patch (NOT applied; copy at `~/ai/flashnext-main/gates38next.mtp-skip.sh`)
**Contract (reworked 2026-10-20):** the caller **declares** `EXPECT_MTP` (defaulted from the entry's `SPEC_OFF`: `SPEC_OFF=1`→`EXPECT_MTP=0`, else `1`). The gate reads the server's **actual** speculative state from `/slots` (this build's `/props` omits it: `speculative=true` or `params["speculative.types"]≠"none"` ⇒ MTP on) and a new check **[M]** asserts actual == `EXPECT_MTP`:
- `/slots` unreachable or unparseable → **FAIL** (never skip on an unverified state).
- actual ≠ `EXPECT_MTP` → **FAIL** (e.g. you meant MTP on but it isn't drafting; or meant off but it's on).
- only `EXPECT_MTP=0` **and** server-confirmed `speculative=off` → set `MTP_SKIP=1`, so `#5`/`#9`'s draft parts **SKIP** (vision is already proven at `#8`, so `#9` skips its draft-coexistence assertion but still FAILs if vision actually broke).

This is intent-checked rather than intent-inferred: a *broken* MTP on an entry you declared MTP-on still FAILs instead of being silently skipped.

**Self-test:** `SELFTEST=1 bash gates38next.mtp-skip.sh` runs the decision (no server) over expected on/off × actual on/off, `/slots`-down for both expects, and a malformed body — **8 PASS / 0 FAIL**, each cross-checking the return code against the verdict:

```
   st-PASS: expect-on  actual-on        -> RUN  :: MTP on confirmed
   st-PASS: expect-on  actual-on(types) -> RUN  :: on via speculative.types
   st-PASS: expect-on  actual-off       -> FAIL :: state mismatch
   st-PASS: expect-off actual-on        -> FAIL :: state mismatch
   st-PASS: expect-off actual-off       -> SKIP :: MTP off confirmed
   st-PASS: expect-off slots-down       -> FAIL :: unverifiable
   st-PASS: expect-on  slots-down       -> FAIL :: unverifiable
   st-PASS: expect-off malformed        -> FAIL :: unverifiable
```

```diff
@@ pass=0; fail=0
 ok(){ echo "  PASS: $1"; pass=$((pass+1)); }
 no(){ echo "  FAIL: $1"; fail=$((fail+1)); }
+# EXPECT_MTP declared by caller (from SPEC_OFF). We assert the server's ACTUAL /slots state == EXPECT_MTP.
+EXPECT_MTP="${EXPECT_MTP:-$([ "${SPEC_OFF:-0}" = "1" ] && echo 0 || echo 1)}"
+skip=0; sk(){ echo "  SKIP: $1"; skip=$((skip+1)); }
+classify_slots(){ local j="$1"; [ -n "$j" ] || { echo error; return; }
+  echo "$j" | jq -e 'type=="array" and length>0' >/dev/null 2>&1 || { echo error; return; }
+  if echo "$j" | jq -e '(.[0].speculative==true) or ((.[0].params["speculative.types"] // "none")!="none")' \
+       >/dev/null 2>&1; then echo on; else echo off; fi; }
+decide(){ local expect="$1" j="$2" act actb; act=$(classify_slots "$j")
+  [ "$act" = error ] && { echo "FAIL /slots unreachable or unparseable"; return 1; }
+  actb=$([ "$act" = on ] && echo 1 || echo 0)
+  [ "$actb" != "$expect" ] && { echo "FAIL state mismatch expect=$expect server=$act"; return 1; }
+  [ "$expect" = 0 ] && { echo "SKIP MTP off confirmed"; return 0; }; echo "RUN MTP on confirmed"; return 0; }
+# (SELFTEST=1 block exercises decide() over all 4 cases + /slots-down + malformed, then exits)
@@ (after gate 4, before gate 5)
+echo "[M] speculative state vs declared EXPECT_MTP=$EXPECT_MTP"
+MTP_SKIP=0; _mdec=$(decide "$EXPECT_MTP" "$(curl -s -m5 "$B/slots" 2>/dev/null)")
+case "${_mdec%% *}" in RUN) ok "${_mdec#* }";; SKIP) ok "${_mdec#* }"; MTP_SKIP=1;; FAIL) no "${_mdec#* }";; esac
@@ # ---- 5. MTP liveness ----
 echo "[5] MTP live (draft_n>0, acceptance>0.4)"
+if [ "$MTP_SKIP" = 1 ]; then sk "MTP off by config (EXPECT_MTP=0, server confirmed) — draft liveness N/A"; else
 t=$(curl -s -m 900 "$B/completion" ... )
 ... && ok "MTP live, acc>0.4" || no "MTP dead or acc<=0.4"
+fi
@@ # ---- 9. coexistence ----
 v9=$(echo "$vall" | grep -qiE '\bred\b|crimson|scarlet' && echo yes || echo no)
+if [ "$MTP_SKIP" = 1 ]; then
+  [ "$v9" = yes ] && sk "coexistence draft_n N/A (MTP off); vision verified at #8" || no "coexistence broken: vision=$v9"
+else
 d9=$(curl ... '.timings.draft_n // 0')
 [ "$v9" = yes ] && [ "$d9" -gt 0 ] && ok "vision(red)+MTP(draft_n=$d9) coexist with PLE" || no "coexistence broken"
+fi
@@ summary
+echo "########## ... $pass PASS / $fail FAIL / $skip SKIP ##########"
```
With this patch, declaring `EXPECT_MTP=0` (or `SPEC_OFF=1`) on the MTP-off main entry reports **14 PASS / 1 FAIL / 2 SKIP** (the 1 FAIL = the known `reasoning_effort` item, unrelated to this upgrade). Declaring `EXPECT_MTP=1` against that same off server now **FAILs [M]** instead of passing — the state contract is enforced, not inferred.

### Ship-day finalization (2026-10-20) — gate is LIVE and in-repo
The patch above is now **applied and committed**, and the gate is the canonical production copy at **`~/ai/qwen36/flashnext/gates38next.sh`** (qServer repo). `swap/config.yaml{,.q6km,.plain}` point both flash-next entries at `flashnext/run-38-flash-next-main.sh`; `regress.sh` runs this gate for both ids. Three copies are kept byte-identical (in-repo canonical, `~/ai/flashnext/gates38next.sh` rollback, `~/ai/flashnext-main/gates38next.mtp-skip.sh` origin). Two more declared contracts were added so the one gate correctly covers **both** config entries:

- **`EXPECT_VISION`** (from the entry's `VISION` env; `VISION=off`→`0`). The batch entry is text-only (no mmproj), so `#8` (vision) and `#9` (coexistence) **SKIP** on `EXPECT_VISION=0` instead of FAILing. Unlike `EXPECT_MTP`, vision-off is *declared, not asserted* — a wrong declaration only skips a check that would have passed (benign), whereas a silent MTP-on misconfig is dangerous.
- **`PROFILE`** (main|batch; `regress.sh` passes `PROFILE=batch`) selects the `#2` residency bands, because the two entries resident-size differently. Measured as-shipped on mainline 11736: **main** GPU0 **23220** / GPU1 **14196** GiB (bands **21500–24200 / 13000–15500**, recentred on the as-shipped footprint 2026-10-20 pm); **batch** GPU0 **18906** / GPU1 **15568** GiB (bands 16500–20500 / 14200–16200). (The old config comment's "GPU0 20724/GPU1 15094" was stale — corrected in all three configs.)
- **`#12` reworked (2026-10-20 pm) to a two-part directional test** (replacing an intermediate "effect-present ≥5% either direction" rule, which was too lenient):
  - **12a (deterministic plumbing):** `/apply-template` must render DIFFERENT prompts for `reasoning_effort` low vs high — proves the kwarg reaches the chat template with no sampling noise. PASSes on both entries (same template).
  - **12b (behavioural direction):** mean `reasoning_content` length over **3 prompts** at temp 0 must be HIGHER for high than low. **main (f16) PASSes** (mean 754→1057, all 3 prompts longer). **batch (q8_0 KV) FAILs** (mean 733→639; per-prompt low→high 654→465, 741→642, 805→810 — high is *shorter* on 2 of 3). So under q8_0 KV the kwarg is *plumbed* (12a) but *not honored directionally* (12b). Reported, **not loosened** — the batch lane is non-interactive bulk scoring where reasoning length isn't the product concern, but it is a real behavioural difference vs f16.

**As-run counts (mainline 11736, 2026-10-20):**
- **main** `EXPECT_MTP=0`: **15 PASS / 0 FAIL / 2 SKIP** (`SKIP_DEEP=1`; `SKIP_DEEP=0` adds the deep needle → 16/0/2, last green pp≈820/tg≈18 @100K). SKIPs = `#5`, `#9` (MTP off).
- **batch** `EXPECT_MTP=0 EXPECT_VISION=0 PROFILE=batch KVK=q8_0`: **13 PASS / 0 FAIL / 3 SKIP / 1 KNOWN-FAIL** (`SKIP_DEEP=1`). The KNOWN-FAIL = `#12b` (q8_0 reasoning_effort not directional). On `KVK=q8_0` `#12b` is a **KNOWN-FAIL** (doesn't flip the exit code) while high/low ≥ 820 permille (recorded baseline 733→639 = 871); it reverts to a hard FAIL if the numbers degrade below 820, or on any non-q8_0 KV — **exit criterion: an f16-KV batch re-test**. Any OTHER failing check still exits non-zero. SKIPs = `#5` (MTP off), `#8`/`#9` (vision off).

## Arm 4 — CPU threads (main winner config; default was `-t -1`)

| `-t` | code (isolated) | P-8K |
|---|---|---|
| default (-1) | 17.80 | 681 |
| 8 | 16.87 | 677 |
| 12 | 16.64 | 644 |
| 16 | 17.88 | 595 |
| **24** | **19.63** | 656 |
| 32 | 5.48 (HT thrash) | 622 |

Isolated winner **`-t 24`** (+10 % decode, >3 % over all others; 32 oversubscribes the P-cores' logical siblings and collapses).

### Thread contention under concurrent CPU embedding (the decider)
Standalone `qwen-embed` (CPU, `-t 8`) on :8097 driven to a sustained **~7.6 cores** of embedding load, main entry (2e config) measured under it:

| `-t` | code **isolated** | code **under embed load** |
|---|---|---|
| **24** | 19.41 | **5.18** ⚠ collapse |
| **16** | 17.88 | **16.66** ✓ robust |

`-t 24` claims all 24 physical cores, so the always-resident embedder's 8 threads oversubscribe → context-switch thrash → decode craters. `-t 16` leaves exactly the embedder's 8 threads (16 + 8 = 24) and barely moves under load. **Recommendation: `-t 16` for the production main entry** (qwen-embed is persistent/concurrent by design). Use `-t 24` only for a dedicated deployment where no CPU embedding runs. (Alternative "fix": pin/cap the embedder, but the 16+8 partition is the simplest robust choice.) P-8K is GPU-bound and ~unchanged either way (~606 under load).

### Follow-up (2026-10-20): T=24 under *realistic light* query traffic
The ~7.6-core load above is a heavy bulk-indexer. To test the *light RAG query* case — one ~50-token `/v1/embeddings` every 2 s — I ran the 2e config at **`-t 24`** with `qwen-embed` (CPU, `-t 8`, :8097) serving exactly that: **205 queries over 410 s, 0 errors, 48-token queries, 58 ms each (≈0.5/s, ~3 % embedder duty cycle).**

| scenario | `-t 24` code decode |
|---|---|
| isolated (no embed) | 19.41 |
| **light query load (this test)** | **17.90** (17.15–18.13, 3/3 ran to 2048 ceiling) |
| heavy bulk embed (above) | 5.18 |

**Decision: keep `-t 16`.** Even trivial concurrent embedding knocks `-t 24` from 19.4 → **17.9**, which is **below the 18.5 t/s bar**, so the rule says don't adopt `-t 24` for the main entry. The drop isn't the bulk-load collapse — it's an ~8 % nick from 48 ms of CPU every 2 s — but it's enough to erase `-t 24`'s isolated edge, and it would only get worse when real traffic bursts above one-query-every-2 s. `-t 16` stays robust across all three scenarios (16.7–17.9) and is the production default; `-t 24` remains reserved for a dedicated no-embed deployment and the batch lane. (Harness: `t24_qload.sh` + `qload.py`; result `bench/t24q.json`.)

## Arm 5 — spend leftover 4090 VRAM (NCM, main winner + `-t 24`)

| NCM | code | prose | report-100K | peak G0/G1 (free) |
|---|---|---|---|---|
| **31** | 19.6 | 19.63 | 18.43 | 22276/13078 (2288/3233) |
| 30 | 19.18 | 18.82 | 18.74 | 23812/13078 **(752**/3233) |
| 29 | — | — | — | **OOM GPU0** (+862 MiB) |

**Keep NCM31.** Dropping NCM moves only one layer's experts CPU→GPU (~3 % of the CPU work) so decode doesn't improve, while GPU0 fills: NCM30 leaves only 752 MiB on GPU0 (< 1 GiB) and NCM29 OOMs. Not a useful lever here.

## Arm 6 — batch lane (interleaved G0/G1, aggregate decode t/s)

| id | config | conc 1 | conc 4 | conc 8 | P-8K |
|---|---|---|---|---|---|
| G0a | old fork, NCM28 TS37,11, UBdef, f16 | 9.4 | 27.7 | 25.9 | 241 |
| G0b | " (repeat) | 16.8 | 18.7 | 28.2 | — |
| **G1a** | **new, q8_0, `-ub 2048`, `-t 24`, NP4** | 19.3 | 26.9 | **31.9** | **637** |
| G1b | " (repeat) | 18.3 | 31.7 | 32.2 | — |

- **G1 vs G0: conc-8 ~32 vs ~27 (+18–23 %), P-8K 637 vs 241 (+165 %).** `-t 24` also lifts batch over round-1's default-thread G1 (27.5 → 32, +16 %), so **apply `-t 24` to the batch lane** (the batch lane is a dedicated throughput job; if it ever runs *concurrently* with CPU embedding, use `-t 16` there too — same contention rule).
- conc-4 is noisy run-to-run (18.7–31.7); conc-1 and conc-8 are stable. The bench fires short-code completions (decode-bound); the batch lane's real prompt-heavy indexing benefits more from the +165 % prefill than these decode aggregates show.
- **`-ub 4096` at NP4: OOMs GPU1** (needs 2295 MiB more at load). **Batch stays at `-ub 2048`.**

## Vision VRAM with a max-resolution image (main winner, 2e config)
Max-res 3072×2048 image (4134 vision tokens), peak VRAM polled during the request:

| scenario | peak G0 / G1 | free G0 / G1 |
|---|---|---|
| model loaded (no image) | 23220 / 14196 | 1344 / 2115 |
| max-res image alone | 23950 / 14278 | **614** / 2033 |
| max-res image **after 100K text** | 23950 / 14278 | **614** / 2033 |

The image adds ~730 MiB on GPU0 (the mmproj encoder, MG=0). With the max-res image, **GPU0 bottoms at 614 MiB free — above your 512 MiB floor, so TS stays 40,8** (no 39,9 fallback needed). It is below 1 GiB, so if you want ≥1 GiB even with max-res images, either ship `-ub 2048` (main-2e′: +900 MiB headroom, P-100K 624 instead of 821) or set **TS 39,9** (shifts ~1 layer to GPU1's 2 GiB spare). GPU1 is comfortable throughout.

## Production re-run under the round-2 harness (apples-to-apples baseline)
Current production exactly as deployed (old fork, MTP n2, shared head, f16, UBdef, NCM31 TS40,8, vision on), measured with the identical harness:

| metric | production (old fork) | **main winner (2e)** | gain |
|---|---|---|---|
| code decode | 14.84 | 19.41 | **+31 %** |
| prose decode | 16.30 | 19.56 | +20 % |
| decode @100K depth | 11.78 | 18.45 | **+57 %** |
| P-8K prefill | 209 | 590 | +182 % |
| P-100K prefill | 201 | 821 | **4.1×** |
| P-100K wall | 497 s | ~122 s | −75 % |

(2e's `-t 24` number shown; the shipped `-t 16` gives ~17.9 decode isolated / ~16.7 under embed load — still well above production's 14.8, and prefill is thread-independent at ~820.)

---

## Recommended configs (NOT applied)

Winning launcher: **`~/ai/flashnext-main/run-38-flash-next-main.sh`** (new `BIN`, self-contained head default, round-2 knobs `UB / CKPT / T / KVK / KVV / SPEC_OFF`). Same launcher serves both entries via env.

### MAIN `qwen-38-flash-next` — ship this (f16, no q8 gate dependency)
```
env BIN=/home/rohit/ai/flashnext-main/llama.cpp/build/bin/llama-server \
    SPEC_OFF=1 KVK=f16 KVV=f16 UB=3072 T=16 CKPT=8 \
    NO_WARMUP=1 NCM=31 TS=40,8 MG=0 CTX=131072 VISION=on PORT=8085 \
    /home/rohit/ai/flashnext-main/run-38-flash-next-main.sh
```
Decode ~17.9 isolated / ~16.7 under concurrent embed (vs prod 14.8); P-100K ~820 (4.1× prod). `T=24` instead of `T=16` gives ~19.5 decode **only if** no CPU embedding runs alongside. Optional max-prefill variant: `KVK=q8_0 KVV=q8_0 UB=4096` → P-100K ~1017 but decode-at-depth ~13.8 (q8 gate already passed).

### BATCH `qwen-38-flash-next-batch` — ship the throughput win
```
env BIN=/home/rohit/ai/flashnext-main/llama.cpp/build/bin/llama-server \
    NP=4 SPEC_OFF=1 VISION=off KVK=q8_0 KVV=q8_0 UB=2048 T=24 \
    NO_WARMUP=1 NCM=28 TS=37,11 MG=0 CTX=131072 PORT=8089 \
    /home/rohit/ai/flashnext-main/run-38-flash-next-main.sh
```
conc-8 ~32 (+20 % over prod), P-8K 637 (+165 %). Do **not** raise `UB` to 4096 (OOMs GPU1 at NP4). If this lane runs at the same time as CPU embedding, change `T=24`→`T=16`.

### Proposed diff for `swap/config.yaml` (and `.q6km`, `.plain` — all three identical) — apply yourself
```diff
   "qwen-38-flash-next":
     cmd: >-
-      env NO_WARMUP=1 NCM=31 TS=40,8 MG=0 CTX=131072 NMAX=2 VISION=on PORT=8085
-      /home/rohit/ai/flashnext/run-38-flash-next.sh
+      env BIN=/home/rohit/ai/flashnext-main/llama.cpp/build/bin/llama-server
+      SPEC_OFF=1 KVK=f16 KVV=f16 UB=3072 T=16 CKPT=8
+      NO_WARMUP=1 NCM=31 TS=40,8 MG=0 CTX=131072 VISION=on PORT=8085
+      /home/rohit/ai/flashnext-main/run-38-flash-next-main.sh

   "qwen-38-flash-next-batch":
     cmd: >-
-      env NP=4 SPEC_OFF=1 VISION=off NO_WARMUP=1 NCM=28 TS=37,11 MG=0 CTX=131072 PORT=8089
-      /home/rohit/ai/flashnext/run-38-flash-next.sh
+      env BIN=/home/rohit/ai/flashnext-main/llama.cpp/build/bin/llama-server
+      NP=4 SPEC_OFF=1 VISION=off KVK=q8_0 KVV=q8_0 UB=2048 T=24
+      NO_WARMUP=1 NCM=28 TS=37,11 MG=0 CTX=131072 PORT=8089
+      /home/rohit/ai/flashnext-main/run-38-flash-next-main.sh
```
Update the `description:` lines too if you like (MTP is now off; main ≈17 t/s decode / 4× prefill). Keep all three config files in sync.

## Switch-over (you run these)
1. Edit the three config files as above.
2. *(optional but recommended)* install the gate patch so the MTP-off entry scores cleanly: `cp ~/ai/flashnext-main/gates38next.mtp-skip.sh ~/ai/flashnext/gates38next.sh` (or merge the diff).
3. `sudo systemctl restart qwen-swap qwen-router`
4. Pre-warm + gate the main entry:
   `curl -s localhost:8000/load/flash-next` then
   `EXPECT_MTP=0 SKIP_DEEP=0 B=http://127.0.0.1:8085 MODEL_ID=flash-next bash ~/ai/qwen36/flashnext/gates38next.sh`
   → on mainline 11736: **15 PASS / 0 FAIL / 2 SKIP** (`reasoning_effort` now passes — the build honors it). Batch entry:
   `EXPECT_MTP=0 EXPECT_VISION=0 PROFILE=batch SKIP_DEEP=1 B=http://127.0.0.1:8089 MODEL_ID=flash-next-batch bash ~/ai/qwen36/flashnext/gates38next.sh` → **13 PASS / 0 FAIL / 3 SKIP**.
5. `bash ~/ai/qwen36/regress.sh` → `[0]` must stay 7↔7 (every model mapped).
6. `curl -s localhost:8000/unload`; then (optional) `wsl --shutdown`-reopen and `bash ~/ai/qwen36/systemd/verify-boot.sh`.

## Rollback (config-only, no rebuild)
Revert the two `cmd:` blocks in all three files to the old-fork launcher + shared head and `sudo systemctl restart qwen-swap qwen-router`:
```
env NO_WARMUP=1 NCM=31 TS=40,8 MG=0 CTX=131072 NMAX=2 VISION=on PORT=8085
/home/rohit/ai/flashnext/run-38-flash-next.sh
```
The old fork tree (`~/ai/flashnext`), its binary, and the shared head are untouched, so rollback is instant. (If you installed the gate patch, restore it from git or re-add the MTP checks.)

## Could not test / caveats
- **`-t 16` at `-ub 3072` not measured directly** — decode is thread/bandwidth-bound and `-ub`-independent, so `-t 16` decode ≈ the thread-sweep value (~17.9 isolated / 16.7 under load) and prefill is GPU-bound ≈ 821 regardless; stated as projected, not measured, for that exact pair.
- **Contention measured at two load levels** (both now direct, not projected): ~7.6 embed-cores (heavy bulk) collapses `-t 24` to 5.18; light RAG query traffic (one 48-tok embed / 2 s) still pulls it to 17.90 — below the 18.5 bar. So `-t 24`'s decode edge does **not** survive even trivial concurrent embedding → `-t 16` for the main entry. `-t 24` only where the embedder is guaranteed idle (dedicated deployment, batch lane).
- **q8_0 blessed only by 5-needle recall + gates38next**, not by a perplexity slice (no held-out corpus in the repo). Recall at 100K/125K is clean; a perplexity delta was out of scope.
- **`-ub 4096`**: fits the main entry only on q8_0 (2a); OOMs on f16 and at NP4. Not recommended outside the documented 2a variant.
- **Deep 262K / >128K** context not exercised (would need an NCM/TS re-fit); all tests ≤ 125K within the 131072 window.
- **MoE `--moe-cache-mib`** not re-tested (round 1 showed it neutral single-stream and harmful at concurrency); left at 0.
```
