# RESULTS

Two models, one stack. **Part A** = the Qwen3.6-27B dense stack (original build).
**Part B** = the Qwen3.6-35B-A3B MoE added alongside it via llama-swap (2026-07-15).
Both are served from `:8000`; llama-swap holds them mutually exclusive on the same two GPUs.

---

# PART A — Qwen3.6-27B local stack (measured on this machine, M4)

Hardware: RTX 4090 (Server A, text+MTP) + RTX 5060 Ti (Server B, vision). WSL2, CUDA 12.8, llama.cpp `657e011` (b9996-era), sm_89+sm_120, MMQ on, smpbo patch. Date 2026-07-14. All numbers measured here; none carried over from the setup prompt.

## FINAL CHOSEN CONFIG

**Server A (text, RTX 4090):** `-c 131072` · KV **q8_0** · `--spec-type draft-mtp --spec-draft-n-max 4` (MTP **on**) · `-fa on -np 1 --ctx-checkpoints 32` · v19 template · `--reasoning-preserve` · temp 0.6 (code) / 1.0 (general).
**Server B (vision, RTX 5060 Ti):** `-c 32768` · KV q4_0 · UD-Q3_K_XL + mmproj-F16 · no MTP · `--image-min-tokens 1024`.
**Router :8000** → text :8080 / vision :8081 (image-content routing, mandatory system prompt injected).

Reasoning per choice below.

## Step 1 — MTP is live (HARD GATE: PASSED)
- Local `gguf-dump`: `blk.64.nextn.eh_proj.weight` = **Q8_0** [10240,5120], nextn norms F32. Not INT4-packed → draft path loads.
- `/completion` timings fields (verified): **`draft_n`, `draft_n_accepted`**. Server log also prints per-request `draft acceptance = a/g, mean len = AL`.
- Live @n=2: acceptance **78.6–91.4%**, **AL 2.53–2.78**. AL≫1 → MTP contributes. Gate passed.

## Step 2 — Context to 131072 (PASSED)
- Configured `n_ctx_slot = 131072`. Needle probe at **125,962 tokens** → correct recall (`MERIDIAN-COBALT-7`), pp **1308 t/s**, no OOM.
- KV self size (computed, 16 full-attn layers × 4 KV-heads × 256 × 2 × 131072): **q8_0 ≈ 4352 MiB**, q4_0 ≈ 2176 MiB. Corroborated by an observed 5145 MiB prompt-cache entry for a 104K state.
- Peak VRAM @131072: **23372–23446 MiB** (n=4, active) → headroom **~1.1–1.2 GiB** (≥1 GiB ✓). (VRAM is preallocated for full ctx at boot, so depth doesn't change it.)

## Step 3 — `--spec-draft-n-max` sweep (chose n=4)
tg t/s p50 (AL) per shape:

| shape | n=2 | n=3 | n=4 | n=5 |
|---|---|---|---|---|
| short-code | 86.5 (2.55) | 92.5 (3.06) | **101.0 (3.53)** | 99.6 (3.79) |
| long-code | 87.4 (2.61) | 92.9 (3.15) | **118.0 (4.15)** | 112.5 (4.53) |
| rag-100k | 50.9 (2.17) | 59.4 (2.60) | 55.4 (2.60) | 67.7 (3.25)* |
| creative | 77.0 (2.22) | 74.7 (2.52) | 73.2 (2.72) | 68.4 (2.81) |

Aggregate acceptance falls as depth rises (short-code: .79→.70→.65), i.e. far draft positions are rejected more — llama.cpp exposes only aggregate acceptance + mean-len (AL), not per-position (vLLM does); the AL/n_max gap implies the decline. **Code (the #1 workload) peaks at n=4; n=5 regresses.** rag tg is prefill-bound/noisy on 24-token gens (*low sample). creative declines with depth but isn't the priority. **Chosen: n=4.**

## Step 4 — KV precision q8_0 vs q4_0 (chose q8_0)
Multi-needle retrieval at **100,081 tokens** (5 needles, greedy):

| KV | retrieval | VRAM | code acceptance |
|---|---|---|---|
| **q8_0** | **5/5** | 23372 MiB | ~0.65–0.80 |
| q4_0 | **2/5** (3 needles → empty output, even at 256 tok) | 21324 MiB | 0.86 |

q4_0 saves ~1.75 GiB but **measurably corrupts long-range retrieval** (structurally expected: only 16/64 layers carry KV, and quantization error there compounds at depth). Code-gen speed/acceptance were fine under q4_0, but retrieval is the deciding signal. Headroom for q8_0 exists. **Chosen: q8_0.**

## Step 5 — MTP on vs off (keep ON everywhere)
tg t/s p50:

| shape | MTP off | MTP on (n=4) | speedup |
|---|---|---|---|
| short-code | 44.3 | 101.0 | **2.28×** |
| long-code | 44.6 | 118.0 | **2.65×** |
| rag-100k | 36.3 | 55.4 | 1.53× |
| creative | 44.4 | 73.2 | 1.65× |

**Every** shape is net-faster with MTP — including creative and RAG. Decision gate (disable if acceptance <40% or latency regresses) **not triggered**. Keep MTP on.

## Step 6 — Router overhead (negligible)
Same prompt, streaming: direct :8080 ttft **332 ms** / tg 76.7 · router :8000 ttft **327 ms** / tg 82.8. Delta (−5 ms ttft, within noise) — proxy adds nothing measurable. SSE confirmed streaming (1100 chunks, unbuffered).

## Step 7 — Dual-loop soak (50 min, both GPUs) — PASS
3022 s, text agent-loop on GPU0 + heavy vision loop on GPU1, concurrent. **806 text turns, 762 vision generations.**

| assertion | result |
|---|---|
| RSS climb (MTP memory blowup) | **+31 MiB** over 50 min (noise; range 13.8–15.2 GB) — no climb |
| VRAM0 / VRAM1 climb | **0 / 0 MiB** — steady 23446 / 15444 whole run |
| GPU1 #24399 device exception | **0 over 762 vision gens** — not exposed (F16 mmproj, zero Q8_0 confirmed) |
| prefix reprefill (cache alive at end) | **109.1 → 109.7 ms** (max 114.9) — no degradation |

Text errors: 10/806 (1.2%) in a single ~t=1400–1640 s window, then none — **root-caused in P1 below as tool-call output truncation (not VRAM/TDR)**, now fixed by the router guard. Vision errors: 0. **VERDICT: PASS** — both feared failure modes (MTP blowup on 4090, #24399 on 5060 Ti) absent.

## Step 9 — Power (optional)
4090 already at **450 W** (= default = max; draws ~365 W under load). No cap to raise (the reference +10% was from lifting a *lowered* cap). Not settable from WSL (`Insufficient Permissions`; WDDM needs Windows-side admin). N/A here.

## TTFT
Short prompts (streamed, thinking on): ~230–330 ms to first token across shapes. Long-ctx RAG TTFT is dominated by prefill (~90 s for a full 126K prefill at pp 1308 t/s; ~0.4 s at 100K when cached).

## Requirement scorecard
1. Max speed — MTP on, n=4: **101–118 t/s** on code (2.3–2.7× vs off). ✓
2. ≥100K — 131072 held, 126K recall verified, no OOM, ~1.1 GiB headroom. ✓
3. Vision — 4/4 fine-text + 12/12 document fields; 762-gen soak clean. ✓
4. Clean output — digit-loop clean, tool calls clean (temp 0.6 & 0.2), q8_0 retrieval 5/5. ✓

---

# Post-ship hardening

## P1 — soak's 10 text errors: root-caused (NOT VRAM)
Tool-call output truncated by `max_tokens`: soak used `max_tokens:300` on tool turns; with thinking on, long-reasoning sessions overran the budget and cut the tool-call JSON mid-argument → llama.cpp 500 (`Failed to parse tool call arguments as JSON`). Proven by a max_tokens sweep (mt=80 truncates `{"cmd":"ls`, mt=512 clean). Ruled out: VRAM (GPU0 flat at 23446 MiB through the window), WDDM TDR (no `nvlddmkm` events), router timeout (soak hit :8080 directly), context (exceed:0).
**Fix:** router `_guard_tool_budget` raises `max_tokens` to a floor (`TOOL_MIN_TOKENS`, default 512) when tools are present. A/B verified (direct :8080 truncates, router :8000 clean). Regression **test #13** added → suite now **15 PASS / 0 FAIL**.

## P2 — N/A
The errors were not VRAM, so the VRAM-lever ladder does not apply. **KV stays q8_0** (locked — q4_0's 2/5 retrieval is disqualifying). No tuning touched.

## P3 — systemd installed and VERIFIED on a real cold boot (PASS)
Units hardened: explicit `Environment=CUDA_VISIBLE_DEVICES=0/1` (belt-and-suspenders over the run scripts' `exec env` pinning, since systemd doesn't inherit the shell env); router `ExecStartPre` **readiness gate** waits for both backends' `/health` before starting (fixes `After=` ordering ≠ readiness); `Restart=on-failure`.

**Install:** `sudo bash ~/ai/qwen36/systemd/install.sh` — performed, 3 symlinks created, units enabled.

**Install-time failure (a conflict, NOT a unit misconfiguration).** On first enable, all three units flapped in `activating (auto-restart)` and `verify-boot.sh` returned FAIL. Root cause: the **three manually-launched processes from the build session** (PIDs 31594/45464 llama-server, 75904 uvicorn — up ~20 h) still held ports 8080/8081/8000 and all VRAM. Router logged `[Errno 98] address already in use`; text/vision logged `llama_server: exiting due to HTTP server error`. Diagnostic that mattered: the text unit's log showed **CUDA loaded cleanly before the bind failure** → the systemd env was already correct, so this was a port/VRAM conflict, not an env problem. Confusingly, `:8000` *did* answer "Paris" during the failure — the **stale** router was serving it. Fix: kill the 3 stale PIDs → GPU freed to 0/0 → systemd's auto-restart took the clean field and all three were `active` within ~6 s. (The `NRestarts=105` seen afterwards was cumulative scar tissue from the ~9 min conflict window, not ongoing flapping — `ActiveEnterTimestamp` was stable.)

**Cold-boot verification (the real gate, 2026-07-15 14:57): PASS.** After a full WSL restart — uptime 3 min, `llama-server` PIDs **270/271** with `ppid=1`, **`NRestarts=0` on all three units** — the stack came up unattended, first try, with no manual processes possible:

| check | result |
|---|---|
| units | `qwen-text` / `qwen-vision` active @14:57:31, `qwen-router` active @14:57:46 |
| readiness gate | router waited **15 s** for both backends' `/health` — gate working as designed |
| GPU pinning | **GPU0 23442 MiB / GPU1 15366 MiB** — held across boot |
| `:8000/healthz` | `{"router":"ok","backends":{"text":200,"vision":200}}` |
| end-to-end | "Paris" |
| **verdict** | **BOOT VERIFY: PASS** |

Tool-budget guard (P1) confirmed live in the **systemd-managed** router: at `max_tokens=80`, `finish=tool_calls`, args clean `{"cmd":"ls -l /var/log"}`. systemd `is-system-running` = `degraded` is the pre-existing unrelated unit flagged back in M1; all three of ours are active.

## P4 — Server B (Q3_K_XL) vision quality
- **16384-context IQ4_XS budget: does NOT fit.** IQ4_XS weighs +922 MiB over Q3_K_XL; dropping ctx 32K→16K frees only 288 MiB (KV is tiny — 16/64 layers) → ~311 MiB free, below CUDA-ctx + compute-buffer needs. **Context is not a VRAM lever on this hybrid arch.** So the "trade vision context for quality" path is closed.
- **Q3_K_XL quality: 12/12 field-level extraction** on a realistic document (`models/test-doc.png`: table rows, currency, tax %, alphanumeric codes `ACM-8842-XT`/`qz7-VELUM-3388`/`00-CX-91724`, name, city). Plus earlier 4/4 and the 762-gen soak (0 errors). The vision **encoder is full F16** (OCR-critical); Q3 only quantizes text-decode. **No measurable degradation observed → accept Q3_K_XL.**
- **Caveat:** validated on synthetic docs only (user's real documents not available). A Q6 reference diff would have ~no delta given Q3 is already at ceiling here; worth running only against real target documents. Fallback if real docs degrade: move vision to the 4090 with model-swapping (IQ4_XS won't fit the 5060 Ti at any context).

---

# PART B — Qwen3.6-35B-A3B (MoE) via llama-swap (2026-07-15)

Same binary (CUDA 12.8, sm_89;120, MMQ, smpbo patch) — **no rebuild**. Arch `qwen35moe`, 41 blocks
(40 + 1 MTP), 256 experts / 8 active, `full_attention_interval=4` → 10 KV-bearing layers.
`UD-Q4_K_XL` 21.284 GiB, sha256 `55983c5a…` verified, local tensor table byte-identical to the
remote header probe.

## FINAL CHOSEN CONFIG (Server C)
`-c 131072` · KV **q8_0** · **min-spill expert offload: `-ot blk.(28|29|30|31|32|33).ffn_(gate|down|up)_exps.weight=CUDA1`** ·
`-sm layer -ts 1,0 -mg 0` · `--spec-type draft-mtp --spec-draft-n-max 3` (MTP **on**) · `-fa on -np 1` ·
v19 template (embedded one is **stale**) · `--reasoning-preserve`. Residency **GPU0 21422 / GPU1 2978 MiB**.

## Placement: min-spill, not "fill GPU1"
The 5060 Ti is **overflow storage, not a second compute engine**. Blocks are strictly sequential, so
total decode = `t_GPU0 + t_GPU1`, never `max()` — there is no parallelism to harvest, and every byte
on the slow card (448 GB/s vs the 4090's 1008) is a straight tax, plus a PCIe round-trip + sync per
spilled layer. So spill the FEWEST layers that let the budget close. Spill pool is **plain-464 MiB
layers only** (blk.34/38 are Q6_K-bumped to 498, blk.39 to 562 — moving a fat layer costs more
bytes/token for zero benefit); blk.40 (MTP block) stays whole on CUDA0.

**#24399 is structurally dead here, by construction:** routed experts are exclusively Q4_K/Q5_K/Q6_K;
every Q8_0 tensor (attn_qkv, attn_gate, ssm_out, shexp, token_embd, output) is non-expert and stays
on CUDA0. The `_exps`-only regex is what holds the gate — so residency is verified at boot, not trusted.

## SPILL sweep + expert-offload vs tensor-split (MTP on, n=3, tg t/s median of 3)

| config | GPU0 | GPU1 | short-code | long-code | creative | rag-100k* |
|---|---|---|---|---|---|---|
| **spill 6 (SHIPPED)** | 21422 | 2978 | **206.1** | **198.3** | **132.0** | 126.5 |
| spill 10 | 19568 | 4834 | 179.5 (−12.9%) | 169.2 (−14.7%) | 117.4 | 121.1 |
| spill 16 | 16784 | 7618 | 148.6 (−27.9%) | 144.3 (−27.2%) | 124.9 | 107.5 |
| tensor-split .85 | 20842 | 5188 | 174.9 (−15.2%) | 164.6 (−17.0%) | 110.5 | 102.8 |
| tensor-split .91 (matched GPU1) | 22412 | 3608 | 169.8 (−17.6%) | 166.0 (−16.3%) | 112.2 | 104.7 |

**The curve is monotonically worsening — there is no flat region.** Spilling more never buys
throughput (no parallelism to unlock), it only adds bandwidth + sync cost. Prefill degrades too
(pp 205 → 166 → 141). **Min-spill confirmed empirically, not just modelled.**

**Expert-offload beats naive tensor-split by ~18–21% on every shape, at matched GPU1 residency.**
The gap is too large to be GPU1-bytes alone (tsplit has 21% more there; the spill sweep prices that
at ~4%) — the remaining ~16% is structural: tensor-split drags **attention and KV** onto the slow
card, not just expert weights. That is why rag-100k suffers worst (104.7 vs 126.5).

## MTP on vs off — MTP ships ON

| shape | OFF | ON (n=3) | speedup | accept | AL | AL/2 |
|---|---|---|---|---|---|---|
| short-code | 112.8 | **205.7** | **1.82×** | 0.888 | 3.61 | 1.80 |
| long-code | 112.0 | **198.0** | **1.77×** | 0.828 | 3.45 | 1.72 |
| creative | 112.7 | **140.8** | **1.25×** | 0.459 | 2.36 | 1.18 |
| rag-100k* | 88.0 | **132.2** | 1.50× | 0.750 | 2.80 | 1.40 |

Net-positive on **every** shape; the "<40% acceptance → disable" gate never triggered (worst 0.459).

**Baseline gate PASSED: MTP-off = 112.8 t/s, 2.6× above the 44 t/s floor** → the cross-GPU expert
split is placed correctly. Corroborated: 112.8 is **2.55× the 27B dense's 44.3 t/s**, exactly what
3B-active-vs-27B-dense predicts.

**Decode is memory-bandwidth-bound, and the MTP-off column proves it:** short-code 112.8, long-code
112.0, creative 112.7 — a **0.7% spread across completely different content**. Content-independent
decode is incompatible with compute-binding. (rag-100k's 88.0 is the one deviation = KV reads growing
with context, a different tensor path.) The prior "a MoE gains less from MTP because few active
params" reasoned about compute; every token reads the same weight volume regardless.

**speedup ≈ AL/2 across all four shapes**, holding over a 1.5× range of AL: a verify step costs ~2× a
decode step and returns AL tokens, because weights are read **once per verify step instead of once per
token**. Slightly *above* AL/2 in 3 of 4 → verify is a bit cheaper than 2×.
**Not established:** that this amortization is *MoE-specific*. The 35B reads ~586 MiB of experts vs
~1958 MiB of dense weights per token, so it mostly amortizes the dense path. Settling it needs a
dense-vs-MoE comparison at matched bandwidth, which this hardware can't provide.

## n_max sweep (at min-spill) — n=3 ships

| config | short-code | long-code | creative | rag-100k* |
|---|---|---|---|---|
| n=2 | 175.8 | 169.8 | 140.4 | 115.3 |
| **n=3 (SHIPPED)** | **206.1** | **198.3** | **132.0** | 126.5 |
| n=4 | 207.4 | 192.4 | 112.1 | 157.7* |

Mixed, and decided on the priority workload: short-code is a wash (206.1 vs 207.4), long-code prefers
n=3 (+3%), creative strongly prefers n=3 (**+18%**). Only RAG prefers n=4. Code is priority #1 →
**n=3**. `SPEC_NMAX` is exposed for RAG-heavy use. (Contrast: the dense 27B chose n=4 — the MoE peaks
earlier, and acceptance falls faster with depth: 0.934 → 0.909 → 0.849 on short-code.)

## * The rag-100k numbers in the tables above are INFLATED — read this
The sweep's rag shape predicts only **48 tokens** on a deterministic answer (`MERIDIAN-COBALT-7`).
The model is echoing a near-certain string, so acceptance and tg are both inflated. The tell was a
**non-monotonic acceptance** (n=2 0.80 → n=3 0.75 → **n=4 0.92**), the opposite of the established
falls-with-depth pattern. Re-measured with a real 250-token prose generation at 99937 tokens:

| | tg | acceptance | AL |
|---|---|---|---|
| n=3 | **95.2** (reps 97.9/94.4/95.2) | **0.495** | 2.45 |
| n=4 | **107.0** (reps 98.3/107.0/138.2 — noisy) | **0.505** | 2.98 |

**True long-context RAG decode is ~95–107 t/s, not 126–158**, and true acceptance is ~0.50, not
0.75–0.92. Any future rag comparison must use a long, non-deterministic generation.

## Swap correctness (llama-swap) — PASS
Pass condition is **not** "the next model started" — a partial evict looks like success until VRAM
runs out mid-load. It is: **GPU0 → ~0 FIRST, then the next model loads.** Proven on a 200 ms VRAM trace:

```
+0.00s GPU0=23422 GPU1=0     27B loaded
+0.72s GPU0=   10 GPU1=0     <-- fully drained
+0.95s GPU0=  384 GPU1=128   <-- 35B's FIRST allocation, AFTER
+23.8s GPU0=21468 GPU1=2990  ready
```
**Zero overlap**, and structural rather than lucky: llama-swap SIGTERMs, waits for **process exit**,
and the driver reclaims VRAM when the CUDA context dies with the process. Effective margin ~22 s.
Verified both directions; the 27B holds **GPU1=0** under llama-swap (pinning survives), and GPU1
drains to 0 on swap-back (spilled experts released too).

**Swap latency: 27B→35B 23.9 s, 35B→27B 10.0 s** — disk-bound, not swap-bound (the 16 s flat stretch
at GPU0=384 is 21 GB paging in). Warm page cache = 4.9 s. With 38 GB of weights against 47 GB of RAM
both models cannot stay cached, so expect **5–25 s** depending on cache state. Unrelated: the **first
request after any load is ~15% slow** (174.7 vs 205 t/s) from graph warmup — not a swap regression.

## Architecture (as shipped)
```
client -> qwen-router :8000  (tool-budget guard, system-prompt injection)
       -> llama-swap  :9000  (mutual exclusion, load-on-demand)
       -> llama-server :8080 qwen-27b   (GPU0 only)   | aliases: qwen36, qwen36-text
       -> llama-server :8082 qwen-35b   (GPU0+GPU1)   | aliases: qwen36-35b
```
**The router still exists because llama-swap cannot express the tool guard.** Its `filters.setParams`
is an *unconditional* set ("always runs for the model"), not a floor, and cannot condition on `tools`.
Setting `max_tokens: 512` there would cap **every** request at 512 — truncating long code and 100K RAG
answers. The guard needs `IF tools AND max_tokens < floor THEN raise`, which is conditional logic on
the request body. Verified live through the chain: `raised max_tokens 80 -> 512`, `finish=tool_calls`,
clean args (regression #13 survives the re-architecture).

**Alias `qwen36` → qwen-27b, deliberately.** Existing clients and regress.sh send `qwen36` expecting
27B behaviour; the shipped stack stays intact and the 35B is opt-in by name.

## M2 gates (35B): 16 PASS / 0 FAIL
Digit-loop canary **PASS** under real pressure — 16 samples, 4148–5423 chars, **10,627 drafts
rejected**, zero runaways. Embedded template **rejected as stale** (see LEARNINGS). Tool-in-think clean
at temp 0.6 **and** 0.2. SSM 41/41. 100K needle recalled (`MERIDIAN-COBALT-7`, pp 3566 t/s).

## 35B soak (45 min, through the shipped chain) — PASS
2703 s, **641 turns, 0 errors**, run through `:8000` router → llama-swap → 35B (the real client path;
the 27B soak hit `:8080` directly and therefore had no tool guard).

| assertion | result |
|---|---|
| **tool turns @ `max_tokens=80`** (the 27B's exact failure budget) | **128 turns, 0 errors** — guard in path |
| **digit-loop canary** (temp 1.2, cross-GPU MTP rollback) | **128 turns, 0 runaways** |
| **long-context turns** (~100K, needle asserted each) | **64 turns, 0 recall misses** |
| **GPU0 VRAM climb** | **+0 MiB** — dead flat at 21498 the whole run |
| **GPU1 VRAM climb** | **+0 MiB** — dead flat at 3000; **the spilled experts do not leak** |
| **server RSS (steady state)** | **+8 MiB** over the 2nd half (10142 → 10150) — no MTP blowup |
| tg | median **143.9 t/s** (min 129.5, max 221.9) across the mixed workload |
| acceptance | median **0.519** |

**RSS must be read as steady-state, not full-run.** The full-run delta is **+2422 MiB** (7728 → 10150)
and that is **mmap page-in, not a leak** — the 21 GB of weights fault in gradually, so RSS necessarily
climbs early. Reporting the whole-run number would report warm-up as the very memory blowup the soak
exists to detect. The leak signal is the second-half climb: **+8 MiB**.

**The in-soak prefix probe was contaminated — do not read it as a cache result.** It swung 939–8573 ms
with no trend (2nd-half median 3323 ms was *lower* than the 1st half's 6825). The server runs
`--parallel 1`, so the probe queued behind the worker's in-flight request; the 8573 ms max matches a
1200-token canary turn at ~140 t/s almost exactly. Measured properly on the **idle** server after the
soak: **615 ms cold → 134–143 ms cached, `prompt_n=4`** of ~2600 tokens (rest served from cache) —
prefix caching is alive and fast after 45 min of load. **Any future in-soak cache probe needs an idle
server or a dedicated slot.**

**The tool-guard result is the headline.** The 27B soak produced 10/806 truncation errors at
`max_tokens:300`; this soak deliberately used the *harsher* `max_tokens=80` on every tool turn and got
**0/128** — the guard fix verified under sustained load, not just at rest.

---

# PART C — Qwen3.5-122B-A10B (MoE) via llama-swap, TRI-TIER (2026-07-15/16)

**EXPERIMENT, not a production ship.** Bar: biggest Qwen MoE that fits and clears **10 t/s** without
being overly quantized. Same binary — **no rebuild** (arch `qwen35moe`, same family as the 35B).

## FINAL CHOSEN CONFIG (Server D)
`UD-IQ3_S` (48.44 GiB, 3-part, sha256 verified) · `-c 131072` · KV **q8_0** · **MTP on, `--spec-draft-n-max 3`**
(NOT the card's 6 — see below) · `-fa on -np 1` · `-sm layer -ts 1,0 -mg 0` · v19 template · `--reasoning-preserve`.
Tri-tier `-ot` (ONE comma-separated arg):
```
blk.(0..16).ffn_(gate|down|up)_exps.weight=CPU , blk.(17..32).ffn_(gate|down|up)_exps.weight=CUDA1
```
Residency **GPU0 23936 / GPU1 14648 MiB / RSS ~17–25 GiB**.

## Measured vs the 10 t/s floor — clears it 2.4–4×

| shape | MTP off | **MTP on (n=3)** | floor |
|---|---|---|---|
| short-code | 24.43 | **36.86** (1.51×) | 10 |
| long-code | 25.37 | **36.60** (1.44×) | 10 |
| creative | 24.84 | **24.40** (0.98×) | 10 |
| rag @100K | — | **26.8** | 10 |
| (100K / 125K needle, n=6 cfg) | — | 39.0 / 35.5 | 10 |

**Baseline gate PASSED**: MTP-off 24.4 t/s, 2.4× the floor. Gates **17 PASS / 2 FAIL** (both non-issues:
GPU0 headroom, and a harness `max_tokens` artifact — see below).

## The tri-tier split — "minimise the SLOWEST TIER", which INVERTS the 35B's rule

| tier | experts | layers | note |
|---|---|---|---|
| GPU0 4090 | 14.48 GiB | 16 (incl. fat blk.46 @1290 MiB and MTP blk.48) | + nonexpert 4.96 + KV 1.73 |
| GPU1 5060 Ti | 14.06 GiB | 16 | **FILLED, not min-spilled** |
| **CPU RAM** | **14.94 GiB** | **17** | **34.4%** — the slowest tier, IQ2_S/IQ4_XS |

The 35B's "min-spill" was never "minimise GPU1" — it was **minimise the slowest tier in play**. There,
GPU1's alternative was the *faster* 4090, so min-spill starved GPU1. Here the slow tier is the CPU
(DDR5 ~85 GB/s **plus i-quant dequant compute**), and GPU1 at 448 GB/s is the *fast* alternative to it.
Same principle, **opposite direction**: fill GPU1 to starve CPU.

## THE KEEPER NUMBER — CPU tax, and it is a CURVE not a constant

| n_cpu | tg | ms/tok |
|---|---|---|
| **17 (shipped)** | **40.54** | 24.665 |
| 21 | 37.73 | 26.503 |
| 25 | 33.50 | 29.849 |

**Marginal cost of one 900 MiB expert layer moved GPU1→CPU: 0.459 ms (17→21) → 0.837 ms (21→25).
It nearly DOUBLES as the tier fills — the CPU saturates.**
CPU tax at n_cpu=17 ≈ **11 ms/tok of 24.7 (~45%)**; back-computed **all-VRAM ceiling ≈ 60–66 t/s** vs
40.5 measured. (llama.cpp exposes NO per-device timing — llama-bench isn't built, `GGML_SCHED_DEBUG`
shows placement not time. Ablation is better anyway: it gives the MARGINAL cost, which a counter can't.)

## MTP DOES NOT AMORTIZE CPU WORK — and on creative it goes NET-NEGATIVE

**The tell:** the CPU slope is IDENTICAL with MTP on vs off — **0.648 vs 0.682 ms/layer (ratio 0.95)**.
If MTP amortized CPU work the ON slope would be ~AL/2 smaller. **CPU cost scales with TOKENS, not
verify steps** — the signature of compute-bound work.

Decomposed at n_cpu=17: raw speedup **1.70×** vs AL/2 = 2.77 (law appears broken); **GPU-portion
speedup 2.22×** vs AL/2 = 2.77 (**law approximately HOLDS**). The AL/2 law is intact where it applies;
the CPU tier is the entire deviation. Predicted and confirmed: more CPU → less MTP win (1.70× at
n_cpu=17 → 1.59× at n_cpu=25).

**Which produces a SIGN FLIP on creative:**

| n_max | short-code | long-code | creative | creative acc |
|---|---|---|---|---|
| 3 (**shipped**) | 36.86 (1.51×) | 36.60 (1.44×) | 24.40 (**0.98×**) | 0.47 |
| 4 | 38.79 (1.59×) | 36.90 (1.45×) | 21.43 (**0.86×**) | 0.36 |
| 6 (card's rec) | **40.71 (1.67×)** | 36.41 (1.44×) | 18.10 (**0.73×**) | 0.24 |

**At the card's n=6, MTP makes creative 27% SLOWER than MTP-off.** On the all-VRAM 35B, creative at
acceptance **0.459** still GAINED **1.25×** — a rejected draft on a bandwidth-bound GPU is ~free (the
weights were read anyway). Here **a rejected draft costs FULL CPU COMPUTE**. Same acceptance, opposite
sign; the tier is the whole difference. **n=3 is the only depth that is never harmful → shipped.**

**COUPLING: n>=7 will NOT BOOT.** `failed to create MTP context` is the symptom; `cudaMalloc failed:
out of memory` is the cause — deeper draft buffers don't fit GPU0's 602 MiB headroom. Raising n_max
past 6 requires rebalancing a layer off GPU0 first.

## GPU0 headroom: 602 MiB, accepted ON EVIDENCE (below the 1 GiB gate)
The ~125K prefill passed with VRAM **identical before and after** (23962 both), zero OOM. VRAM is
preallocated at boot and does **not** grow with prefill, so 602 MiB is stable headroom, not a cliff.
The 45-min soak confirmed: GPU0 climb **+4 MiB**, GPU1 **+2 MiB**. Rebalancing one layer to CPU would
buy 900 MiB at ~2% speed — declined; the measurement earned it.

## Expert-split sweep: COMPLETE BY CONSTRUCTION
GPU0 and GPU1 are both full, so CPU↔GPU1 is the only axis, and it worsens monotonically. `n_cpu=17`
is the minimum that fits and therefore the optimum. Nothing left to try.

## Swap correctness (three-way, incl. the NEW RAM tier) — ALL PASS

| transition | latency | GPU0 min during | RAM |
|---|---|---|---|
| cold → 27B | 14.9 s | 0 | — |
| 27B → 122B | **49.1 s** | **12 MiB** | — |
| 122B → 35B | 17.4 s | 0 | **RSS → 26 MiB** |
| 35B → 122B | 40.1 s | 2 MiB | — |
| 122B → 27B | 21.0 s | 0 | **RSS → 19 MiB** |

**The RAM slice is released by process lifetime, exactly like VRAM** — the 122B holds ~17–25 GiB RSS
and it dips to 19–26 MiB before the next model allocates. That was the unproven mechanism; it holds.
Swap latency **40–49 s in / 17–21 s out** (48 GiB across three tiers) — disk-bound, reported not fixed.
`healthCheckTimeout` raised 300 → **900** (a COLD 122B load is far slower than the ~90 s warm case;
300 would time it out and look like a crash).

## Soak (45 min, through the shipped chain) — PASS
2730 s, **104 turns, 0 errors**, via `:8000` router → llama-swap → 122B.

| assertion | result |
|---|---|
| tool turns @ `max_tokens=80` (per-model floor raises to 4096) | **21 turns, 0 errors** |
| digit-loop canary (temp 1.2) | **21 turns, 0 runaways** |
| long-context turns (~100K, needle asserted) | **10 turns, 0 misses** |
| GPU0 / GPU1 VRAM climb | **+4 / +2 MiB** — flat |
| **RSS steady-state slope (regression)** | **−218 MiB/hour → NO LEAK** |
| tg / acceptance | median **30.4 t/s** / **0.591** |
| prefix cache (idle probe, post-soak) | **4744.9 → 168–202 ms**, `prompt_n=4` of 2620 |

**RSS needed a REGRESSION, not an endpoint delta.** The naive 2nd-half endpoint delta read **+595 MiB**
and looked like a small leak. The mmap'd CPU tier's pages are continuously evicted and re-faulted, so
RSS genuinely oscillates (**sd 533 MiB, spread 2680 MiB**) — the +595 is well inside the noise. The
2nd-half regression slope is **−218 MiB/hour** and the thirds show no trend (24717 → 24966 → 24831).
1st-half slope +2844 MiB/hr = page-in. (The 35B's +8 MiB was unambiguous so endpoint-delta happened to
work there; on a noisy series it is the wrong statistic.)

## Quality — see PROBE_RESULTS.md
`UD-IQ3_S` is **~52% IQ2_S by expert bytes** (gate/up at 2-bit; `down_exps` at IQ4_XS because it writes
back into the residual stream and is error-sensitive; attention/ssm/shexp all Q6_K). The label is a
weighted average, not a description.
**M2 probe: 122B 24/24, 35B control 24/24, zero inconsistency** — but the control also aced it, so it
had no discriminating power. **A discriminating probe could NOT be built**: across 73 items in 5
categories and 2 difficulty tiers the 35B aced **93%**, leaving only 3 clean gap items. Verdict:
**set not discriminating — inconclusive**; `UD-Q4_K_XL` was **NOT downloaded** (nothing to score it on).
IQ3_S ships on the stated bar (clean outputs) — **absence of evidence for harm, not evidence of absence**.

## The 397B-A17B question — the answer is NO, and it is the compounding of two curves
A 397B at IQ3 would be ~150 GB against 40 GiB of VRAM → **~73% on CPU**, far past where the CPU slope
already doubled (0.459 → 0.837 ms/layer), **and** MTP recovers less at every step as the CPU share grows
(1.70× → 1.59×), going outright negative on low-acceptance work. **The 122B is approximately the wall
for this box** — not because 40 t/s is slow, but because the next rung lands where both curves turn
against you simultaneously.

# Part D — 27B Q6_K_XL (quality-first dense, ADDED alongside the Q4 entry)

Added `qwen-27b-q6` (alias `qwen36-q6`) as a 4th mutually-exclusive llama-swap member. Same 27B, higher
quant: **UD-Q6_K_XL 24.23 GiB** (vs Q4_K_XL 17.9 GiB). Too big for the 4090 alone, so it spans both GPUs
via `--split-mode layer --tensor-split 3,1` (NOT `-ot`: dense model → per-tensor placement would thrash
PCIe twice/layer). The Q4 entry (`qwen-27b`) and its `.gguf` are untouched — Q6 is additive, Q4 stays as
the fast/fallback option.

**Residency @ TS=3,1 (measured):** GPU0 22326 MiB used / 2238 free · GPU1 10402 MiB used / 5909 free.
Both cards ≥1 GiB free. Note this CONTRADICTS the brief's "GPU1 in 6–9 GiB" estimate — GPU1 actually
carries ~10.4 GiB because **GPU0 (24.5 GiB) is the binding constraint**: 3,1 already pins GPU0 at 2.2 GiB
free, so GPU1 must hold the rest. 3,1 is also **speed-optimal** (minimises the slow 5060 Ti's share — the
same min-spill principle as the 35B); the brief's remedy "shift toward 5,2" addresses the opposite
failure and would only slow it down. The residency gate band was corrected to the measured envelope; the
real guard is ≥1 GiB free on each card.

**Speed (apples-to-apples, identical prompts, same box):**
| | Q4 (GPU0-only) | Q6 (dual-GPU) | retained |
|---|---|---|---|
| code decode | 117.6 t/s | 64.3 t/s | 55% |
| creative decode | 68.5 t/s | 37.0 t/s | 54% |
| MTP acceptance / AL | 0.896 / 4.49 | 0.860–0.891 / 4.34–4.49 | ~equal |
| 100K-ctx decode | — | 42.5 t/s | — |

The ~45% cost decomposes cleanly: ~0.74× from bigger weights (24.2/17.9 GiB bandwidth) × ~0.73× from the
split (slow GPU1 + PCIe residual hop). MTP is **equally healthy** on both — the slowdown is pure
weight-size + split, not a draft-path regression. The flagged risk (does draft-mtp survive a
`--tensor-split` layer boundary?) is **resolved**: it drafts and accepts normally across devices.

**Gates:** `gates27q6.sh` **16 PASS / 0 FAIL** (dual-GPU residency, digit-loop canary across split,
tool-in-think both temps, prefix-cache, MTP-live, 100K needle `MERIDIAN-COBALT-7`). `regress.sh` against
the Q4 fallback **18 PASS / 0 FAIL / 1 SKIP** — the +1 over the prior 17 is exactly the new gate-[0] line
asserting `qwen-27b-q6 → gates27q6.sh` (a live gate, not drift). Gate [0]'s both-directions check now
covers all four models.

---

# Part E — Qwen3.8-27B (dense + native VISION, replaced the 3.6-27B dense line)

Qwen3.8 shipped. The **entire 3.6-27B dense line** (Part A's Q4 `qwen-27b` + Part D's Q6 `qwen-27b-q6`,
plus the 3.6 vision projector) was **retired and its weights deleted** (~59 GiB freed), replaced by a
single Qwen3.8-27B multimodal server `qwen-38-27b`. The 35B MoE and 122B are untouched.

**What it is.** `unsloth/Qwen3.8-27B-GGUF` → `Qwen3.8-27B-Q6_K.gguf` (22.9 GiB, sha256-verified against
the HF LFS oid `ade6d66…f22f8ccc`) + `mmproj-F16.gguf` (F16, not Q8_0 → dodges Blackwell #24399). Arch is
`qwen3_5` — the **same family as the 35B/122B** (Gated-DeltaNet hybrid + vision), weights differ; the
existing llama.cpp build (657e011) already has the `qwen3vl` graph, `clip_graph_qwen3vl`, and
`gated_delta_net` kernels, so **no rebuild**. MTP head is embedded (`mtp_num_hidden_layers:1`).

**Placement.** Q6_K won't fit the 4090 alone once KV + vision buffers are added, so it spans both GPUs:
`--split-mode layer -ts 3,1 -mg 0`. Measured residency GPU0 **21810 MiB** (free 2754) / GPU1 **9360 MiB**
(free 6951) — ≥1 GiB free on each. mmproj/clip sits on GPU0 with output.

**The two risks, both resolved by measurement:**
1. *Vision on this build* — a solid-red ground-truth image is read as "red" end-to-end (pixels → tower →
   LLM). The `qwen3vl` mtmd path works.
2. *MTP coexisting with `--mmproj`* (the 3.6 vision path ran WITHOUT MTP over slot-position fears) —
   the text path still drafts with the vision tower loaded: **acceptance 0.845, AL 4.31, 73.4 t/s** code.

**Speed vs the retired 3.6-Q6:** ~73 t/s code (up from 64.3) and 48 t/s @100K (up from 42.5) — 3.8-Q6_K
is 22.9 GiB vs 3.6's 26.0 GiB Q6_K_XL, so it's both faster *and* gains vision. The embedded 3.8 chat
template parses reasoning and tool calls cleanly (no v19 hand-fix needed).

**Gates:** `gates38.sh` **17 PASS / 0 FAIL** — dual-GPU residency, digit-loop canary across split (with
the padded-past-200 / exit==1 self-test), thinking-closure, tool-in-think both temps, MTP-live-with-mmproj,
prefix-cache, **VISION**, 100K needle (`MERIDIAN-COBALT-7`). Run standalone on a test port (:8099) while
the training GPU stayed untouched. `regress.sh [0]` now maps `qwen-38-27b → gates38.sh`, `qwen-35b`,
`qwen-122b` — three models, three live gates, both directions.

**Aliases carried forward:** `qwen36`, `qwen36-q6`, `qwen36-text` all resolve to `qwen-38-27b`, so every
existing client keeps working; new names are `qwen38` / `qwen38-27b`.
