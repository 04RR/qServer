# LEARNINGS — Qwen3.6-27B local stack

Log of every bug/surprise hit during the build, with fixes, flag names, and source URLs. Dated 2026-07-14.

## M1 — toolkit + build

### Environment ground truth (differed from the setup prompt's estimates)
- WSL already had 47 GiB RAM / 16 GiB swap (`.wslconfig` = 48GB/16GB/sparseVhd **already applied** before this run). The prompt's "31 GiB / 8 GB swap, must raise" step was already done.
- Both discrete GPUs clean at boot: GPU0 RTX 4090 (cc **8.9**), GPU1 RTX 5060 Ti (cc **12.0**), 0 MiB, empty PROCESSES.
- Driver 591.86 / CUDA 13.1 max. Root `/` 780 GB free. `/mnt/d` is 9p (unusable). Build + models live in `~/ai/qwen36`.

### Research verification (July 2026) — key corrections to the prompt
- `--spec-type draft-mtp` CONFIRMED present in `--help` (full option list: none, draft-simple, draft-eagle3, **draft-mtp**, draft-dflash, ngram-*). `--spec-draft-n-max` default is **3**. `--draft/--draft-n/--draft-max` removed. MTP PR #22673 (am17an) merged.
- **mmproj Q8_0 does NOT exist** — unsloth ships BF16/F16/F32 only; froggeric has an f16. Prompt's `mmproj-*-Q8_0.gguf` is wrong → use **F16** (also sidesteps Blackwell Q8_0 bug #24399). [https://huggingface.co/unslothai... /Qwen3.6-27B-MTP-GGUF]
- MTP + `--mmproj` still broken (#22867, #23371 open) → two-server design confirmed.
- Blackwell #23385 (smpbo MMQ crash) & #24399 (Q8_0 sm_120) both OPEN, no merged fix as of b9996.
- "Digit-loop fixed in b9300+" UNVERIFIED (#23577 open) → treated as empirical gate (regress test #3), MTP-off is the fallback.
- Latest tag ~b9996 (2026-07-14). Cloned master `657e011`.

### Bug hit: stray CUDA 11.5 nvcc hijacked CMake (NOT in the prompt)
- `which -a nvcc` → `/usr/local/cuda-12.8/bin/nvcc`, `/usr/local/cuda-12.4/bin/nvcc`, **`/usr/bin/nvcc` (release 11.5!)**, `/bin/nvcc` (11.5).
- Despite `export PATH=/usr/local/cuda-12.8/bin:$PATH`, CMake's CUDA-compiler detection resolved the 11.5 nvcc → `nvcc fatal : Unsupported gpu architecture 'compute_89'` (11.5 predates Ada). `/usr/local/cuda` symlink → `/etc/alternatives/cuda`.
- **Fix:** pass `-DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.8/bin/nvcc` explicitly (and export `CUDACXX`). Verified 12.8 nvcc accepts both `compute_89` and `compute_120` standalone.

### smpbo fallback patch (#23385) — APPLIED
- Master `657e011` still assigns `info.devices[id].smpbo = prop.sharedMemPerBlockOptin;` at `ggml/src/ggml-cuda/ggml-cuda.cu:291` with **no guard**. No fix had landed.
- Patched: after the assignment, if `smpbo == 0 || smpbo > (1<<20)` → warn (`#23385`) and fall back to `prop.sharedMemPerBlock`. Only fires on implausible values, so the Ada 4090 path is untouched. ggml commit reports `657e011-dirty` (confirms patch present).
- On this driver (591.86) the fallback did **not** warn during enumeration — reported smpbo appears valid here. Kept the guard as insurance; will watch Server B at M3.

### Build config + gates
- `cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.8/bin/nvcc -DCMAKE_CUDA_ARCHITECTURES="89;120" -DGGML_CUDA_FORCE_CUBLAS=OFF -DCUDAToolkit_ROOT=/usr/local/cuda-12.8 -DBUILD_SHARED_LIBS=OFF -DCMAKE_BUILD_TYPE=Release`
- CMake auto-replaced `120` → `120a` (Blackwell). NCCL-not-found warning irrelevant (single-GPU servers).
- Build: `cmake --build build -j 8 --target llama-cli llama-mtmd-cli llama-server llama-gguf-split`. Peak memory fine (~4.6 GiB used, 41 GiB free at -j 8) — no need to drop to -j 4. Completed exit 0 in ~6 min.
- **Cache gate PASS:** `GGML_CUDA_FORCE_CUBLAS:BOOL=OFF`, `CMAKE_CUDA_COMPILER=/usr/local/cuda-12.8/bin/nvcc`, arch `89;120`. (`CUDAToolkit_VERSION` is not stored as a queryable cache entry in CMake 3.22; toolkit version pinned via the compiler path, nvcc = 12.8.93.)
- **Runtime gate PASS:** `llama-cli --list-devices` → CUDA0 RTX 4090, CUDA1 RTX 5060 Ti; `nvidia-smi` confirms cc **8.9** + **12.0**. `--help` confirms `--spec-type draft-mtp`.

### Canonical rebuild command (MUST pin the compiler — `CUDAToolkit_ROOT` alone is insufficient)
```
export PATH=/usr/local/cuda-12.8/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:$LD_LIBRARY_PATH
export CUDACXX=/usr/local/cuda-12.8/bin/nvcc
cd ~/ai/qwen36/llama.cpp && rm -rf build
cmake -B build -DGGML_CUDA=ON \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.8/bin/nvcc \   # <-- REQUIRED; stray /usr/bin/nvcc is CUDA 11.5
  -DCMAKE_CUDA_ARCHITECTURES="89;120" -DGGML_CUDA_FORCE_CUBLAS=OFF \
  -DCUDAToolkit_ROOT=/usr/local/cuda-12.8 -DBUILD_SHARED_LIBS=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build build -j 8 --target llama-cli llama-mtmd-cli llama-server llama-gguf-split
```

## Blackwell (sm_120 / RTX 5060 Ti) risk tracker — TWO SEPARATE OPEN BUGS
- **#23385 — smpbo/MMQ init-dispatch (CLEARED on this driver, patch kept as net):** garbage `sharedMemPerBlockOptin` → MMQ aborts at init. Fires at load time. Driver 591.86 reports a *valid* value here, so the compiled-in fallback patch stays dormant. Status: mitigated + inert.
- **#24399 — `mul_mat_q<Q8_0>` runtime device exception (UNTESTED, but NOT EXPOSED by our stack):** intermittent out-of-range shared-mem store during *decode*, only when Q8_0 dense tensors are present on sm_120. Completely separate from #23385; nothing has tested it. **M2 tensor scans show ZERO Q8_0 in Server B's weights (IQ4_XS and UD-Q3_K_XL) AND the F16 mmproj — so #24399 is not exposed on the 5060 Ti with this stack.** If Server B is ever moved to a quant containing Q8_0 dense tensors, this risk goes live → substitute Q6_K. Keep watching for intermittent decode-time device exceptions regardless.
- Server A's Q8_0 tensors (49 of them incl. the nextn MTP head) run on the Ada 4090 (sm_89), which is unaffected by either bug.

## M2 — GGUF header gates (read over HTTP range, zero GB downloaded)
- Arch string is **`qwen35`** (not qwen36). Config: head_count 24, **head_count_kv 4**, **key_length=value_length=256**, embedding 5120, block_count 64 (+1 nextn block on MTP repo), context_length **262144**, ssm.{inner 6144, state 128, group 16, conv 4}.
- **Layer topology (corrected):** 64 layers = **16 full-attention + 48 Gated DeltaNet**. The discriminator is `blk.N.ssm_conv1d.weight` (48 GDN layers). GDN layers *also* carry q/k/v projections, so counting attn tensors overcounts — only the 16 full-attn layers hold a growing KV cache; the 48 GDN layers hold a fixed ~150 MiB recurrent state.
- **KV formula:** per token = 16 layers × (K 4×256 + V 4×256) = 32768 elems/token. @131072: q8_0 ≈ 4.25 GiB, q4_0 ≈ 2.25 GiB. @32K: q4_0 ≈ 0.56 GiB.
- **Gate 3 PASS:** Server A `blk.64.nextn.eh_proj.weight` = **Q8_0** [10240,5120] (+ enorm/hnorm/shared_head_norm F32). MTP heads present & correct precision → draft path will load.
- **Gate 4 PASS:** `ssm_conv1d.weight` present on all 48 GDN layers in every probed model (no blk.40/blk.64 missing-tensor bug).
- **Gate 2:** Server B IQ4_XS types = F32/IQ4_XS/Q5_K/Q6_K/Q4_K (0 Q8_0); UD-Q3_K_XL types = F32/Q3_K/Q4_K/IQ4_XS/Q6_K/Q5_K/IQ3_S/IQ3_XXS (0 Q8_0); mmproj-F16 = clip arch, F32/F16 only (0 Q8_0). All clean.
- **Gate 1 (VRAM):** Server B **IQ4_XS fails** (0.68 GiB free < 1.5); **UD-Q3_K_XL passes** (1.59 GiB free). Decision: Server B = UD-Q3_K_XL @ 32K (q4_0 KV). Server A = q8_0 KV first @131072, empirical fallback. mmproj is a `clip` vision tower: 27 blocks, embed 1152, 16 heads.
- Downloads verified by sha256 vs HF LFS oid: UD-Q4_K_XL `4085665e…`, UD-Q3_K_XL `cff4a2da…`, mmproj-F16 `eacf610d…`.

### Flag corrections vs the setup prompt (verified against `llama-server --help`, b9996)
- **`-cpent` DOES NOT EXIST.** The prompt's `--no-context-shift -cpent 256 -ctxcp 32 --cache-reuse 256` mitigation maps to real flags: `--no-context-shift` (default already disabled), `-ctxcp/--ctx-checkpoints N` (default **32**), `-cms/--checkpoint-min-step N` (default 8192), `--cache-reuse N` (default **0** → MUST set, e.g. 256; "min chunk size to reuse from cache via KV shifting"). Dropped `-cpent`; kept `--ctx-checkpoints 32 --cache-reuse 256`.
- `--spec-draft-n-max` default is **3** (not the community-2 the prompt cited). Per user: A/B both 2 and 3 at M4. Scripts use `${SPEC_NMAX:-2}` as the initial value, overridable.
- `--image-min-tokens` default = "read from model"; set to 1024 explicitly (prompt's fine-detail fix).
- `-c` = `--ctx-size`, `-ctk/-ctv` = `--cache-type-k/v`, `-fa` = `--flash-attn`, `-np` = `--parallel`. All present.
- Template: pinned `qwen3.6-v19.jinja` (13203 B) downloaded; a newer top-level `chat_template.jinja` (v20+, 16289 B) also saved as `qwen3.6-latest.jinja` for optional A/B. v19 markers OK: `</think>`×11, `preserve_thinking`, tool_call×26, no `|items`/`|safe` (C++ crashers). Full validation at M3 via minja.
```
export PATH=/usr/local/cuda-12.8/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:$LD_LIBRARY_PATH
```
Always invoke the binary by absolute path: `~/ai/qwen36/llama.cpp/build/bin/llama-server`.

## M3 — servers + router + regression (14 assertions PASS / 0 FAIL; soak deferred to M4)
- **Deprecation migrated:** `--chat-template-kwargs '{"enable_thinking":...,"preserve_thinking":...}'` is deprecated in this build (warns). Migrated to **`--reasoning-preserve`**. Verified thinking still ON by default with v19 and extracted into `reasoning_content` (content stays clean). No enable_thinking flag needed.
- **`--cache-reuse` is UNSUPPORTED for this hybrid (GDN recurrent) context** → server disables it with a warning. Dropped the flag; kept `--ctx-checkpoints 32`. Prefix caching STILL works via per-request `cache_prompt:true` + checkpoints: test 8 measured **1859 ms → 117 ms**. (This supersedes the M2 note that kept `--cache-reuse 256`.)
- Boot ~7 s (mmap lazy paging). `common_fit_params: failed to fit ... n_gpu_layers already set to 99, abort` = informational (auto-fit skipped), not an error.
- **No Blackwell errors on Server B (sm_120):** clean boot, no `smpbo`/`mmq_x_best` abort (#23385 dormant), no decode exception (#24399 not exposed). Vision read 4/4 fine-text markers.
- **Tool-in-think bug NOT observed** on UD-Q4_K_XL: clean `{"city":"Paris"}` at temp 0.6 AND 0.2, no call inside `<think>`. No need to escalate to Q5_K_XL.
- **`jq` not installed** → static `jq-1.7.1` in `~/.local/bin` (no root). Put `~/.local/bin` on PATH to run regress.sh/bench.sh.
- **regress.sh harness fixes:** 100K prompt (~500 KB) exceeds shell ARG_MAX via `jq --arg` → build+POST in Python; filler ≈10 tok/sentence (not 7) → 10500 sentences ≈104408 tokens (<131072).
- Measured: Server A @131072 q8_0 = 23072 MiB (~1.46 GiB headroom); 104K prefill peak 23112 MiB (no OOM); MTP 83–94% acceptance, tg 88 t/s short / 45 t/s @104K, pp 1547 t/s @104K. Server B @32K = 15366 MiB (~0.92 GiB headroom — tight; drop to 24K if heavy images OOM).

## M4 — tuning + soak
- **MTP is a huge lever here:** vs `--spec-type none`, MTP-on (n=4) is **2.28× short-code, 2.65× long-code, 1.53× rag, 1.65× creative**. Baseline decode is only ~44 t/s; MTP brings code to 101–118 t/s. Net-positive on *every* workload — the decision-gate (disable if <40% accept or regresses) never triggered.
- **`--spec-draft-n-max` = 4 chosen.** Sweep: code tg peaks at n=4 (short 101, long 118 t/s), regresses at n=5; rag noisy (prefill-bound, 24-tok gens); creative declines monotonically with depth (not the priority). AL at n=4: 3.5–4.15. Aggregate acceptance drops with depth (far positions rejected). **llama.cpp exposes only aggregate acceptance + `mean len` (AL) per request — NOT per-position (only vLLM does).** AL formula used: `predicted_n/(predicted_n - draft_n_accepted)`.
- **KV q8_0 chosen over q4_0 on RETRIEVAL, not capacity.** Multi-needle @100K: **q8_0 5/5, q4_0 2/5** (3 needles → empty output even at 256 tokens — real degradation, not truncation). q4_0 code acceptance was actually fine (0.86) — the failure is long-range retrieval, exactly the structurally-unusual exposure from only 16/64 layers carrying KV. q4_0 saves ~1.75 GiB VRAM (23372→21324) but ≥100K quality matters more; headroom for q8_0 is there.
- **Router overhead negligible:** direct vs :8000 within run-to-run noise (~5 ms ttft, ±8% tg both directions). SSE streams unbuffered (1100 chunks).
- **Peak VRAM at final config (n=4, q8_0, 131072): 23372–23446 MiB → ~1.1–1.2 GiB headroom** (tighter than n=2's 1.46 GiB — bigger draft buffers at n=4, still ≥1 GiB).
- **Power:** 4090 already pinned at 450 W (=default=max). Not settable from WSL (`Insufficient Permissions`; WDDM needs Windows-side admin). No gain available (already unconstrained).
- **Server restart friction (important for ops):** launching a long-lived server from a foreground Bash tool via `nohup ... &`/`disown` gets SIGKILLed when the tool call returns. Must use the harness's background mechanism (or a helper that `exec`s the server). `restartA.sh` (in scratchpad) does `pkill -9 -f "llama-server.*--port 8080"` → wait for GPU0<500 MiB → `exec run-text-mtp.sh`, launched as a background job. `run-text-mtp.sh` now supports `MTP=on|off`, `SPEC_NMAX`, `KV`, `CTX` via env.

## Post-ship hardening
### Priority 1 — the 10/806 soak text errors: ROOT-CAUSED (not VRAM, not TDR)
- All 10 log lines identical: `500 "Failed to parse tool call arguments as JSON: ... missing closing quote; last read: '"ls -l /var/log'"`.
- **Cause: tool-call OUTPUT truncated by `max_tokens`.** The soak set `max_tokens:300` on tool turns; with thinking on, in sessions where reasoning ran long, thinking + the tool-call JSON exceeded the budget and the arguments were cut mid-string → llama.cpp 500s (when the cut lands after the tool-call open tag; elsewhere it returns `finish_reason=length` with partial args). Proven by a max_tokens sweep: mt=80 → `finish=length tool_args='{"cmd":"ls'` (truncated); mt=512 → clean `{"cmd":"ls -la /var/log"}`.
- **Ruled out:** VRAM (GPU0 dead-flat at 23446 MiB / free 1118 through the whole error window — no spike); WDDM TDR (zero `nvlddmkm` events in Windows Event Viewer); router timeout (soak hit :8080 directly); context-exceeded (`exceed: 0` in log, prompts tiny). **Not a VRAM problem → Priority 2 (VRAM levers) N/A; KV/ubatch/context/n_max left untouched.**
- **Fix:** router guard `_guard_tool_budget` — when `tools` are present, raise `max_tokens`/`max_completion_tokens` to a floor (`TOOL_MIN_TOKENS`, default 512; 0=off). Logs when it fires. Verified A/B: at mt=80, direct :8080 truncates, router :8000 returns valid `{"cmd":"ls -l /var/log"}`. Added as **regression test #13**.

### Priority 2 — N/A (the errors were not VRAM). KV stays q8_0 (locked; q4_0's 2/5 retrieval is disqualifying).

### Priority 3 — systemd: installed, and the failure that looked like a config bug wasn't
- **`After=` is ordering, NOT readiness.** systemd starts the router as soon as the backend units are *started*, not when llama-server has bound its port — so the router raced ahead and failed its first requests. Fixed with an `ExecStartPre` loop polling both backends' `/health` (2 s × 120 = 240 s cap). Confirmed working on cold boot: router came up **15 s** after the backends, not with them.
- **systemd does not inherit the interactive shell env** → `CUDA_VISIBLE_DEVICES` set explicitly in the unit (`Environment=`) *and* in the run script (`exec env`). Belt-and-suspenders; pinning held across a real boot.
- **The install-time flap was a stale-process conflict, not a unit bug.** After `install.sh`, all 3 units flapped in `auto-restart`. The three **manually-launched servers from the build session (PIDs 31594/45464/75904, up ~20 h)** still owned ports 8080/8081/8000 and all VRAM. Router: `[Errno 98] address already in use`. Text/vision: `llama_server: exiting due to HTTP server error`.
  - **Diagnostic that discriminated:** the text unit's journal showed **CUDA initialising cleanly, then the bind failure** → the systemd environment was already correct. A CUDA/env fault fails *earlier* and differently. Don't tune the unit when the log says the port is taken.
  - **The trap:** `verify-boot.sh` showed `:8000` answering "Paris" and both GPUs loaded *while the units were failing* — because the **stale** stack was serving. A passing endpoint check does not prove *your* units are the ones serving. Check `ppid`/PIDs, not just the port.
  - **`NRestarts` is cumulative, not current.** It read 105 long after the units were stable — scar tissue from the ~9 min conflict window (5 s retries). `ActiveEnterTimestamp` is the field that tells you whether it's flapping *now*.
- **Cold boot is CLEANER than the first install**, and is the real gate: manual processes cannot survive `wsl --shutdown`, so only the enabled units start, into an empty field. Verified 2026-07-15 14:57 — uptime 3 min, llama-server PIDs **270/271** `ppid=1`, **`NRestarts=0`** on all three, pinning held (23442/15366 MiB), `BOOT VERIFY: PASS` first try. Tool-budget guard confirmed live in the systemd-managed router.
- `systemctl is-system-running` = **`degraded` is expected here** and unrelated (a pre-existing non-qwen unit, flagged in M1 recon). Never gate on it; check the three units explicitly — which is why `verify-boot.sh` does.

---

# 35B-A3B (MoE) via llama-swap — 2026-07-15

## Inherit the fixes, RE-VERIFY the gates — and that applies to the TEST CODE too
The 27B's *config* fixes all carried (v19, `--reasoning-preserve`, mandatory system prompt,
`--repeat-penalty 1.0`, CUDA 12.8 pin, tool guard). The 27B's **test harness did not**, and every
assumption that silently didn't transfer produced a WRONG verdict:

1. **Bash quoting killed the canary.** `$PY -c '...'` with Python literals `'ok'`/`'RUNAWAY'` inside —
   the single quotes terminated bash's `-c` string, so the detector died with `NameError` and the
   gate reported FAIL without ever testing the model. → detectors live in **files** (`canary.py`), not inline.
2. **`content` vs `reasoning_content` — a VACUOUS PASS on the hard gate.** With `--reasoning-preserve`
   the model generated its full 600-token budget into `reasoning_content` and `content` was **EMPTY**
   (`finish_reason=length`, `predicted_n=600`, content len 0). The canary inspected a 1-char string and
   printed "ok". **A gate that cannot fail is worse than no gate — it manufactures confidence**, and it
   was sitting on the exact cross-GPU MTP path that was the one genuinely untested thing in the build.
   → canary now reads `reasoning_content + content`, AND self-tests against a synthetic runaway before
   any PASS is trusted. (Same bug class as the 27B's router-overhead test counting only `delta.content`.)
3. **Needle filler constant didn't transfer.** The 27B's 10500 sentences ≈ 104K tokens; the **35B
   tokenizes the same text at ~18 tok/sentence vs ~10** → 188408 tokens → legitimate 400
   `exceed_context_size_error`. → `needle35.py` calibrates against `/tokenize` instead of hardcoding.
4. **The naive digit-loop detector false-positives on correct output.** It flagged (a) the model
   correctly **echoing an ascending sequence that was in the prompt**, and (b) the model **counting
   words** to satisfy "write 400 words" — `The(1) moment(2) skin(3)...`, a coherent 93-long ascending
   run. → **gap discriminator**: a real digit-loop has nothing between the numbers (median gap 1-2
   chars); deliberate counting has words between (gap 8-9). Clean separation, validated in BOTH
   directions (fires on synthetic loops, silent on the 7 real word-count texts that fooled v1).
   → **Canary prompt rules:** no ascending integer run in the prompt, and never ask for "N words".

## Embedded chat template was STALE — dump it, don't trust it
The 35B is a separate conversion and ships a **different, smaller** template (8057 vs v19's 13203
chars) carrying **`| safe` on line 126 — in the tool-call argument rendering path**, the exact C++
Jinja crasher v19 removes. v19 has zero occurrences. → always `--chat-template-file` v19. Checked from
the **HTTP-range header probe before downloading 21 GB** — `tokenizer.chat_template` is in the metadata.

## Placement: the 5060 Ti is OVERFLOW STORAGE, not a second compute engine
The original spec said "fill GPU1 to ~1 GiB headroom". That reasons about **fit**; what matters is
**decode time**. Blocks are strictly sequential → total = `t_GPU0 + t_GPU1`, never `max()`. There is no
parallelism to harvest, so every byte on the slow card (448 GB/s vs 1008) is pure tax, plus a PCIe
round-trip + sync per spilled layer (no NVLink). **Spill the FEWEST layers that let the budget close.**
Measured, monotonically worsening, no flat region: 6 layers 206.1 → 10 layers 179.5 (−12.9%) →
16 layers 148.6 (−27.9%) t/s. Filling GPU1 would have shipped a ~15-20% slower model to use VRAM
there was no reason to use.
- **Spill only the cheap uniform layers.** UD quant bumped blk.34/38 to Q6_K (498 MiB) and blk.39 to
  562 MiB; moving a fat layer costs more bytes/token for zero benefit. Pool = plain-464 only.
  blk.40 (MTP block) stays whole on CUDA0 — the draft path round-trips enough already.
- **Expert-offload beats naive `--tensor-split` by ~18-21% at MATCHED GPU1 residency.** Too big to be
  bytes alone (the spill sweep prices tsplit's 21% extra GPU1 bytes at ~4%); the remaining ~16% is
  structural — tensor-split drags **attention and KV** onto the slow card, not just experts. Worst hit
  is rag-100k (104.7 vs 126.5), exactly where KV placement matters.
- **`-ts 1,0` not `-sm none`:** both CUDA backends must be initialized or the `-ot ...=CUDA1` target
  can't resolve; `-ts 1,0` weights all layers to GPU0 while keeping the CUDA1 backend live.
- **Verify residency at boot; don't trust the regex.** GPU1 = 2978 MiB ≈ 2784 spill + ~194 ctx proved
  `-ot` applied AND that `-ts` didn't distribute layers behind our back.

## #24399 is structurally dead on this model — because of WHAT the regex matches
Routed experts are **exclusively Q4_K/Q5_K/Q6_K**; every Q8_0 tensor (attn_qkv, attn_gate, ssm_out,
shexp, token_embd, output) is **non-expert** and stays on CUDA0. An `_exps`-only regex therefore puts
**zero Q8_0 on the Blackwell card**. This is stronger than "we avoided it" — but it holds *because of*
the regex, so residency is verified, not assumed.

## Decode is MEMORY-BANDWIDTH-bound — the MTP-off baseline is the proof
short-code 112.8, long-code 112.0, creative 112.7 — a **0.7% spread across completely different
content**. Content-independent decode speed is what pure bandwidth-binding looks like and is flatly
incompatible with compute-binding. (rag-100k 88.0 is the one deviation = KV reads growing with context.)
- Kills the prior "a MoE gains less from MTP (few active params → less to accelerate)" — that reasons
  about compute. Every token reads the same weight volume regardless of what it writes.
- **MTP ships ON:** 1.25-1.82× on every shape, worst acceptance 0.459 (gate floor 40%).
- **speedup ≈ AL/2**, holding across a 1.5× range of AL (1.80/1.82, 1.72/1.77, 1.18/1.25, 1.40/1.50):
  a verify step costs ~2× a decode step and returns AL tokens, because weights are read **once per
  verify step instead of once per token**.
- **NOT established:** that the amortization is MoE-*specific*. 586 MiB experts vs ~1958 MiB dense read
  per token → it mostly amortizes the dense path. Settling it needs dense-vs-MoE at matched bandwidth.
- **Baseline sanity that proves placement:** 112.8 = **2.55× the 27B dense's 44.3**, exactly what
  3B-active-vs-27B-dense predicts. A baseline BELOW 44 would have meant a misconfigured split.
- **n=3 ships (not the 27B's n=4).** The MoE peaks earlier; acceptance falls faster with depth
  (0.934 → 0.909 → 0.849). creative prefers n=3 by **+18%**, long-code by 3%, short-code is a wash.

## A short, deterministic generation is not a benchmark — it inflates MTP numbers
The rag shape predicted only **48 tokens** on a near-certain answer (`MERIDIAN-COBALT-7`). The model
echoing a deterministic string inflated **both** acceptance and tg. The tell was **non-monotonic
acceptance** (n=2 0.80 → n=3 0.75 → **n=4 0.92**) — the opposite of falls-with-depth. Re-measured with
a real 250-token prose generation at 99937 tokens: **tg 95.2 (n=3) / 107.0 (n=4), acceptance ~0.50
both** — not 126-158 t/s and not 0.75-0.92. Long-context RAG is far less predictable than a needle echo.
**Any MTP measurement needs a long, non-deterministic generation.**

## llama-swap (v240)
- **Schema drifts — verify it.** `groups` is **no longer top-level**: it moved to
  `routing.router.settings.groups`, and there are now two engines (`group` | `matrix`).
- **Mutual exclusion** = both models in ONE group with `swap: true` (+ `exclusive: true`). They must
  never share a `swap: false` group — the 27B needs GPU0 alone and the 35B needs GPU0+GPU1.
- **`cmd` points at the run scripts**, not duplicated flags — otherwise they drift. The scripts `exec`
  the binary, so the PID llama-swap tracks IS llama-server; SIGTERM reaches the real process instead of
  killing a wrapper and orphaning a server holding 21 GB.
- **Eviction is fully synchronous and correct** (the thing that had bitten 3 times). 200 ms VRAM trace:
  GPU0 23422 → **10 MiB at +0.72s** → 35B's first allocation **384 MiB at +0.95s**. Zero overlap,
  structural not lucky: llama-swap waits for **process exit**, and the driver reclaims VRAM when the
  CUDA context dies with the process. Effective margin ~22 s.
  **Pass condition is the TRACE, not the endpoint** — "the next model started" is exactly the assertion
  that stays green while a partial evict is present.
- **Swap latency is disk-bound:** 27B→35B 23.9 s cold / 4.9 s warm page cache; 35B→27B 10.0 s. 38 GB of
  weights vs 47 GB RAM → both cannot stay cached. **First request after any load is ~15% slow** (174.7
  vs 205 t/s) from graph warmup — do not mistake it for a swap regression.
- **llama-swap CANNOT express the tool-budget guard** → the custom router stays in the chain.
  `filters.setParams` is an **unconditional** set ("always runs for the model"), not a floor, and cannot
  condition on `tools`. `max_tokens: 512` there would cap EVERY request at 512, truncating long code and
  100K RAG answers. The guard needs `IF tools AND max_tokens < floor THEN raise` — conditional logic on
  the request body. Chain: `client -> router :8000 -> llama-swap :9000 -> llama-server`.
- **Aliases:** `qwen36` → **qwen-27b** deliberately (existing clients/regress.sh expect 27B behaviour);
  the 35B is opt-in by name. Aliases don't appear in `/v1/models` (`includeAliasesInList` default false)
  but they route.

## systemd, round two — the old assumptions became latent BOOT failures
- **`qwen-text` was left `enabled`.** llama-swap now owns those processes; an enabled qwen-text would
  start at boot, grab port 8080 + 23 GB of GPU0, and llama-swap's qwen-27b would fail to bind — the
  stale-process conflict again, but permanent and boot-triggered. → `install.sh` disables it,
  `qwen-swap.service` declares `Conflicts=`, `verify-boot.sh` asserts it stays disabled.
- **The router's readiness gate became wrong.** It waited for `:8080/health` AND `:8081/health` (both
  backends always resident). Under llama-swap `ttl:0` load-on-demand, **no model is loaded at boot** and
  those ports aren't listening → the gate would burn 240 s, fail, and flap forever. → wait on
  llama-swap's own `/v1/models`.
- **`verify-boot.sh`'s VRAM assertion became wrong.** `GPU0>18000 && GPU1>10000` assumed two resident
  servers. Now **an idle GPU is the CORRECT boot state**. → prove the stack by *exercising a swap* and
  asserting the 35B reaches min-spill residency (only reachable if the 27B was evicted first).
- **No `Environment=CUDA_VISIBLE_DEVICES` on qwen-swap.service** (unlike the old per-server units):
  pinning is now **per-model** (27B=GPU0, 35B=GPU0,1), so the run scripts own it. One unit-level value
  would be wrong for at least one model.
- **`KillMode=control-group`** stated explicitly because it is load-bearing: stopping llama-swap must
  also kill the llama-server it spawned, or an orphan keeps holding 21 GB and blocks the next load.

## Harness note (ops)
Long-lived servers launched from the Bash tool do **not** survive. `nohup &`/`disown` dies immediately
(exit 144); `run_in_background` only defers it — one server lived ~40 min through the full gate suite
and both benchmarks, then was SIGKILLed (log ended mid-stream with **zero errors**, the signature of an
external kill, not a crash). **Anything long-lived must be systemd/llama-swap-owned.** For sweeps, run
the server as a **child of a bounded script** (`sweep_one.sh`) so it cannot outlive the measurement and
squat VRAM. Also: `pkill -f "bin/llama-swap"` does NOT match — llama-swap's argv[0] is bare
`llama-swap` (PATH-resolved). And multi-line inline Bash commands can get their newlines collapsed —
put loops in a script file.

## Soak instrumentation traps (35B, 45 min)
- **RSS climb must be read as STEADY-STATE, not full-run.** The model is mmap'd, so RSS necessarily
  climbs as 21 GB of weights page in (+2422 MiB full-run; +3362 MiB in the first 75 s alone). That is
  page-in, **not** the MTP memory blowup the soak exists to detect — reporting it would be a false
  alarm generated by the harness. Steady-state (2nd-half) climb was **+8 MiB**.
- **An in-soak prefix-cache probe is contaminated by queue wait.** With `--parallel 1`, the probe
  queues behind the worker's in-flight request: probes swung 939–8573 ms with **no trend** (2nd half
  *faster* than the 1st), and the 8573 ms max matches a 1200-token canary turn at ~140 t/s. Measured on
  an **idle** server post-soak: 615 ms cold → **134–143 ms cached** with `prompt_n=4` of ~2600 tokens.
  → probe an idle server, or give the probe its own slot. A no-trend, high-variance series is the tell
  that you are measuring contention, not the thing you named.

---

# 122B-A10B (MoE, TRI-TIER: 4090 + 5060 Ti + system RAM) — 2026-07-15/16

## "Min-spill" is not a rule — the RULE is "minimise the SLOWEST TIER IN PLAY", and its direction flips
On the 35B, GPU1's alternative was the *faster* 4090 → min-spill starved GPU1. On the 122B the slow
tier is the **CPU** (DDR5 ~85 GB/s **plus i-quant dequant compute**) and GPU1 at 448 GB/s is the *fast*
alternative → **fill GPU1 to the brim to starve CPU**. Same principle, opposite conclusion. Copying the
35B's "min-spill GPU1" comment into a tri-tier config would be exactly backwards. The prompt that
commissioned this contained both instructions and they contradicted; the physics picks.

## `-ot` SPECIFIED TWICE SILENTLY DROPS ALL BUT THE LAST RULE
`W DEPRECATED: argument '-ot' specified multiple times, use comma-separated values instead (only last
value will be used)`. Two `-ot` flags = the CPU rule is **discarded**; layers 0-16 default to CUDA0 and
GPU0 needs ~29 GiB on a 24 GiB card -> OOM. The `--help` text said `<pattern>=<buftype>,...` (comma
separated) all along. **Use ONE comma-separated `-ot` argument.** The warning is easy to miss in a
48 GiB boot log.

## A quant's NAME is a weighted average, not a description
`UD-IQ3_S` is **~52% IQ2_S by expert bytes**: gate/up at **2-bit**, `down_exps` at IQ4_XS, and only 1.5%
actually IQ3_S. Attention/ssm/shexp are Q6_K. The allocation is deliberate — `ffn_down` writes back into
the residual stream (error-sensitive) while gate/up are more forgiving — but "IQ3" hides a 2-bit half.
**Read the tensor types, not the filename.** (Also: the HF web size shown was **part 2 of 3** — 46.55 GiB
of a 48.44 GiB model. Sum the parts.)

## CPU-tier cost is a CURVE, and MTP does not amortize it
- **Marginal cost of an expert layer on CPU nearly DOUBLES as the tier fills**: 0.459 ms (n_cpu 17→21)
  → 0.837 ms (21→25). The CPU saturates. CPU tax at n_cpu=17 ≈ 11 ms/tok of 24.7 (~45%); all-VRAM
  ceiling ≈ 60-66 t/s vs 40.5 measured.
- **llama.cpp exposes NO per-device timing** (llama-bench not built; `GGML_SCHED_DEBUG` = placement,
  not time; server timings are whole-graph). **Ablation is better than a counter would be**: a timer
  reports ELAPSED cpu time, but the useful number is the MARGINAL cost of a layer on CPU vs GPU —
  that x n_cpu IS the tax, and it back-computes the all-VRAM ceiling.
- **MTP does NOT amortize CPU work.** The tell: the CPU slope is the SAME with MTP on and off
  (**0.648 vs 0.682 ms/layer, ratio 0.95**). If it amortized, the ON slope would be ~AL/2 smaller.
  **CPU cost scales with TOKENS, not verify steps** = compute-bound. Correcting for it, the GPU-portion
  speedup is **2.22×** vs AL/2 = 2.77 — **the AL/2 law is intact where it applies; the CPU tier is the
  whole deviation** (raw speedup 1.70× only *looks* like the law broke).

## MTP SIGN-FLIPS on a CPU tier — a rejected draft is free on a GPU, expensive on a CPU
Creative: **n=6 gives 18.10 t/s vs 24.84 MTP-OFF — 27% SLOWER**. Acceptance collapses 0.47 → 0.36 → 0.24
with depth. On the all-VRAM 35B, creative at acceptance **0.459** still *gained* **1.25×**. Same
acceptance band, opposite sign. **Mechanism:** a rejected draft on a bandwidth-bound GPU is ~free (the
weights were read anyway); on a compute-bound CPU tier it is wasted work. **Deep drafting only pays
where acceptance is high.** → shipped **n_max=3** (the card said 6; 6 peaks short-code by 10% and costs
creative 26%). Do not copy an upstream card's n_max onto a CPU-offloaded config.

## n_max is CAPPED BY VRAM HEADROOM, and the error message lies about why
`n>=7` refuses to boot with **"failed to create MTP context"** — but the real cause two lines up is
`cudaMalloc failed: out of memory`: deeper draft buffers don't fit GPU0's 602 MiB. **The headroom
decision and the n_max ceiling are coupled.**

## RSS on a mmap'd CPU tier oscillates — use a REGRESSION, not an endpoint delta
The 35B taught "read RSS as steady-state, not full-run" (full-run climb is page-in). The 122B refines
it: the mmap'd CPU tier's pages are **continuously evicted and re-faulted**, so RSS genuinely oscillates
(**sd 533 MiB, spread 2680 MiB** around ~24.8 GiB). The naive 2nd-half **endpoint delta read +595 MiB**
and looked like a leak; the 2nd-half **regression slope is −218 MiB/hour** and the thirds show no trend.
**On a noisy series, endpoint-minus-endpoint is the wrong statistic.** (The 35B's +8 MiB was
unambiguous, so endpoint-delta happened to work there — which is why the flaw didn't surface.)

## The RAM tier is released by process lifetime, same as VRAM
Verified on the trace, not assumed: the 122B holds ~17-25 GiB RSS; swapping away dips it to **19-26 MiB**
before the next model allocates. Swap latency **40-49 s in / 17-21 s out** (48 GiB across 3 tiers),
disk-bound. **`healthCheckTimeout` must be raised (300 → 900)**: a COLD 122B load is far slower than the
~90 s warm case and 300 would time it out and look like a crash.

## A single global TOOL_MIN_TOKENS cannot serve models with 4x different verbosity
The 122B's reasoning is long AND variable — **4472 chars one run, 6647 the next on the SAME prompt**; at
`max_tokens=2048` it intermittently emitted its whole budget into `reasoning_content` and returned
**EMPTY content**. The 512 floor (tuned on the 27B/35B) would truncate its tool calls outright. →
`TOOL_MIN_BY_MODEL` per-model floors (122B: 4096). Verified live: `raised 80 -> 512` (27B) vs
`raised 80 -> 4096` (122B).

## verify-boot.sh must RETRY the endpoint, not assume it
`systemctl restart` returns when ExecStart is **spawned** (Type=simple), not when uvicorn has **bound**
:8000. Running verify immediately after a restart raced the bind by ~1s → false FAIL, and then a false
"no models registered" because that check hit the same dead socket. → 30s retry loop. (Wouldn't fire on
a real `wsl --shutdown` boot, but it made a healthy stack look broken.)

## A discriminating quality probe could not be built — and that is the finding
Across **73 items** in 5 categories over 2 difficulty tiers (up to 26-step chains, 8-person constraint
grids, nested-loop traces, 34-item compositions, 3-fact needles at 100K), **the 35B aced 93%**, leaving
**3 clean gap items** against a target of 30-40. Two more "failures" were **truncation artifacts**
(16000-token budget exhausted mid-trace) and were excluded — a truncated chain is wrong for the wrong
reason. **On the programmatically-verifiable problem space, the 35B is already at ceiling**, so
"checkable answers only" and "inside the 35B→122B gap" are **in tension**: escalating difficulty hit the
output-length wall before it opened a capability gap. Whatever edge a 122B holds lives where automatic
verification can't reach. → verdict **inconclusive**; Q4_K_XL **not downloaded**; IQ3_S ships on
**absence of evidence for harm, not evidence of absence**. See PROBE_RESULTS.md.

## Ops
- **HF downloads of 46 GB shards break mid-stream** (`ChunkedEncodingError: IncompleteRead` at 2.8 GB).
  Retry per part in a loop — `hf_hub_download` resumes from the `.incomplete`, so retries are cheap.
  The background task reported **exit 0 while the download had actually failed** — verify the bytes.
- **A dead llama-swap ORPHANS its llama-server**, which keeps holding 38 GiB of VRAM. This is exactly
  what `KillMode=control-group` prevents for the systemd unit (a bare process has no cgroup parent).
- `pkill -f "bin/llama-swap"` does NOT match — argv[0] is bare `llama-swap` (PATH-resolved).
- The 122B prefills at only **~400 t/s** (vs the 35B's ~3600), so a 100K rag rep costs ~250s. Exclude
  rag from sweeps and measure it once at the winner, or the sweep is all prefill.

## The suite that guards a model can rot silently — assert the MAPPING, not just the tests
The llama-swap re-architecture **orphaned `regress.sh`** (the 27B's requirement-#4 suite). It still
hardcoded `TEXT=:8080` / `VISION=:8081` as always-resident backends, but under llama-swap :8080 exists
only while the 27B is loaded and :8081 never exists. So [1] and [11] failed for ARCHITECTURAL reasons
and the 27B's canary / tool-in-think / thinking-closure / prefix-cache assertions **quietly stopped
running for two milestones**. `gates35.sh` and `gates122.sh` existed for the newer models, so nothing
looked missing.
**Nothing caught it because nothing asserted the mapping.** Fixed with `regress.sh [0] every model has
a live gate`: a `GATE_FOR` registry checked in BOTH directions against llama-swap's live `/v1/models` —
(1) every registered model must map to a gate file that exists and is executable, (2) every gate must
target a still-registered model (no dead gates). Self-tested in all three failure directions (orphaned
model / dead gate / missing file) — a gate that cannot fail manufactures confidence (see bug #2).
Also: `/upstream/<model_id>` is llama-swap's primitive for the direct-backend endpoints
(`/apply-template`, `/completion` timings) that the chat API cannot express — use it instead of a
hardcoded port, and load the model first.
Also: **delete retired tests, do not leave them as permanent SKIPs** — a SKIP that can never run is
noise that trains you to ignore the suite's output.
