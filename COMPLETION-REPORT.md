# ✅ MISSION COMPLETE — Qwen3.6-27B Local Inference Stack

**Date:** 2026-07-14 · **Host:** RTX 4090 + RTX 5060 Ti, WSL2, CUDA 12.8, llama.cpp `657e011` (b9996-era)

The stack is **live at `http://localhost:8000`** in its final tuned configuration. All four requirements met and verified empirically on this machine.

---

## Final configuration

| | Server A — text+MTP (RTX 4090) | Server B — vision (RTX 5060 Ti) |
|---|---|---|
| Model | UD-Q4_K_XL (MTP repo) | UD-Q3_K_XL (non-MTP) + mmproj-F16 |
| Context | **131072** @ q8_0 KV | 32768 @ q4_0 KV |
| MTP | **on, n_max=4** | off |
| VRAM | 23446 / 24564 MiB (1.1 GiB headroom) | 15444 / 16311 MiB |
| Port | 8080 | 8081 |

Router on **:8000** → image content routes to vision, else to text; mandatory Qwen system prompt injected when absent; SSE streamed unbuffered.

---

## Requirement scorecard (all measured here)

1. **Max speed** — MTP delivers **101–118 t/s** on code, **2.3–2.7× vs off**. ✓
2. **≥100K context** — 131072 held; **125,962-token** needle recall verified; no OOM. ✓
3. **Vision** — 4/4 fine-text read; 762 generations in the soak, zero failures. ✓
4. **Clean output** — digit-loop canary clean at n=4; tool calls clean at temp 0.6 & 0.2; q8_0 retrieval 5/5. ✓

---

## Key measured results

### MTP on vs off (tg t/s)
| shape | MTP off | MTP on (n=4) | speedup |
|---|---|---|---|
| short-code | 44.3 | 101.0 | **2.28×** |
| long-code | 44.6 | 118.0 | **2.65×** |
| rag-100k | 36.3 | 55.4 | 1.53× |
| creative | 44.4 | 73.2 | 1.65× |

### `--spec-draft-n-max` sweep → chose **n=4**
| shape | n=2 | n=3 | **n=4** | n=5 |
|---|---|---|---|---|
| short-code | 86.5 | 92.5 | **101.0** | 99.6 |
| long-code | 87.4 | 92.9 | **118.0** | 112.5 |

Code peaks at n=4; n=5 regresses. (llama.cpp exposes only aggregate acceptance + mean-len/AL per request, not per-position.)

### KV precision → chose **q8_0** (retrieval, not capacity)
Multi-needle retrieval at 100,081 tokens: **q8_0 = 5/5, q4_0 = 2/5** (q4_0 produces empty output on 3 needles — real degradation, not truncation). q4_0's code acceptance was fine (0.86); it's long-range retrieval that breaks — exactly the 16-of-64-layer KV exposure.

### Dual-GPU soak (50 min) — PASS
806 text turns + 762 vision generations, concurrent.
| assertion | result |
|---|---|
| RSS climb (MTP memory blowup) | **+31 MiB** over 50 min — no climb |
| VRAM0 / VRAM1 climb | **0 / 0 MiB** — dead steady |
| GPU1 #24399 device exception | **0 over 762 vision gens** |
| prefix reprefill (cache alive at end) | **109.1 → 109.7 ms** — no degradation |

Text errors: 10/806 (1.2%) in a single transient window, self-recovered. Vision errors: 0.

### Router overhead — negligible
direct :8080 ttft 332 ms / tg 76.7 · router :8000 ttft 327 ms / tg 82.8 — within run-to-run noise.

### Regression suite — 14 PASS / 0 FAIL (at shipped n=4 config)
load+GPU cc · coherence · **digit-loop canary** · thinking tags · preserve_thinking · **tool calls @0.6 & 0.2** · mid-convo system role · **prefix cache reuse** · **100K recall** · **vision fine-text** · router dispatch.

---

## Decisions that weren't in the prompt, and why
- **n_max=4** (not the prompt's 2, nor the default 3) — measured code peak.
- **q8_0 KV** — q4_0 fails long-range retrieval 2/5 vs 5/5; headroom for q8_0 was there.
- **MTP stays on for every workload** — even creative/RAG are net-faster; the <40% gate never triggered.
- **Both Blackwell bugs harmless for this stack** — #23385 dormant (valid smpbo on driver 591.86; patch kept as a net); **#24399 never fired across 762 vision generations** (F16 mmproj + zero Q8_0, as the tensor scan predicted).

## Contradictions with the prompt, resolved (full detail in LEARNINGS.md)
- `mmproj-*-Q8_0.gguf` doesn't exist → used **F16**.
- `-cpent` flag doesn't exist → `--ctx-checkpoints` / `--cache-reuse`.
- `--cache-reuse` unsupported on this hybrid model → prefix caching works anyway via `cache_prompt`.
- froggeric "vision+MTP works" means *loads*, not *accelerates*.
- A stray CUDA **11.5** `nvcc` hijacked CMake → pin `-DCMAKE_CUDA_COMPILER`.
- "digit-loop fixed in b9300+" was unverified — held empirically (regression test #3).
- `--chat-template-kwargs` deprecated → migrated to `--reasoning-preserve`.

---

## One action left for you — enable start-on-boot (needs sudo)
The three systemd units are written but not installed. Run:
```bash
sudo bash ~/ai/qwen36/systemd/install.sh
```
This copies + enables `qwen-text`, `qwen-vision`, `qwen-router` (router `After=` both servers). Until then, the servers run as background processes and will not survive a WSL restart.

---

## Deliverables (all in `~/ai/qwen36/`)
```
llama.cpp/                   # built, CUDA 12.8, sm_89 + sm_120, MMQ on, smpbo-patched
models/                      # 3 GGUFs (SHA256-verified) + mmproj-F16 + test-image.png
templates/qwen3.6-v19.jinja  # fixed chat template (+ qwen3.6-latest.jinja for A/B)
run-text-mtp.sh              # Server A (env-tunable: MTP / SPEC_NMAX / KV / CTX)
run-vision.sh                # Server B
router/                      # starlette/httpx proxy + isolated .venv
systemd/                     # 3 units + install.sh
bench.sh                     # tok/s, TTFT, MTP acceptance, VRAM
regress.sh                   # the 14-assertion regression suite
RESULTS.md                   # all measured numbers from this machine
LEARNINGS.md                 # every bug + fix + flag name + source URL
COMPLETION-REPORT.md         # this file
```
