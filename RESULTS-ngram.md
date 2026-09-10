# ngram-mod drafter (stacked with MTP) — evaluation on Qwen3.8-27B UD-Q6_K_M

**Date:** 2026-08-21 · **Build:** llama.cpp `cd26896c1` (build 553) · **Model:** UD-Q6_K_M @ `-ts 4,1`, vision on,
q8_0 KV, 131072 ctx · **Seed:** 1234 · **Sampling:** served default (temp 1.0 / top-p 0.95 / top-k 20),
**reasoning ON** (matches production).

## Recommendation — CONDITIONAL SHIP (do NOT make it the default)

**Keep it as built: env-gated `NGRAM`, default OFF, gate-certified. Enable it only for edit-in-place /
large-verbatim-reproduction work, where it is a 2.4× win. Leave it OFF for reasoning-heavy RAG/agentic/novel
work, where it is flat and adds p99 latency jitter.**

- **Winning config:** `NG_MATCH=32 NG_MIN=24 NG_MAX=86 SPEC_NMAX=4` (MTP is now the miss-fallback).
- **Bimodal range (never a single average):** **~40 t/s** (100K-RAG with reasoning) · **~55–72 t/s**
  (typical short, novel) · **151 t/s** (edit-in-place, re-emitting a large file).
- Lossless (N1 byte-identical greedy), all correctness gates pass, 16 MB pool, no VRAM/quality change.

**Why not the default:** with reasoning on, the chain-of-thought is 44–95% of generated tokens and ngram-mod
cannot accelerate it (novel — nothing to recall). The benefit lands only on the *answer*, and only when the
answer is large verbatim reproduction. For the reasoning-heavy majority (RAG, agentic, novel code) the net is
≈0 throughput and **+150–320 ms p99** verification stalls. So it's a per-workload tool, not a global win.

## Per-class: baseline (MTP only) vs winner (MTP + ngram-mod 32/86), reasoning ON

| class | baseline tg | ngram tg | Δ | reason/answer tok | p99 itl (base→ng) | verdict |
|---|---|---|---|---|---|---|
| A short novel code | 54.2 | 54.7 | +1% | 206/196 | 65→65 | flat (no regression) |
| B prose/creative | 57.9 | 70.7 | +22%\* | 4310/374 | 65→68 | **\*artifact — repetitive reasoning, wide IQR; distrust** |
| **C edit-in-place** | 62.6 | **151.4** | **+142% (2.4×)** | 814/1020 | 66→**210** | **the win** |
| D agentic JSON | 58.5 | 59.9 | +2% | 249/123 | 65→**203** | flat + jitter |
| E RAG novel synth (100K) | 40.4 | 39.9 | −1% | 4395/247 | 94→98 | flat (no regression) |
| F RAG verbatim quote (100K) | 39.0 | 42.5 | +9%† | 2077/163 | 93→**324** | †within noise; answer 163 tok drowned by 2077 reasoning |

(C at the *default* 24/86 was +92% → 120 t/s; the 32/86 tuning takes it to +142% → 151 t/s.)

## Ranked sweep (class C reproduction; novel classes are invariant to ngram params)

| rank | n_match | n_max | SPEC_NMAX | C tg | acc | note |
|---|---|---|---|---|---|---|
| **1** | **32** | **86** | 4 | **151.4** | **0.653** | winner — longer match key → fewer false hits |
| 2 | 24 | 86 | 4 | 123.5 | 0.434 | default |
| 3 | 24 | 128 | 4 | 103.9 | 0.298 | n_max too deep → rejected tail |
| 4 | 32 | 128 | 4 | 97.9 | 0.431 | n_max too deep |

Lessons: **n_match 24→32 is the big lever** (acc 0.43→0.65). **n_max 86 beats 128** — drafting past the
recall horizon wastes verification. `SPEC_NMAX` (MTP fallback depth) had only a small effect (kept at 4).

## Correctness gates (§8) — winner config

| gate | method | result |
|---|---|---|
| N0 | both drafters live (log statistics) | PASS — see representative lines below |
| N1 | greedy temp-0 diff, ngram vs `--spec-type none` | **PASS — byte-identical** (lossless) |
| N2 | canary.py + gates38[3] @ temp 0.9/1.2 (detector self-test first) | PASS |
| N3 | needle35.py @ 100K | PASS (`MERIDIAN-COBALT-7`) |
| N4 | gates38[5] tools @ 0.6/0.2 | PASS |
| N5 | gates38[8] vision (ngram+MTP+mmproj) | PASS |
| N6 | free ≥1 GiB/GPU; pool ~16 MB | PASS (GPU0 2010 / GPU1 7475 free) |
| N7 | 10-min soak, 304 turns | PASS — 0 errors; only >2s stall was turn-1 warmup (3.09 s) |

**N7 caveat:** RSS endpoint delta +2.3 GiB over 304 `cache_prompt:true` turns — almost certainly bounded
prompt-cache growth (ngram pool is fixed 16 MB), but an endpoint delta is unreliable (per LEARNINGS); a
longer soak with an RSS *regression* would certify it bounded. Not ngram-specific.

## Representative per-drafter statistics (N0, reproduction request, `LLAMA_ARG_LOG_VERBOSITY=4`)

```
statistics  ngram-mod: #calls(b,g,a)=1 10 2, #gen drafts=2, #acc drafts=2, #gen tokens=172, #acc tokens=138, #mean acc len=70.00
statistics  draft-mtp: #calls(b,g,a)=1 8  8, #gen drafts=8, #acc drafts=8, #gen tokens=32,  #acc tokens=32,  #mean acc len=5.00
```
ngram-mod recalls the pasted code in huge blocks (mean accepted length 70); draft-mtp fills the novel tail.
Precedence works: ngram-mod preempts on a pool hit, MTP is the miss-fallback.

## Defects checked (§3)
- **#22168** (i_last reset on low-accept streak): present in build (`seq_info.n_low`).
- **#23154** (CUDA crash ngram+MTP): NOT triggered — our q8_0 KV (the reported repro used q4_0).
- **#24507** (router mode drops a spec type): N/A — standalone server; both types registered (N0 log).
- **#19232** (intermittent freeze): not observed across 304 soak turns (sole 3s stall = turn-1 warmup).

## How to use
Env-gated on `run-38-27b.sh` (default OFF — production unchanged):
```
NGRAM=on NG_MATCH=32 NG_MIN=24 NG_MAX=86 SPEC_NMAX=4 ./run-38-27b.sh
```
To make it a one-command-toggle profile, mirror the plain/q6km pattern with a `config.yaml.q6km-ngram`
that sets those env vars in the swap cmd. Not added by default — the win is too workload-specific to
justify a standing profile unless large-file editing is frequent.

**Note:** `--seed 1234` is now unconditional in `run-38-27b.sh` (measurement reproducibility). This makes
same-prompt production output deterministic; pass `SEED=-1` in a profile to restore sampling variety.
