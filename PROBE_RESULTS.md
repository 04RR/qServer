# Discriminating quality probe — 122B UD-IQ3_S vs 35B UD-Q4_K_XL

**VERDICT: SET NOT DISCRIMINATING — INCONCLUSIVE.**
The probe could not be built. Not because the items were bad, but because **the 35B does not fail
enough programmatically-checkable problems to calibrate a gap against.** Consequently the
IQ3-vs-Q4 question could not be settled by this method, and **the 68 GB UD-Q4_K_XL download was
NOT performed** — there is nothing to score it on that would mean anything.

Date 2026-07-15. Method per brief: computed ground truth, calibrate on the 35B, discard items the
35B aces, score the survivors 3-way.

## What was built

Ground truth is **computed by code, not hand-written** — each generator poses a problem and solves
it (brute force / direct execution), so difficulty is tunable without the author becoming the error
source. All trace items were independently re-executed to verify their keys before any scoring.

| tier | items | design |
|---|---|---|
| 1 | 42 | 8–12 step chains, 5–6 person constraint grids, code traces, 10–14 item compositions, 2-fact needles |
| 2 | 31 | 20–26 step chains, 7–8 person grids, **nested-loop traces w/ dict+list state**, 26–34 item compositions, **3-fact needles @ ~100K** |

## Calibration result — the 35B is the ceiling, not the floor

| tier | items | 35B aced | 35B failed | truncation artifacts | **clean gap items** |
|---|---|---|---|---|---|
| 1 | 42 | **42** | 0 | 0 | **0** |
| 2 | 31 | 26 | 5 | 2 | **3** |
| **total** | **73** | **68 (93%)** | 5 | 2 | **3** |

Target was 30–40 gap items. **Three** were found. By category, the 35B aced:
`chain 8/8 · sched 5/5 · needle3 4/4 · compose 7/8 · trace 2/6`.

**Only nested-loop code tracing produced any signal at all** — and 2 of its 5 failures were
**truncation artifacts**, not capability failures: the model exhausted its 16000-token budget
mid-trace (`H_trace_3` was 0/2 with `trunc=2`). Those were excluded rather than scored as wrong,
per the brief — a truncated chain is wrong for the wrong reason and poisons the result. That
leaves the "gap" in this category partly an **endurance** limit (simulating ~91 inner iterations),
not a reasoning-depth limit.

## The 3 gap items, scored (underpowered — reported for completeness, not as evidence)

| item | 35B UD-Q4_K_XL | 122B UD-IQ3_S |
|---|---|---|
| `H_trace_0` | 1/2 | 0/2 |
| `H_trace_4` | 0/2 | 0/2 |
| `H_comp34_3` | 1/2 | **2/2** |
| **total** | **2/6** | **2/6** |

The 122B wins one, ties one, loses one. **With 3 items this is statistically meaningless** and must
not be read as "the 122B is not better" — it is a set too small to detect anything. It is recorded
only so the numbers are not hidden.

## Why the probe failed — the actual finding

**The two constraints are in tension on this hardware:**
1. *checkable answers only* (no LLM-judge, no rater drift) — correctly demanded by the brief, and
2. *inside the 35B→122B capability gap*.

On the problem space that is **programmatically verifiable** — arithmetic chains, constraint grids,
code traces, compositional rules, multi-fact retrieval — **the 35B is already at ceiling.** It aced
93% of 73 items across two difficulty tiers, including 26-step dependent arithmetic, 8-person
constraint satisfaction with unique solutions, and 3-fact combination at 100K context.

Escalating difficulty did not open a gap; it hit the **output-length wall** first. The items the 35B
finally missed were ones where *any* model must emit ~10k reasoning tokens to simulate the trace —
which measures endurance and budget, not the depth a 122B would supply.

So the honest conclusion is the one the brief anticipated: **the capability gap between these two
models is smaller than the benchmarks imply on the kinds of problems that can be constructed and
automatically checked.** Whatever advantage a 122B holds appears to live in domains that resist
automatic verification — open-ended reasoning, knowledge breadth, nuance, instruction subtlety —
exactly the domains excluded (rightly) to avoid rater drift.

**The IQ3-vs-Q4 question is therefore moot by this method**: there is no measurable capacity here to
preserve or lose. That is not the same as saying IQ3 is safe — it means this instrument cannot see
the difference, and a bigger/other instrument would be needed to.

## What IS known about IQ3_S quality

From the M2 probe (8 compounding-reasoning items, 3 samples, both models):

| | 122B UD-IQ3_S | 35B UD-Q4_K_XL |
|---|---|---|
| score | 24/24 | 24/24 |
| cross-sample inconsistency | none | none |

Plus, from this probe's 73-item pool run on the 35B and the 3 gap items on the 122B: **no
incoherence, no drift, no cross-sample inconsistency, no truncation-independent failures** observed
on the 122B at IQ3_S.

**Evidence status: absence of evidence for harm, not evidence of absence.** IQ3_S shows no
degradation on everything measurable here, and its `gate`/`up` at IQ2_S did not manifest as the
predicted logic/consistency drift on any item tested.

## Recommendation

**Ship UD-IQ3_S**, on the stated bar ("if IQ3_S's outputs are clean, ship it") — they are clean on
every check available. **Do not download UD-Q4_K_XL**: at 73.25 GiB it would also push ~37 GiB onto
the CPU tier (past the earlier ~28–30 GB estimate) and likely need a `.wslconfig` bump past 48 GB,
and there is no measurement it could inform. If a real-world task later shows the 122B underperforming
expectations, that task becomes the discriminating item — and *then* Q4_K_XL earns the download,
with a concrete thing to compare on.
