#!/usr/bin/env python3
"""Digit-loop / MTP verify-loop runaway detector.

Distinguishes a REAL runaway from benign ascending integers.

The naive "longest ascending integer run" test produces false positives, badly. Observed on the
35B: asked for "400 words", the model does a word-count check in its reasoning --
    The(1) moment(2) skin(3) meets(4) wet(5) stone,(6) ...
which is a coherent, correct, 93-long ascending run. Likewise a prompt containing "1 2 3 ... 15"
makes the model's correct echo of it look like a runaway.

The discriminator is the GAP between consecutive numbers. A real digit-loop emits numbers with
nothing but punctuation between them ("1. 2. 3. 4."), gap ~2-3 chars. Deliberate counting has
words in between, gap >= ~5 chars. So: require BOTH a long ascending run AND a tight median gap.

Also flags single-token spam (the other runaway shape).
Exit 1 = runaway detected.
"""
import sys, re, statistics

ASC_MIN = 15      # ascending run length to be suspicious
GAP_MAX = 4.0     # median chars between consecutive numbers; <=4 means nothing but punctuation
REP_MIN = 30      # identical-token repetition
MIN_LEN = 200     # refuse to judge less than this -- see VACUOUS below

# VACUOUS-INPUT GUARD (exit 2), the lesson from the 35B's bug #2:
# with --reasoning-preserve the model can emit its whole budget into reasoning_content and leave
# `content` EMPTY. A caller reading only `content` handed this detector a 1-char string, and it
# dutifully printed "ok" -- a green light wired to nothing, sitting on the hard gate. A detector
# that cannot fail is worse than no detector: it manufactures confidence. So the detector itself
# now refuses input it could not have judged, rather than trusting every caller to check.

def analyse(t):
    toks = [(int(m.group(1)), m.start(), m.end()) for m in re.finditer(r"\b(\d+)\b", t)]
    best_n, best_i, asc, start = 1, 0, 1, 0
    for i in range(len(toks) - 1):
        if toks[i + 1][0] == toks[i][0] + 1:
            asc += 1
            if asc > best_n:
                best_n, best_i = asc, start
        else:
            asc, start = 1, i + 1
    gap = None
    if best_n >= 2:
        run = toks[best_i:best_i + best_n]
        gaps = [run[i + 1][1] - run[i][2] for i in range(len(run) - 1)]
        gap = statistics.median(gaps)
    words = t.split()
    rep = mrep = 1
    for a, b in zip(words, words[1:]):
        if a == b:
            rep += 1; mrep = max(mrep, rep)
        else:
            rep = 1
    return best_n, gap, mrep

def main():
    t = sys.stdin.read()
    if len(t.strip()) < MIN_LEN:
        print(f"VACUOUS len={len(t.strip())} < {MIN_LEN} — detector saw nothing to judge; "
              f"this is NOT a pass (are you reading reasoning_content + content?)")
        sys.exit(2)
    asc, gap, mrep = analyse(t)
    digit_loop = asc >= ASC_MIN and gap is not None and gap <= GAP_MAX
    spam = mrep >= REP_MIN
    bad = digit_loop or spam
    why = ""
    if digit_loop: why = " DIGIT-LOOP"
    elif spam:     why = " TOKEN-SPAM"
    elif asc >= ASC_MIN: why = f" (asc_run={asc} but gap={gap:.1f} -> deliberate counting, not a loop)"
    g = f"{gap:.1f}" if gap is not None else "-"
    print(f"{'RUNAWAY' if bad else 'ok'} asc_run={asc} gap={g} tok_rep={mrep} len={len(t)}{why}")
    sys.exit(1 if bad else 0)

if __name__ == "__main__":
    main()
