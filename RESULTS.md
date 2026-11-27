# Paper 2 results log

Newest first. Every number here is measured; predictions are written before each run and kept next to the result.
Paper 1 numbers and earlier experiments are in `external/block-ncm/RESULTS.md`. Model: the local Paper 1 run
G=5 + halo 4 (`oursc_M4_G5_halo4`) unless stated; real stories come from the held-out file (`valid_bpe_8192.bin`).

## 2026-11-27: Phase 2, where do doubled words come from? (`experiments/reroll_check.py`, inference only)

512 free stories; then each story keeps its own codes for levels 1..keep and the rest is generated again.

| re-generated levels | doubled words per 1k (free: 8.11) | original doubled words still there |
|---|---|---|
| 9-11, T=1.0 | 8.15 | 36% |
| 6-11, T=1.0 | 7.96 | 5% |
| 4-11, T=1.0 | 8.16 | 1% |
| 6-11, greedy | 8.37 | 7% |
| 4-11, greedy | 12.02 | 2% |

Prediction was "about half survive re-rolling levels 6-11". Wrong: **95% disappear, but new ones appear at the
same rate.** Doubled words are accidents of fine-level sampling, not fixed by the model's coarse codes. So:
- a correction step that re-predicts fine levels can reach them (they are not locked in by the plan);
- re-sampling alone cannot reduce them; something has to detect and fix them;
- greedy decoding from level 4 makes more of them (12.0), so lowering the temperature is not a fix.

## 2026-11-27: Phase 2, rollout alignment check (`experiments/alignment_check.py`, inference only)

256 held-out stories keep their real codes for levels 1..keep; G5+h4 generates the rest. Decoded rollouts are
compared with the decoded real codes over the real story's positions (from the source tokens; see the note below).

| | exact (position by position) | edit distance | edit − exact | doubled words per 1k |
|---|---|---|---|---|
| real codes, decoded | — | — | — | 0.32 |
| keep levels 1-5 (the plan), T=1.0 | 0.903 | 0.920 | +0.018 | 3.26 |
| keep 1-5, greedy | 0.914 | 0.931 | +0.016 | 3.10 |
| keep 1-8, T=1.0 | 0.992 | 0.992 | 0.000 | 0.69 |
| keep 1-10, T=1.0 | 0.998 | 0.998 | 0.000 | 0.46 |

Prediction was exact 0.55-0.70, edit 0.65-0.80, edit clearly above exact. Wrong: rollouts follow the real story
closely and words almost never shift. Decisions for the correction step (docs/paper2_plan.md, Sec. 3c):
- **Supervise in place.** A position-by-position target is fine; shifts cost at most 2 points.
- **Re-predicting fine levels does not move words**, so the fixed-length step cannot repair a missing word by
  shifting; fake errors stay in-place only, as planned.
- With the real plan, rollouts have 3.3 doubled words per 1k: 10× real text but about 40% of free generation
  (8.1). Rollout training data is therefore milder than the model's real mistakes; fake errors should make up
  the difference.

Measurement note: the decoder never outputs PAD (0% of padded positions decode as PAD) but does output EOS where
the story ends (100%). The first version of this check took padding from decoded tokens, so it scored the junk
after each story's end and showed a false drop in chunk 4 (0.60). A padding hypothesis tested next
(`experiments/pad_tail_check.py`: wrong codes in the padding damage the words before the end) was refuted:
77% of padded codes differ from the real ones, but copying the real codes back changes nothing (0.869 → 0.869).
Story positions now come from the source tokens (`p2.stories.heldout_windows`).

## 2026-11-27: Phase 1, evaluation script and consistency metric (`p2/evaluate.py`, `p2/consistency.py`)

**Reproduction check passed.** Rescoring the Paper 1 local runs' samples gives the numbers in block-ncm's results
files exactly: genPPL 24.15 / 21.02 / 19.38 / 19.02 (B0 / G3+h4 / G5+h4 / B2), seam ratio 1.524 / 1.186 /
1.166 / 1.170, entropy 4.503 / 4.406 / 4.357 / 4.360.

512 stories per source, M=4, T=1.0:

| | real (held-out) | B0 | G3+h4 | G5+h4 | B2 |
|---|---|---|---|---|---|
| genPPL (GPT-2 Large) | 6.11 | 24.15 | 21.02 | 19.38 | 19.02 |
| seam ratio | 1.020 | 1.524 | 1.186 | 1.166 | 1.170 |
| entropy | 4.453 | 4.503 | 4.406 | 4.357 | 4.360 |
| doubled words per 1k | 0.02 | 7.56 | 7.68 | 7.82 | 8.22 |
| names per story | 2.33 | 4.85 | 4.02 | 2.91 | 2.63 |
| late one-off names per story | 0.11 | 0.49 | 0.42 | 0.21 | 0.14 |
| protagonist kept | 97.8% | 43.8% | 66.8% | 92.0% | 97.3% |
| protagonist replaced | **0.0%** | **35.9%** | 15.0% | 1.8% | 0.2% |
| plot twin ≥ 0.5 | 2.0% | 0.0% | 0.8% | 0.4% | 1.2% |

**Gate passed:** the consistency metric separates real stories from B0 (protagonist replaced 0.0% vs 35.9%).
The protagonist is the most frequent name in chunk 1; "kept" = still named in the second half; "replaced" = gone
from the second half while a name new to the second half is used there at least twice. Hand check
(`~/ncm/runs/p2/consistency_read.txt`): every "replaced" story read is a real switch (G5+h4 #413 Lily → Sara,
#84 Anna → Sara, B0 #305 Tom → Anna/Ben). "Kept" is lenient (B0 #100 keeps its protagonist while a stray name
appears mid-story), so "replaced" is a lower bound on name mix-ups. False names seen: garbled words ("Cakech")
and "Well" from `"Well done"` (now excluded). Second judge: not run yet (needs a model download).
