# Paper 2 results log

Newest first. Every number here is measured; predictions are written before each run and kept next to the result.
Paper 1 numbers and earlier experiments are in `external/block-ncm/RESULTS.md`. Model: the local Paper 1 run
G=5 + halo 4 (`oursc_M4_G5_halo4`) unless stated; real stories come from the held-out file (`valid_bpe_8192.bin`).

## 2026-11-28: Kaggle seed-2 search grid is invalid: the NCM codec changed on Hugging Face

The Kaggle search grid on seed 2 of G=5 + halo 4 (`kaggle/run_search.sh`) gave genPPL 191 with no search and 159
at 8 plans + K=8 (ar), entropy 3.25 and 30–35 doubled words per 1k words. Locally the no-search cell is 19.1
(entropy 4.35, 7.5 doubled words). These stories are broken, not misjudged. **None of those numbers are results.**

Diagnostic (`kaggle/diag_precision.sh`): Paper 1's own `sample_eval.py --fast` on the same model, 128 stories, seed 42:

| | fp16 (Paper 1's Kaggle setting) | fp32 |
|---|---|---|
| genPPL | 189.3 | 189.2 |
| entropy | 3.24 | 3.23 |

Paper 1 scored this model at 19.6 on Kaggle (2026-11-20). Precision is not the cause.

Cause: on 2026-11-23 HazyResearch pushed three commits to `hazyresearch/ncm-checkpoints` that replace the
TinyStories generator and its VQ-VAE codec ("Release TinyStories 30k trainer EMA with its paired 80k VQ-VAE").
Our Kaggle scripts downloaded `main`, so the Block-NCM model's codes were decoded by a codec it was not trained
with. The local copy matches the earlier revision `d342ce8` exactly (sha256 of `generator.pt` 0acaf4f7…, of
`tokenizer.pt` 76e26003…); `main` now has different files (901d7fa6…, 85ce688e…). `hazyresearch/ncm-datasets`
did not change. Kaggle also had torch 2.11.0+cu130 and transformers 5.20.2.

Fix: every download now pins `revision="d342ce8aaea58830eef3b8ad473664673e41f692"` (`kaggle/run_search.sh`,
`run_eval.sh`, `diag_precision.sh`, and block-ncm's Kaggle scripts and README). Not verified on Kaggle yet: the
pinned diagnostic should give about 19.6 again before the seed-2 grid is rerun. All local results and Paper 1's
Kaggle seeds used the earlier codec and stand.

## 2026-11-28: Second judge (Qwen2.5-1.5B) on the search grid (`p2/evaluate.py --ref-model Qwen/Qwen2.5-1.5B`)

The same 28 cells (512 stories each, local G5+h4), scored again by Qwen2.5-1.5B (bf16 weights, fp32 log-probs,
batch 1). Qwen has a different tokenizer (151k vocabulary, about 5% fewer tokens per story), is twice the size of
GPT-2 Large, and was trained on different data. Ratios are relative to the no-search cell under the same judge.

| cell | GPT-2 Large | Qwen2.5-1.5B | ratio GPT-2 L | ratio Qwen |
|---|---|---|---|---|
| no search (1 plan, K=1) | 19.11 | 18.36 | 1.000 | 1.000 |
| 8 plans, K=1 | 16.97 | 16.33 | 0.888 | 0.889 |
| 1 plan, K=8, self | 17.86 | 17.18 | 0.935 | 0.936 |
| 1 plan, K=8, ar | 15.44 | 14.77 | 0.808 | 0.804 |
| 8 plans, K=4, ar | 14.79 | 14.16 | 0.774 | 0.771 |
| 8 plans, K=8, self | 15.97 | 15.34 | 0.836 | 0.835 |
| 8 plans, K=8, ar | 14.12 | 13.49 | 0.739 | 0.735 |
| real held-out stories | 6.11 | 3.98 | — | — |

- **The search gains survive a second judge.** Over all 28 cells the two judges correlate at 0.9996 (Pearson;
  Spearman 0.999). Each cell's gain over no search agrees within 0.7 percentage points, and the ar selector's
  gain is the same size under Qwen (−26.5% at 8 plans + K=8, vs −26.1% under GPT-2 Large). So the gain is not
  GPT-2 Large happening to agree with TinyStories-33M, which picks the stories.
- The close match is real, not a scoring bug. 64 stories scored one at a time with each judge: the models are
  `GPT2LMHeadModel` and `Qwen2ForCausalLM`, with 423 vs 403 tokens per story, and per-story log-perplexity
  correlates at 0.956 (largest gap 0.20 nats). They differ story by story but agree on which settings are better.
- Qwen finds real stories far more likely than GPT-2 Large does (3.98 vs 6.11), while generated stories score only
  4% lower. TinyStories has been public since 2023 and is probably in Qwen's training data. So under Qwen the gap
  to real stories looks larger (13.49 vs 3.98, 3.4×; GPT-2 Large: 2.3×). Do not compare Qwen's absolute numbers
  with real stories.
- Seam ratios under Qwen are 0.01–0.04 higher than under GPT-2 Large (1.14–1.24 vs 1.11–1.20).
- What is left: both judges and the selector are AR language models. The blind side-by-side on about 100 story
  pairs (plan, Phase 1) is still needed before claiming the stories are better to read.

## 2026-11-27: Generation precision: bf16 explains the "local outlier" at 6 chunks (`experiments/position_interp.py --amp`)

Same script, same seeds (2 × 256), same batch size; only the autocast dtype during generation differs (decoding
to tokens is fp32 in both, as in Paper 1).

| | 6 chunks, bf16 | 6 chunks, fp32 | 4 chunks, bf16 | 4 chunks, fp32 |
|---|---|---|---|---|
| G5+h4 genPPL | **25.10** | **21.72** | 19.60 | 19.07 |
| G5+h4 doubled words / 1k | 22.2 | 13.0 | 7.98 | 7.51 |
| B2 genPPL | 33.63 | 30.94 | — | — |
| B2 doubled words / 1k | 43.9 | 34.5 | — | — |

- bf16 reproduces Paper 1's local 6-chunk numbers (G5+h4 25.19, B2 33.53; rescored with p2.evaluate, exact match),
  so the scorer is not the cause.
- **The local run's 25.2 at 6 chunks, which block-ncm's RESULTS.md called an unexplained outlier (Kaggle seeds:
  20.0 / 21.0 / 20.8, fp16 on T4), comes from bf16 generation.** In fp32 the same model gives 21.7, in line with
  the Kaggle seeds. The listed alternatives (local encoding, seed) are not needed.
- At 4 chunks the effect is small (−0.5, about seed noise, same direction as the search grid's fp32 19.11 vs
  Paper 1's 19.38). bf16 hurts most where the model is already off its training length.
- fp32 costs no time at batch 1 (search grid, fp32: 0.36–0.40 s vs Paper 1's 0.37–0.38 s under autocast).
- Paper 1's claims stand: its 6-chunk table uses the Kaggle 3-seed means, B2 still collapses in fp32 (30.9 vs 18.7
  at 4 chunks) and G5 still holds. Paper 1's text is frozen; this goes in its next version.
- **Paper 2 generates in fp32 from now on.** All Paper 2 numbers so far (search grid, position test) are fp32; the
  Phase 1 table rescored Paper 1's bf16 samples.
- Not tested: which operation is sensitive (a guess: attention or rotary angles over the longer global sequence);
  one seed pair per cell.

## 2026-11-27: Phase 2, position interpolation at 6 chunks (`experiments/position_interp.py`, inference only)

6 chunks, fp32, 2 seeds × 256, T=1.0. "fill" = chunk embeddings 5–8 set to chunk 4's; "interp" = cross-chunk
(global) positions multiplied by 4/6, so the plan of 6 chunks spans the positions of 4. The scale-1 wrapper gives
identical codes to the unwrapped model (checked for both runs).

| | base | fill | interp | fill + interp |
|---|---|---|---|---|
| G5+h4 genPPL | 21.72 | **20.48** | 20.90 | 20.63 |
| G5+h4 mean tokens | 474 | 464 | **621** | 591 |
| G5+h4 doubled words / 1k | 13.0 | 10.0 | 10.1 | 10.1 |
| G5+h4 entropy | 4.313 | 4.322 | 4.419 | 4.403 |
| B2 genPPL | 30.94 | **25.99** | 75.89 | 74.97 |
| B2 doubled words / 1k | 34.5 | 25.6 | 198.7 | 199.3 |

Prediction (written before the run): fill + interp G5+h4 about 21 from fill's 22.7 (bf16 numbers), B2 about 23 from
26.7; interp alone helps less than fill. Result:
- **Wrong for genPPL.** For G5+h4, interp adds nothing on top of fill (20.63 vs 20.48). The rest of the 6-chunk gap
  is not unseen positions. Mixed-length training (T1) stays the fix; T4 is dropped as a quality fix.
- **Wrong, badly, for B2:** interp destroys it (75.9, 199 doubled words per 1k). In B2 the word-level levels are
  global too, so squeezing their positions scrambles word order. Interpolation is only safe on short coarse levels.
- **New: interp changes story length.** Without it, 6-chunk stories end at about 470 tokens, the length of a
  4-chunk story; with it they run to 590–620 tokens and use the extra chunks, with higher entropy and the same
  genPPL. The position scale acts as a length control for the plan. Relevant to "ending on time" (plan slots,
  Phase 4); not pursued now.
- Fill confirms the earlier result in fp32: G5+h4 −1.2, B2 −5.0.
- Caveat: "protagonist kept" rises from 62% to 96% with interp only because stories reach the second half of the
  6 chunks; the metric depends on length at 6 chunks.

## 2026-11-27: Phase 2, search grid on the local G5+h4 run (`experiments/search_grid.py`, inference only)

Plan search (n plans, the most likely kept) × best-of-K, 2 seeds × 256 stories per cell, T=1.0, M=4. Selectors:
**ar** = lowest NLL under TinyStories-33M (a separate small AR model); **self** = highest mean log probability
per code under Block-NCM itself. Seconds per story: batch 1, RTX 5050, median of 10, generation + decoding (+ AR
scoring for "ar").

genPPL (GPT-2 Large), ar selector (K=1: no selection):

| plans \ K | 1 | 2 | 4 | 8 |
|---|---|---|---|---|
| 1 | 19.11 | 17.48 | 16.22 | 15.44 |
| 2 | 18.31 | 16.81 | 15.65 | 14.78 |
| 4 | 17.34 | 16.12 | 15.16 | 14.41 |
| 8 | 16.97 | 15.63 | 14.79 | **14.12** |

self selector: K=2/4/8 = 18.56/18.24/17.86 (1 plan), 16.55/16.28/15.97 (8 plans).

Seconds per story at batch 1 (two runs of the timing loop, this one and the run before it):

| plans \ K | 1 | 2 | 4 | 8 |
|---|---|---|---|---|
| 1 | 0.36–0.40 | 0.36–0.41 (+0.08 ar) | 0.52–0.53 (+0.09) | 0.95–0.96 (+0.12) |
| 2 | 0.31 (0.81 once, an outlier) | 0.36–0.38 | 0.58 | 1.09–1.10 |
| 4 | 0.34–0.35 | 0.41 | 0.71 | 1.38–1.41 |
| 8 | 0.34 | 0.53–0.55 | 1.01–1.02 | 2.05–2.07 |

Best quality for the time (ar selector unless noted):

| seconds | best cell | genPPL | vs AR (≈6.3 s) | doubled / 1k | entropy | plot twin ≥ 0.5 |
|---|---|---|---|---|---|---|
| 0.36 | no search | 19.11 | 17× | 7.51 | 4.350 | 0.0% |
| 0.34 | 8 plans, K=1 | **16.97** | 18× | 7.41 | 4.311 | 2.0% |
| 0.55 | 8 plans, K=2, self | 16.55 | 11× | 7.40 | 4.315 | 3.1% |
| 0.64 | 8 plans, K=2 | 15.63 | 10× | 6.71 | 4.294 | 3.1% |
| 0.80 | 4 plans, K=4 | 15.16 | 8× | 6.12 | 4.289 | 2.1% |
| 1.11 | 8 plans, K=4 | 14.79 | 5.7× | 6.13 | 4.281 | 4.3% |
| 1.53 | 4 plans, K=8 | 14.41 | 4.1× | 5.71 | 4.281 | 0.8% |
| 2.17 | 8 plans, K=8 | 14.12 | 2.9× | 5.44 | 4.272 | 4.1% |

(Real held-out stories: 6.11. Paper 1: AR 9.9. The no-search cell, 19.11, matches Paper 1's 19.38 within seed
noise; the stories differ because the batch size changes the random draws.)

Prediction (written before the run) vs result:
- ar, 8 plans, K=8: "about 15.5" → **14.12**, better than predicted; the two kinds of search mostly add up
  (plans alone −2.1, K=8 alone −3.7, both −5.0 of a summed −5.8).
- self selector "about 17" → 15.97 at 8 plans / K=8; self selection on its own is weak (K=8, 1 plan: −1.3 vs
  −3.7 for ar). Most of self's gain comes from the plans, which are also chosen by Block-NCM's own likelihood.
- entropy drop "at most 0.05" → 0.078 for ar (4.350 → 4.272), 0.025 for self. Slightly over the bound.
- batch-1 time "about 0.6 s" → **2.07 s. Wrong.** Best-of-K is not close to free on this GPU: K=2 costs nothing,
  but K=4 is 1.4× and K=8 2.6×. Plan search is free (8 plans at K=1: 0.34 s), because the plan levels are short.

What this means:
- **Plan search with 8 plans is a free gain of about 2 genPPL** (19.1 → 17.0 at the same time). It should be the
  default from now on and the baseline every Paper 2 method is compared against.
- **At the 5×-faster-than-AR limit (≈1.26 s) the bar is 14.8** (8 plans, K=4, ar). A new method that costs time
  has to beat that at equal seconds.
- The ar selector cuts doubled words (7.5 → 5.4 per 1k) because it reads the words; self does not (7.2–7.5).
- Consistency (protagonist replaced 0.2–2.0%) does not change with search; it was already low for G5+h4.
- Costs to watch: plot twins rise from 0–2% to 3–4% with many plans (real stories: 2.0%); ar picks slightly
  shorter stories (415 → 403 tokens), self picks longer ones (up to 459).
- Caveat (resolved 2026-11-28): ar picks with one AR language model and is judged by another (GPT-2 Large), so
  part of the gain could have been the two agreeing. A second judge, Qwen2.5-1.5B, gives the same gains within
  0.7 percentage points (see 2026-11-28). A blind side-by-side is still to come.
- One seed of the model (local run). The first Kaggle seed-2 grid was invalid (NCM codec changed on Hugging Face,
  see 2026-11-28); a rerun with the pinned revision is pending.

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
and "Well" from `"Well done"` (now excluded). Second judge: Qwen2.5-1.5B, run 2026-11-28 on the search grid.
