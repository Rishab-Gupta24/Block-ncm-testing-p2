# Block-NCM, Paper 2 (testing)

Working repository for the second Block-NCM paper: **better text from Block-NCM without losing its speed**.
Paper 1 showed that Block-NCM writes a 512-token story 16.5× faster than an autoregressive model at batch 1,
but its text is about 2× worse (genPPL 19.2 vs 9.9). Paper 2 asks how much of that gap can be closed while
staying at least 5× faster than AR.

The plan, with every measured number and what is still only an estimate, is in
[docs/paper2_plan.md](docs/paper2_plan.md). Nothing in it is built yet.

## How this repo relates to Paper 1

- The Paper 1 code lives in [block-ncm](https://github.com/Rishab-Gupta24/block-ncm) and is included here as a
  git submodule in `external/block-ncm`, pinned to one commit. Paper 2 builds on top of it; it is not copied.
- New Paper 2 code (correction step, rollouts, fake errors, test-time search) lives only in this repo.
- If a change to `blockncm/` is needed, it is made and tested here first. Only bug fixes that also apply to
  Paper 1, and pieces that become part of Block-NCM itself, go back to the block-ncm repo, as a new version.
  The block-ncm repo keeps matching Paper 1.

## Setup

```bash
git clone --recursive https://github.com/Rishab-Gupta24/Block-ncm-testing-p2.git
cd Block-ncm-testing-p2
# already cloned without --recursive:
git submodule update --init
```

Environment, released NCM checkpoint and data: as in the block-ncm README (WSL2, Python venv at
`~/venvs/ncm`, checkpoints and data under `~/ncm`). Checkpoints, datasets and generations never go into git.

## Layout

```
docs/paper2_plan.md     the plan: goal, measured baselines, what to add, phases with go / no-go gates
RESULTS.md              results log (measured numbers, predictions next to them)
p2/                     evaluation (evaluate.py), consistency metric, rollouts, story loading
experiments/            one script per experiment
external/block-ncm/     Paper 1 code (submodule, pinned)
```
