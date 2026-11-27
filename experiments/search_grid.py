"""Search grid (Phase 2): plan search {1,2,4,8} x best-of-K {1,2,4,8}, two selectors, seconds per story.

For each number of plan candidates n, 8 candidate stories are generated per story slot (2 seeds x 256 slots, as in
block-ncm's sample_eval.py). Best-of-K uses the first K candidates, so every K shares one pool. Selectors:
  self   the candidate the model itself finds most likely (mean log probability per code, no outside model)
  ar     lowest mean token NLL under TinyStories-33M (Paper 1's best-of-K scorer; a separate small AR model)
Each cell is written as a *_samples.json for p2.evaluate (GPT-2 Large genPPL, consistency, mistakes).
Seconds per story are measured at batch 1: one story's K candidates (and n plans each) run as one batch.

Prediction (written before the run): ar selector, n=8, K=8: genPPL 19.4 -> about 15.5; self selector weaker
(about 17); entropy drops by at most 0.05; batch-1 time for n=8, K=8 about 0.6 s.

    python experiments/search_grid.py --run oursc_M4_G5_halo4 --out-dir ~/ncm/runs/p2/search_g5
    python -m p2.evaluate --M 4 --real none --samples $(ls ~/ncm/runs/p2/search_g5/*_samples.json | sed 's/.*\\/\\(.*\\)_samples.json/\\1=&/')
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import p2  # noqa: E402,F401
from blockncm.fast import FastChunkedNCM  # noqa: E402
from blockncm.load import load_released  # noqa: E402
from p2.search import generate_scored  # noqa: E402
from p2.stories import decode, full_tokens  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--run", default="oursc_M4_G5_halo4", help="run dir name under ~/ncm/runs, or a path")
ap.add_argument("--M", type=int, default=4)
ap.add_argument("--plans", type=int, nargs="+", default=[1, 2, 4, 8])
ap.add_argument("--Ks", type=int, nargs="+", default=[1, 2, 4, 8])
ap.add_argument("--seeds", type=int, nargs="+", default=[42, 0])
ap.add_argument("--n", type=int, default=256, help="story slots per seed")
ap.add_argument("--rows", type=int, default=64, help="candidate stories x plans per generate call")
ap.add_argument("--temp", type=float, default=1.0)
ap.add_argument("--scorer", default="roneneldan/TinyStories-33M")
ap.add_argument("--scorer-tok", default="EleutherAI/gpt-neo-125M")
ap.add_argument("--timing-reps", type=int, default=5)
ap.add_argument("--out-dir", required=True)
a = ap.parse_args()

run_dir = a.run if os.path.isdir(os.path.expanduser(a.run)) else os.path.join(p2.RUNS, a.run)
st = torch.load(os.path.join(os.path.expanduser(run_dir), "model.pt"), map_location="cpu", weights_only=False)
G, halo = st["args"]["G"], st["args"].get("halo", 0)
cfg, tk, net, sd = load_released(os.path.join(p2.NCM_ROOT, "checkpoints", "tinystories"))
model = FastChunkedNCM(net, tk, sd, halo=halo).cuda().eval()
model.load_state_dict(st["model"], strict=True)
C, M, KMAX = model.chunk_len, a.M, max(a.Ks)
out_dir = os.path.expanduser(a.out_dir)
os.makedirs(out_dir, exist_ok=True)
print(f"{run_dir}: G={G} halo={halo}; plans {a.plans}, K {a.Ks}, seeds {a.seeds} x {a.n}", flush=True)


@torch.no_grad()
def to_tokens(codes):
    B = codes.shape[0]
    _, zl = model.vq.idx_to_zhats(codes.reshape(B * M, -1), return_zlevels=True)
    return model.vqvae.decoder(zl[-1].float()).argmax(-1).view(B, M, C).tolist()


# ---- batch-1 timing: one story, K candidates x n plans in one batch (generation only; the AR scorer is timed below)
timing = {}
for n in a.plans:
    for K in a.Ks:
        ts = []
        for r in range(a.timing_reps + 1):
            torch.cuda.synchronize(); t0 = time.time()
            codes, _ = generate_scored(model, K, M, G, a.temp, n_plans=n)
            to_tokens(codes)
            torch.cuda.synchronize()
            if r:  # first call is warm-up
                ts.append(time.time() - t0)
        timing[f"n{n}_K{K}"] = float(np.median(ts))
        print(f"  batch-1 time, plans {n}, K {K}: {timing[f'n{n}_K{K}']:.3f} s", flush=True)

# ---- candidate pools
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402
stok = AutoTokenizer.from_pretrained(a.scorer_tok)
slm = AutoModelForCausalLM.from_pretrained(a.scorer).cuda().eval()


@torch.no_grad()
def ar_nll(texts):
    out = []
    for i in range(0, len(texts), 32):
        enc = [[stok.eos_token_id] + stok(t)["input_ids"][:600] for t in texts[i:i + 32]]
        L = max(len(e) for e in enc)
        ids = torch.full((len(enc), L), stok.eos_token_id)
        mask = torch.zeros((len(enc), L))
        for j, e in enumerate(enc):
            ids[j, :len(e)] = torch.tensor(e); mask[j, :len(e)] = 1
        ids, mask = ids.cuda(), mask.cuda()
        lp = torch.nn.functional.cross_entropy(slm(ids, attention_mask=mask).logits[:, :-1].float().transpose(1, 2),
                                               ids[:, 1:], reduction="none")
        m = mask[:, 1:]
        mean = (lp * m).sum(1) / m.sum(1).clamp(min=1)
        out += mean.masked_fill(m.sum(1) == 0, float("inf")).tolist()  # an empty story is never picked
    return out


# AR scoring time for K candidates at batch 1 (added to the ar selector's seconds per story)
dummy = ["Once upon a time, there was a little girl named Lily. " * 20] * KMAX
ar_time = {}
for K in a.Ks:
    ts = []
    for r in range(a.timing_reps + 1):
        torch.cuda.synchronize(); t0 = time.time(); ar_nll(dummy[:K]); torch.cuda.synchronize()
        if r:
            ts.append(time.time() - t0)
    ar_time[K] = float(np.median(ts)) if K > 1 else 0.0

summary = {"run": run_dir, "G": G, "halo": halo, "timing_b1": timing, "ar_scoring_b1": ar_time, "cells": {}}
for n in a.plans:
    pool = []  # per seed: list over slots of KMAX candidates (tokens, self score)
    per = max(1, a.rows // (KMAX * n))  # story slots per generate call
    for seed in a.seeds:
        torch.manual_seed(seed)
        slots = []
        for s in range(0, a.n, per):
            B = min(per, a.n - s)
            codes, sc = generate_scored(model, B * KMAX, M, G, a.temp, n_plans=n)
            toks = to_tokens(codes)
            sc = sc.tolist()
            slots += [[(toks[b * KMAX + k], sc[b * KMAX + k]) for k in range(KMAX)] for b in range(B)]
        pool.append((seed, slots))
        print(f"plans {n}, seed {seed}: {len(slots)} slots x {KMAX} candidates", flush=True)
    flat = [c for _, slots in pool for slot in slots for c in slot]
    ar = ar_nll([decode(full_tokens(tok)) for tok, _ in flat])
    i = 0
    for seed_i, (seed, slots) in enumerate(pool):
        for slot in slots:
            for k in range(KMAX):
                slot[k] = (slot[k][0], slot[k][1], ar[i]); i += 1
    for K in a.Ks:
        for sel in (("self", "ar") if K > 1 else ("none",)):
            samples = []
            for seed, slots in pool:
                for slot in slots:
                    cand = slot[:K]
                    pick = 0 if sel == "none" else (max(range(K), key=lambda k: cand[k][1]) if sel == "self"
                                                    else min(range(K), key=lambda k: cand[k][2]))
                    samples.append((seed, cand[pick][0]))
            name = f"n{n}_K{K}_{sel}"
            secs = timing[f"n{n}_K{K}"] + (ar_time[K] if sel == "ar" else 0.0)
            json.dump({"seeds": a.seeds, "samples": samples, "latency_b1": secs, "run": run_dir, "G": G,
                       "n_plans": n, "K": K, "selector": sel},
                      open(os.path.join(out_dir, f"{name}_samples.json"), "w"))
            summary["cells"][name] = {"latency_b1": secs}
            print(f"  wrote {name} ({secs:.3f} s per story at batch 1)", flush=True)
json.dump(summary, open(os.path.join(out_dir, "search_summary.json"), "w"), indent=1)
print("->", out_dir)
