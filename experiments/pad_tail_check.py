"""Why do rollouts match the real story much worse in chunk 4? (follow-up to alignment_check.py)

Hypothesis (written before the run): chunk 4 is where stories end. After EOS the real codes describe PAD, but
sampled codes there can be wrong, and the VQ-VAE decoder is a transformer over the whole chunk, so wrong codes in
the padding change the decoded words *before* the end. If so: chunks that are full (no EOS) match like chunks
1-3, chunks that end match worse, and copying the real codes back into the padded positions repairs them.
A level-t code j covers tokens [floor(j*C/l), ceil((j+1)*C/l)); it counts as padding if all of them are PAD.

    python experiments/pad_tail_check.py --keeps 5 10 --n 256
"""
import argparse
import math
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import p2  # noqa: E402,F401
from blockncm.fast import FastChunkedNCM  # noqa: E402
from blockncm.load import load_released  # noqa: E402
from p2.rollout import generate_forced  # noqa: E402
from p2.stories import heldout_windows  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--run", default="oursc_M4_G5_halo4")
ap.add_argument("--keeps", type=int, nargs="+", default=[5, 10])
ap.add_argument("--n", type=int, default=256)
ap.add_argument("--bs", type=int, default=16)
a = ap.parse_args()

st = torch.load(os.path.join(p2.RUNS, a.run, "model.pt"), map_location="cpu", weights_only=False)
G, halo = st["args"]["G"], st["args"].get("halo", 0)
cfg, tk, net, sd = load_released(os.path.join(p2.NCM_ROOT, "checkpoints", "tinystories"))
model = FastChunkedNCM(net, tk, sd, halo=halo).cuda().eval()
model.load_state_dict(st["model"], strict=True)
C, PAD = model.chunk_len, model.pad
real = torch.from_numpy(np.load(os.path.join(p2.NCM_ROOT, "data", "chunked", "heldout_M4.npy"))[:a.n].astype(np.int64))
M = real.shape[1]


@torch.no_grad()
def tokens(codes):
    B = codes.shape[0]
    _, zl = model.vq.idx_to_zhats(codes.reshape(B * M, -1).cuda(), return_zlevels=True)
    return model.vqvae.decoder(zl[-1].float()).argmax(-1).view(B, M, C).cpu()


real_tok = torch.cat([tokens(real[s:s + a.bs]) for s in range(0, a.n, a.bs)])
pad_tok = torch.from_numpy(heldout_windows(a.n, M) == PAD)  # (N, M, C) from source tokens, not decoded ones
# code-level padding mask over the raw code columns (level 0 is never padding)
cols = [torch.zeros(a.n, M, model.l0, dtype=torch.bool)]
for l in model.lt:
    m = torch.zeros(a.n, M, l, dtype=torch.bool)
    for j in range(l):
        lo, hi = math.floor(j * C / l), math.ceil((j + 1) * C / l)
        m[:, :, j] = pad_tok[:, :, lo:hi].all(-1)
    cols.append(m)
pad_code = torch.cat(cols, -1)  # (N, M, L) True = code covers only PAD

ends = pad_tok[:, M - 1].any(-1)  # chunk 4 contains the end of the story
print(f"{a.n} stories: chunk 4 ends the story in {int(ends.sum())}, is full in {int((~ends).sum())}", flush=True)
for temp in (1.0, 0.0):
    for keep in a.keeps:
        torch.manual_seed(0)
        kw = {"temperature": 1.0, "top_k": 1} if temp == 0 else {"temperature": temp}
        roll = torch.cat([generate_forced(model, real[s:s + a.bs].cuda(), G, keep, **kw).cpu()
                          for s in range(0, a.n, a.bs)])
        fixed = torch.where(pad_code, real, roll)  # real codes copied back into padded positions only
        t_roll = torch.cat([tokens(roll[s:s + a.bs]) for s in range(0, a.n, a.bs)])
        t_fix = torch.cat([tokens(fixed[s:s + a.bs]) for s in range(0, a.n, a.bs)])

        def exact(t, sel):
            num = den = 0
            for b in torch.nonzero(sel).flatten().tolist():
                p = ~pad_tok[b, M - 1]
                num += int((t[b, M - 1][p] == real_tok[b, M - 1][p]).sum()); den += int(p.sum())
            return num / max(1, den)

        c13 = float((t_roll[:, :M - 1] == real_tok[:, :M - 1]).float().mean())
        print(f"T={'greedy' if temp == 0 else temp} keep 1-{keep}: chunks 1-3 {c13:.3f} | chunk 4 full {exact(t_roll, ~ends):.3f}"
              f" | chunk 4 ending {exact(t_roll, ends):.3f} -> with real codes in the padding {exact(t_fix, ends):.3f}"
              f" | padded codes that differ from real: {float((roll != real)[pad_code].float().mean()):.3f}", flush=True)
