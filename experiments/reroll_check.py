"""Where do doubled words come from: the coarse codes or the fine ones? (Phase 2, inference only)

Generate free stories, then keep each story's own codes for levels 1..keep and generate the rest again. If doubled
words survive re-rolling the fine levels, they are fixed by the coarse codes, and a correction step that only
re-predicts fine levels cannot remove them. If they mostly disappear, they are sampling accidents at the fine
levels, which a fine-level correction step can fix.

Prediction (written before the run): with keep = 5 (re-roll levels 6-11), about half of the doubled words survive
(free 8 per 1k -> about 5); with keep = 3, about a third survive.

    python experiments/reroll_check.py --keeps 3 5 8 --n 512
"""
import argparse
import json
import os
import re
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import p2  # noqa: E402,F401
from blockncm.fast import FastChunkedNCM  # noqa: E402
from blockncm.load import load_released  # noqa: E402
from p2.rollout import generate_forced  # noqa: E402
from p2.stories import decode  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--run", default="oursc_M4_G5_halo4")
ap.add_argument("--keeps", type=int, nargs="+", default=[3, 5, 8])
ap.add_argument("--n", type=int, default=512)
ap.add_argument("--bs", type=int, default=32)
ap.add_argument("--M", type=int, default=4)
ap.add_argument("--out", default="~/ncm/runs/p2/reroll_check.json")
a = ap.parse_args()

st = torch.load(os.path.join(p2.RUNS, a.run, "model.pt"), map_location="cpu", weights_only=False)
G, halo = st["args"]["G"], st["args"].get("halo", 0)
cfg, tk, net, sd = load_released(os.path.join(p2.NCM_ROOT, "checkpoints", "tinystories"))
model = FastChunkedNCM(net, tk, sd, halo=halo).cuda().eval()
model.load_state_dict(st["model"], strict=True)
C, PAD, EOS, M = model.chunk_len, model.pad, model.eos, a.M
DUP = re.compile(r"\b([A-Za-z']+)\s+\1\b", flags=re.I)


@torch.no_grad()
def texts(codes):
    ids = []
    for s in range(0, codes.shape[0], a.bs):  # decode in batches (the decoder's logits are large)
        c = codes[s:s + a.bs]
        _, zl = model.vq.idx_to_zhats(c.reshape(c.shape[0] * M, -1), return_zlevels=True)
        ids += model.vqvae.decoder(zl[-1].float()).argmax(-1).view(c.shape[0], M * C).tolist()
    out = []
    for row in ids:
        row = row[:row.index(EOS)] if EOS in row else row
        out.append(decode([i for i in row if i != PAD]))
    return out


def dup_stats(ts):
    words = sum(len(re.findall(r"[A-Za-z']+", t)) for t in ts)
    found = [set((m.start(), m.group(1).lower()) for m in DUP.finditer(t)) for t in ts]
    return 1000 * sum(len(f) for f in found) / max(1, words), found


torch.manual_seed(0)
free = torch.cat([model.generate(min(a.bs, a.n - s), M, G, temperature=1.0) for s in range(0, a.n, a.bs)])
free_t = texts(free)
base, base_found = dup_stats(free_t)
res = {"run": a.run, "n": a.n, "free_doubled_per_1k": base, "rows": []}
print(f"{a.run}: {a.n} free stories, doubled words {base:.2f} per 1k ({sum(len(f) for f in base_found)} found)", flush=True)
for temp in (1.0, 0.0):
    for keep in a.keeps:
        torch.manual_seed(1)
        kw = {"temperature": 1.0, "top_k": 1} if temp == 0 else {"temperature": temp}
        rr = torch.cat([generate_forced(model, free[s:s + a.bs], G, keep, **kw) for s in range(0, a.n, a.bs)])
        rr_t = texts(rr)
        rate, found = dup_stats(rr_t)
        # the same doubled word (same word, within 20 characters of the old position) still there after re-rolling
        survive = sum(1 for f0, f1 in zip(base_found, found) for (p, w) in f0
                      if any(w == w1 and abs(p - p1) <= 20 for (p1, w1) in f1))
        n0 = sum(len(f) for f in base_found)
        row = {"temp": temp, "keep": keep, "doubled_per_1k": rate, "survived": survive / max(1, n0)}
        res["rows"].append(row)
        print(f"T={'greedy' if temp == 0 else temp} re-roll levels {keep + 1}-{model.K}: doubled {rate:.2f} per 1k; "
              f"{100 * row['survived']:.0f}% of the original doubled words survive", flush=True)
out = os.path.expanduser(a.out)
os.makedirs(os.path.dirname(out), exist_ok=True)
json.dump(res, open(out, "w"), indent=1)
print("->", out)
