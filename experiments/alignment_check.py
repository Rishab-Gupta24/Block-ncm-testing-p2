"""Rollout alignment check (Phase 2, inference only).

Real held-out stories keep their codes for levels 1..keep; a trained model generates the rest. The rollout is
decoded and compared with the decoded real codes chunk by chunk, over the real story's non-PAD positions:

  exact    share of positions with the same token (position by position)
  edit     1 - Levenshtein(rollout, real) / len(real), PAD removed (tolerates words shifted by a position)
  edit - exact   large = the decoder shifts words; small = mistakes stay in place

Also: doubled words per 1,000 words in rollouts vs the real stories, so we know whether rollouts contain the
model's typical mistakes (the point of using them as training data).

Prediction (written before the run): at keep = G = 5, exact about 0.55-0.70 and edit about 0.65-0.80, edit
clearly above exact (shifts exist); doubled words in rollouts 4-8 per 1k (free samples: 7.8).

    python experiments/alignment_check.py --run oursc_M4_G5_halo4 --keeps 5 8 10 --n 256
"""
import argparse
import json
import os
import re
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import p2  # noqa: E402,F401  (puts block-ncm on the path)
from blockncm.fast import FastChunkedNCM  # noqa: E402
from blockncm.load import load_released  # noqa: E402
from p2.rollout import generate_forced, levenshtein  # noqa: E402
from p2.stories import decode, heldout_windows  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--run", default="oursc_M4_G5_halo4")
ap.add_argument("--keeps", type=int, nargs="+", default=[5, 8, 10])
ap.add_argument("--temps", type=float, nargs="+", default=[1.0, 0.0], help="0 = greedy (top-1)")
ap.add_argument("--n", type=int, default=256)
ap.add_argument("--bs", type=int, default=16)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out", default="~/ncm/runs/p2/alignment_check.json")
a = ap.parse_args()

R = os.path.join(p2.RUNS, a.run)
st = torch.load(os.path.join(R, "model.pt"), map_location="cpu", weights_only=False)
G, halo = st["args"]["G"], st["args"].get("halo", 0)
cfg, tk, net, sd = load_released(os.path.join(p2.NCM_ROOT, "checkpoints", "tinystories"))
model = FastChunkedNCM(net, tk, sd, halo=halo).cuda().eval()
model.load_state_dict(st["model"], strict=True)
C, PAD, EOS = model.chunk_len, model.pad, model.eos
real_codes = np.load(os.path.join(p2.NCM_ROOT, "data", "chunked", "heldout_M4.npy"))[:a.n].astype(np.int64)
M = real_codes.shape[1]


@torch.no_grad()
def tokens(codes):
    """(B, M, L) codes -> (B, M, C) token ids, the decode path of sample_eval.py."""
    B = codes.shape[0]
    _, zl = model.vq.idx_to_zhats(codes.reshape(B * M, -1), return_zlevels=True)
    return model.vqvae.decoder(zl[-1].float()).argmax(-1).view(B, M, C).cpu().numpy()


def doubled_per_1k(tok_rows):
    """Doubled words per 1,000 words over the real story's positions (source tokens up to its EOS; the
    decoder does not reliably output EOS or PAD, so the decoded text is cut where the real story ends)."""
    words = dup = 0
    for row, s in zip(tok_rows, src):
        keep = (s.reshape(-1) != PAD) & (s.reshape(-1) != EOS)
        ids = row.reshape(-1)[keep].tolist()
        text = decode(ids)
        words += len(re.findall(r"[A-Za-z']+", text))
        dup += len(re.findall(r"\b([A-Za-z']+)\s+\1\b", text, flags=re.I))
    return 1000 * dup / max(1, words)


real_tok = np.concatenate([tokens(torch.from_numpy(real_codes[s:s + a.bs]).cuda())
                           for s in range(0, a.n, a.bs)])
src = heldout_windows(a.n, M)  # story vs padding comes from the source tokens (decoded PAD is unreliable)
assert src.shape == real_tok.shape
print(f"decoded real codes vs source tokens, story positions: {float((real_tok == src)[src != PAD].mean()):.4f}; "
      f"at EOS positions: {float((real_tok == src)[src == EOS].mean()):.3f}; "
      f"PAD positions decoded as PAD: {float((real_tok == PAD)[src == PAD].mean()):.3f}")
# sanity check: forcing every level must reproduce the real codes exactly
chk = generate_forced(model, torch.from_numpy(real_codes[:4]).cuda(), G, keep=model.K)
assert torch.equal(chk.cpu(), torch.from_numpy(real_codes[:4])), "keep=K did not reproduce the real codes"

results = {"run": a.run, "G": G, "halo": halo, "n": a.n, "real_doubled_per_1k": doubled_per_1k(real_tok), "source_doubled_per_1k": doubled_per_1k(src), "rows": []}
print(f"{a.run} (G={G}, halo={halo}), {a.n} held-out stories; real doubled words "
      f"{results['real_doubled_per_1k']:.2f} per 1k", flush=True)
for temp in a.temps:
    for keep in a.keeps:
        torch.manual_seed(a.seed)
        roll = []
        for s in range(0, a.n, a.bs):
            c = torch.from_numpy(real_codes[s:s + a.bs]).cuda()
            kw = {"temperature": 1.0, "top_k": 1} if temp == 0 else {"temperature": temp}
            roll.append(tokens(generate_forced(model, c, G, keep, **kw)))
        roll = np.concatenate(roll)
        ex, ed, per_chunk = [], [], [[] for _ in range(M)]
        for b in range(a.n):
            for m in range(M):
                r, g = real_tok[b, m], roll[b, m]
                pos = src[b, m] != PAD  # story positions (EOS included), from the source tokens
                if pos.sum() < 8:  # chunk (almost) empty in the real story
                    continue
                e1 = float((r[pos] == g[pos]).mean())
                rn, gn = r[pos], g[pos]
                e2 = 1 - levenshtein(gn, rn) / len(rn)
                ex.append(e1); ed.append(e2); per_chunk[m].append((e1, e2))
        row = {"temp": temp, "keep": keep, "exact": float(np.mean(ex)), "edit": float(np.mean(ed)),
               "per_chunk": [[float(np.mean([p[0] for p in pc])), float(np.mean([p[1] for p in pc]))]
                             for pc in per_chunk if pc],
               "doubled_per_1k": doubled_per_1k(roll), "n_chunks": len(ex),
               "example": " ‖ ".join(decode([i for i in roll[0, m].tolist() if i not in (PAD, EOS)]) for m in range(M))}
        results["rows"].append(row)
        print(f"T={'greedy' if temp == 0 else temp} keep levels 1-{keep}: exact {row['exact']:.3f}  edit "
              f"{row['edit']:.3f}  (edit-exact {row['edit'] - row['exact']:+.3f})  doubled {row['doubled_per_1k']:.2f}/1k",
              flush=True)

out = os.path.expanduser(a.out)
os.makedirs(os.path.dirname(out), exist_ok=True)
json.dump(results, open(out, "w"), indent=1)
print("->", out)
