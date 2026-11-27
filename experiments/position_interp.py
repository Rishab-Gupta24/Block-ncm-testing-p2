"""Position interpolation test at 6 chunks (Phase 2, T4; inference only, no training).

Models trained on 4 chunks see rotary positions 0 .. 4*l-1 in the cross-chunk (global) levels. At 6 chunks the
global levels reach 6*l-1, positions never seen in training. Position interpolation squeezes them back into the
trained range: global positions are multiplied by M_train / M (4/6), so the last chunk ends where chunk 4 ended
in training. The local levels (own chunk, positions 0..l-1, halo -h..l+h-1) are unchanged; they see the plan's
keys at the squeezed positions, as they saw the 4-chunk plan in training.

Conditions (each on 2 seeds x 256 stories, T=1.0, written as *_samples.json for p2.evaluate --M 6):
  base         as trained
  fill         chunk embeddings 5..8 set to chunk 4's (the earlier test: G5+h4 25.2 -> 22.7, B2 33.5 -> 26.7)
  interp       positions squeezed, embeddings as trained
  fill_interp  both

Prediction (written before the run): the earlier test says about half of the collapse is the untrained
embeddings and "the rest is probably unseen positions". If so, fill_interp should get G5+h4 from 22.7 to about
21 (M=4 is 19.4) and B2 from 26.7 to about 23 (M=4 is 18.7); interp alone helps less than fill.

    python experiments/position_interp.py --runs oursc_M4_G5_halo4 b2c_M4_G11 --out-dir ~/ncm/runs/p2/posinterp
"""
import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import p2  # noqa: E402,F401
from blockncm.fast import FastChunkedNCM  # noqa: E402
from blockncm.load import load_released  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--runs", nargs="+", default=["oursc_M4_G5_halo4", "b2c_M4_G11"])
ap.add_argument("--M", type=int, default=6)
ap.add_argument("--conds", nargs="+", default=["base", "fill", "interp", "fill_interp"])
ap.add_argument("--seeds", type=int, nargs="+", default=[42, 0])
ap.add_argument("--n", type=int, default=256)
ap.add_argument("--bs", type=int, default=32)
ap.add_argument("--temp", type=float, default=1.0)
ap.add_argument("--out-dir", required=True)
a = ap.parse_args()
out_dir = os.path.expanduser(a.out_dir)
os.makedirs(out_dir, exist_ok=True)
cfg, tk, net, sd = load_released(os.path.join(p2.NCM_ROOT, "checkpoints", "tinystories"))


def squeeze_positions(model, scale):
    """Wrap model._gen_layout so global-level positions (and the plan seen by local steps) are multiplied by
    scale. The mask is unchanged; prefix and local positions are unchanged."""
    orig = model._gen_layout

    def gen_layout(M, G, t, device):
        mask, _ = orig(M, G, t, device)
        P, l = model.P, model.lt[t]
        pos = [torch.arange(P).float()]
        if t < G:
            pos.append(torch.arange(M * l).float() * scale)
        else:
            if 0 < G < model.K:
                pos.append(torch.arange(M * model.lt[G - 1]).float() * scale)
            pos.append(torch.arange(l).float())
        return mask, model._rope(torch.cat(pos), device)

    model._gen_layout = gen_layout


summary = {}
for run in a.runs:
    run_dir = run if os.path.isdir(os.path.expanduser(run)) else os.path.join(p2.RUNS, run)
    st = torch.load(os.path.join(os.path.expanduser(run_dir), "model.pt"), map_location="cpu", weights_only=False)
    G, halo, M_train = st["args"]["G"], st["args"].get("halo", 0), st["args"]["M"]
    name = os.path.basename(os.path.normpath(run_dir))
    for cond in a.conds:
        model = FastChunkedNCM(net, tk, sd, halo=halo).cuda().eval()
        model.load_state_dict(st["model"], strict=True)
        if "fill" in cond:
            with torch.no_grad():
                w = model.chunk_emb.weight
                w[M_train:] = w[M_train - 1]
        if "interp" in cond:
            squeeze_positions(model, M_train / a.M)
        C = model.chunk_len
        samples = []
        for seed in a.seeds:
            torch.manual_seed(seed)
            for s in range(0, a.n, a.bs):
                B = min(a.bs, a.n - s)
                with torch.no_grad():
                    codes = model.generate(B, a.M, G, temperature=a.temp)
                    _, zl = model.vq.idx_to_zhats(codes.reshape(B * a.M, -1), return_zlevels=True)
                    ids = model.vqvae.decoder(zl[-1].float()).argmax(-1).view(B, a.M, C).tolist()
                samples += [(seed, row) for row in ids]
        tag = f"{name}_{cond}"
        json.dump({"seeds": a.seeds, "samples": samples, "run": run_dir, "G": G, "M": a.M, "cond": cond,
                   "scale": M_train / a.M if "interp" in cond else 1.0},
                  open(os.path.join(out_dir, f"{tag}_samples.json"), "w"))
        summary[tag] = len(samples)
        print(f"{tag}: {len(samples)} stories (G={G}, halo={halo}, trained on M={M_train})", flush=True)
        del model
        torch.cuda.empty_cache()
json.dump(summary, open(os.path.join(out_dir, "posinterp_summary.json"), "w"), indent=1)
print("->", out_dir)
