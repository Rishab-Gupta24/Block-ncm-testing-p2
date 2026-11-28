"""One evaluation for every Paper 2 run: scores stories already generated (no generation here).

Input: one or more *_samples.json files (block-ncm's sample_eval.py format, or any Paper 2 generator writing
the same format), plus real stories as a reference. For each source:

  genppl, genppl_sd   GPT-2 Large perplexity of the whole story, per sampling seed, then mean and sd
                      (block-ncm's sample_eval.py protocol: NCM repo's generation_perplexity, words pooled)
  judge2              the same with a second judge model (--judge2), so gains are not one model pleasing another
  entropy, distinct4  unigram entropy of BPE tokens; share of distinct 4-grams (repetition checks)
  seam_ratio          genPPL of 64-token windows across a chunk seam / windows inside a chunk
  mistakes            rule-based mistakes per 1,000 words (block-ncm error_types.py)
  consistency         who-is-who: names per story, late one-off names, protagonist kept / replaced
  plot_twin           share of stories with a near-duplicate plot in the same set (TF-IDF cosine >= 0.5)
  latency             seconds per story from the samples file (batched generation, so not batch-1 latency;
                      batch-1 timing stays with block-ncm time_gen.py)

    python -m p2.evaluate --M 4 --samples G5=~/ncm/runs/oursc_M4_G5_halo4/eval_M4_G5_t1.0_samples.json \
        B2=~/ncm/runs/b2c_M4_G11/eval_M4_G11_t1.0_samples.json --real valid --out ~/ncm/runs/p2/eval_m4.json
"""
import argparse
import json
import math
import os
import re
import sys
from collections import Counter

import numpy as np

from . import NCM_ROOT
from . import consistency, plots
from .stories import chunk_texts, decode, eos_pad, full_tokens, load_samples, real_stories

ap = argparse.ArgumentParser()
ap.add_argument("--samples", nargs="*", default=[], help="name=path pairs (path alone: name = file name)")
ap.add_argument("--M", type=int, required=True, help="chunks per story")
ap.add_argument("--real", choices=["valid", "train", "none"], default="valid",
                help="real reference stories: valid = held-out file (default), train = block-ncm's validation split")
ap.add_argument("--n-real", type=int, default=512)
ap.add_argument("--ref-model", default="gpt2-large")
ap.add_argument("--judge2", default="", help="second judge model id (empty = skip)")
ap.add_argument("--score-bs", type=int, default=4)
ap.add_argument("--window", type=int, default=32)
ap.add_argument("--cpu-only", action="store_true", help="skip the judge models (rule-based metrics only)")
ap.add_argument("--out", default="")
a = ap.parse_args()

EOS, PAD = eos_pad()
sources, b1 = {}, {}
if a.real != "none":
    sources[f"real ({a.real})"] = (None, real_stories(a.n_real, a.M, a.real), None)
for spec in a.samples:
    name, path = spec.split("=", 1) if "=" in spec else (os.path.basename(spec), spec)
    seeds, samples = load_samples(path)
    meta = json.load(open(os.path.expanduser(path)))
    lat = meta.get("latency")
    sources[name] = (seeds, samples, lat)
    if "latency_b1" in meta:  # measured at batch 1 by the generator (e.g. search_grid.py)
        b1[name] = meta["latency_b1"]

judges = {}
if not a.cpu_only:
    import torch
    sys.path.insert(0, NCM_ROOT)
    from evaluation.gen_ppl import generation_perplexity
    from transformers import AutoModelForCausalLM, AutoTokenizer

    def load_judge(mid):
        tok = AutoTokenizer.from_pretrained(mid)
        lm = AutoModelForCausalLM.from_pretrained(mid, torch_dtype=torch.float32 if "gpt2" in mid else torch.bfloat16)
        if "gpt2" not in mid:  # bf16 weights to fit 8 GB, but log-probs in fp32 so per-token NLL is not rounded
            lm.register_forward_hook(lambda m, i, o: o.__setitem__("logits", o.logits.float()) or o)
        return tok, lm.cuda().eval()

    judges["genppl"] = a.ref_model
    if a.judge2:
        judges["judge2"] = a.judge2


def ppl(judge, texts):
    texts = [t for t in texts if t.strip()]
    if not texts:
        return float("nan")
    tok, lm = judge
    return float(generation_perplexity(lm, tok, [""] * len(texts), texts, batch_size=a.score_bs).mean_ppl)


def entropy(ids):
    c, m = Counter(ids), len(ids)
    return -sum(v / m * math.log(v / m) for v in c.values()) if m else 0.0


def distinct4(ids):
    g = [tuple(ids[i:i + 4]) for i in range(len(ids) - 3)]
    return len(set(g)) / len(g) if g else 1.0


def seam_windows(samples, M, W):
    """Boundary and inner windows of 2W tokens (sample_eval.py's definition)."""
    bwin, iwin = [], []
    clean = lambda ids: [i for i in ids if i not in (EOS, PAD)]
    for _, ch in samples:
        flat, starts = [], []
        for c in ch:
            starts.append(len(flat))
            flat += [i for i in c if i != PAD]
        starts.append(len(flat))
        end = flat.index(EOS) if EOS in flat else len(flat)
        for m in range(M - 1):
            b = starts[m + 1]
            if b - W >= 0 and b + W <= end:
                bwin.append(decode(clean(flat[b - W:b + W])))
        for m in range(M):
            mid = (starts[m] + starts[m + 1]) // 2
            if mid - W >= 0 and mid + W <= end:
                iwin.append(decode(clean(flat[mid - W:mid + W])))
    return bwin, iwin


def mistakes(stories):
    from error_types import analyse
    r = analyse(stories)
    return {t: {"per_1k_words": 1000 * c / max(1, r["n_words"]), "share_of_stories": r["stories_with"][t] / r["n"]}
            for t, c in sorted(r["per_type"].items(), key=lambda x: -x[1])}


# ---- rule-based metrics (CPU)
res, texts_of = {}, {}
for name, (seeds, samples, lat) in sources.items():
    toks = [full_tokens(ch) for _, ch in samples]
    texts = [decode(t) for t in toks]
    per_chunk = [chunk_texts(ch) for _, ch in samples]
    cons, _ = consistency.summarise([chs for chs, _ in per_chunk], a.M)
    r = {
        "n": len(samples), "seeds": seeds,
        "entropy": float(np.mean([entropy(t) for t in toks if t])),
        "distinct4": float(np.mean([distinct4(t) for t in toks if len(t) > 3])),
        "mean_tokens": float(np.mean([len(t) for t in toks])),
        "frac_reaching_last_chunk": float(np.mean([len(t) > (a.M - 1) * 128 for t in toks])),
        "mistakes": mistakes(per_chunk),
        "consistency": cons,
        "plot_twin": plots.twin_share(texts[:512]),
    }
    if lat:
        r["latency_s_per_story"] = float(sum(t for t, _ in lat[1:]) / max(1, sum(b for _, b in lat[1:])))
        r["latency_batch"] = int(max(b for _, b in lat))  # batched generation, not batch-1 latency
    if name in b1:
        r["latency_b1"] = b1[name]
    res[name], texts_of[name] = r, (samples, texts, seeds)
    print(f"{name}: rule-based metrics done ({len(samples)} stories)", flush=True)

# ---- judge models (GPU), one model in memory at a time
for key, mid in judges.items():
    judge = load_judge(mid)
    for name, (samples, texts, seeds) in texts_of.items():
        groups = {s: [t for (s_, _), t in zip(samples, texts) if s_ == s] for s in (seeds or [-1])}
        vals = np.array([ppl(judge, g) for g in groups.values() if g])
        res[name][key] = float(vals.mean())
        res[name][key + "_per_seed"] = vals.tolist()
        if key == "genppl":
            res[name]["genppl_sd"] = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
            bwin, iwin = seam_windows(samples, a.M, a.window)
            res[name]["boundary_genppl"], res[name]["inner_genppl"] = ppl(judge, bwin), ppl(judge, iwin)
            res[name]["seam_ratio"] = res[name]["boundary_genppl"] / res[name]["inner_genppl"]
        print(f"{name}: {key} ({mid}) {res[name][key]:.2f}", flush=True)
    del judge
    torch.cuda.empty_cache()

# ---- summary table
cols = list(res)
rows = [("genPPL (GPT-2 L)", lambda r: r.get("genppl"), "{:.2f}"),
        ("judge 2", lambda r: r.get("judge2"), "{:.2f}"),
        ("seam ratio", lambda r: r.get("seam_ratio"), "{:.3f}"),
        ("entropy", lambda r: r["entropy"], "{:.3f}"),
        ("distinct-4", lambda r: r["distinct4"], "{:.3f}"),
        ("mean tokens", lambda r: r["mean_tokens"], "{:.0f}"),
        ("doubled words / 1k words", lambda r: r["mistakes"].get('doubled word ("the the")', {}).get("per_1k_words", 0), "{:.2f}"),
        ("names per story", lambda r: r["consistency"]["names_per_story"], "{:.2f}"),
        ("late one-off names / story", lambda r: r["consistency"]["late_one_off_per_story"], "{:.3f}"),
        ("protagonist kept", lambda r: r["consistency"]["protagonist_kept"], "{:.1%}"),
        ("protagonist replaced", lambda r: r["consistency"]["protagonist_replaced"], "{:.1%}"),
        ("plot twin >= 0.5", lambda r: r["plot_twin"], "{:.1%}"),
        ("seconds per story, batch 1", lambda r: r.get("latency_b1"), "{:.3f}"),
        ("seconds per story (batched generation; batch 1: time_gen.py)", lambda r: r.get("latency_s_per_story"), "{:.3f}")]
print("\n| | " + " | ".join(cols) + " |\n|---|" + "---|" * len(cols))
for label, f, fmt in rows:
    vals = [f(res[c]) for c in cols]
    if all(v is None for v in vals):
        continue
    print(f"| {label} | " + " | ".join("—" if v is None or (isinstance(v, float) and math.isnan(v)) else fmt.format(v)
                                     for v in vals) + " |")
if a.out:
    out = os.path.expanduser(a.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump({"args": vars(a), "results": res}, open(out, "w"), indent=1)
    print("->", out)
