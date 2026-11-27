"""Write stories with the consistency metric's verdicts for checking by hand.

For each source, picks stories at random within each verdict (replaced / kept / no protagonist), prints the
chunks separated by ‖ with the names found in each chunk, so a reader can mark the verdict right or wrong.

    python -m p2.consistency_read --M 4 --n 10 --samples B0=...json G5=...json --out ~/ncm/runs/p2/consistency_read.txt
"""
import argparse
import os
import random

from . import consistency
from .stories import chunk_texts, load_samples, real_stories

ap = argparse.ArgumentParser()
ap.add_argument("--samples", nargs="*", default=[])
ap.add_argument("--real", choices=["valid", "train", "none"], default="valid")
ap.add_argument("--M", type=int, default=4)
ap.add_argument("--n", type=int, default=10, help="stories per verdict per source")
ap.add_argument("--out", required=True)
a = ap.parse_args()

srcs = {}
if a.real != "none":
    srcs[f"real ({a.real})"] = real_stories(512, a.M, a.real)
for spec in a.samples:
    name, path = spec.split("=", 1)
    srcs[name] = load_samples(path)[1][:512]

rnd = random.Random(0)
out = os.path.expanduser(a.out)
os.makedirs(os.path.dirname(out), exist_ok=True)
with open(out, "w") as fh:
    for name, samples in srcs.items():
        stories = [chunk_texts(ch)[0] for _, ch in samples]
        _, rs = consistency.summarise(stories, a.M)
        for verdict in ("replaced", "kept"):
            idx = [i for i, r in enumerate(rs) if r[verdict]]
            for i in rnd.sample(idx, min(a.n, len(idx))):
                r, per = rs[i], consistency.story_names(stories[i])
                fh.write(f"=== {name} #{i}  verdict: {verdict}  protagonist: {r['protagonist']}  "
                         f"names per chunk: {[sorted(set(p)) for p in per]}\n")
                fh.write(" ‖ ".join(t.strip() for t in stories[i]) + "\n\n")
print("->", out)
