"""Near-duplicate plots inside one set of stories (block-ncm experiments/plot_dup.py, made importable).

Each story becomes a TF-IDF vector over content words (names and sentence-initial capitalised words dropped,
stop words removed, simple plural folding). A story has a "twin" if another story in the same set has cosine
similarity >= tau. Difference from plot_dup.py: IDF is computed within the evaluated set, not pooled over all
models, so values can differ slightly from block-ncm/RESULTS.md (real 1.2%, models 0.4% at tau 0.5).
"""
import math
import re
from collections import Counter

import numpy as np

STOP = set("""a an the and or but so if then than that this these those there here it its it's he she they them
his her hers their we us our you your i me my mine is was were are be been being am do did does done have has had
to of in on at by for with from up down out over under into onto off about as not no yes very too also just
can could will would should may might must shall said say says asked ask one two all some any every each more most
much many little big go went goes going come came get got make made see saw look looked want wanted like liked
know knew think thought day time once upon what when where who why how which while because after before again
back away now then too oh wow let lets let's s t don't didn't can't won't im i'm""".split())


def words(text):
    out = []
    for w in re.findall(r"[A-Za-z']+", text):
        if w[0].isupper() and w.lower() not in STOP:
            continue
        w = w.lower().strip("'")
        if len(w) > 2 and w not in STOP:
            out.append(w[:-1] if w.endswith('s') and len(w) > 4 else w)
    return out


def twin_share(texts, tau=0.5):
    docs = [words(t) for t in texts]
    if len(docs) < 2:
        return float("nan")
    df = Counter(w for d in docs for w in set(d))
    vocab = {w: i for i, w in enumerate(sorted(w for w, c in df.items() if c >= 2))}
    idf = np.array([math.log(len(docs) / df[w]) for w in sorted(vocab, key=vocab.get)])
    X = np.zeros((len(docs), len(vocab)), dtype=np.float32)
    for r, d in enumerate(docs):
        for w, c in Counter(d).items():
            if w in vocab:
                X[r, vocab[w]] = (1 + math.log(c)) * idf[vocab[w]]
    X /= np.linalg.norm(X, axis=1, keepdims=True) + 1e-9
    S = X @ X.T
    np.fill_diagonal(S, -1)
    return float((S.max(1) >= tau).mean())
