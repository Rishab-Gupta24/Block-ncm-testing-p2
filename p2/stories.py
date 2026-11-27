"""Stories as per-chunk token lists: model samples (sample_eval.py's *_samples.json) and real stories.

A story is a list of M chunks of C=128 BPE ids (PAD and EOS included, as generated). Text helpers follow
block-ncm's sample_eval.py exactly, so scores here match the numbers in block-ncm/RESULTS.md.
"""
import json
import os
import random

import numpy as np

from . import NCM_ROOT

C = 128
_tok = None


def tokenizer():
    """The released NCM's BPE-8192 pre-tokenizer (CPU only)."""
    global _tok
    if _tok is None:
        import sys
        from omegaconf import OmegaConf
        sys.path.insert(0, NCM_ROOT)
        from tokenization.bpe_tokenizer import BPETokenizer
        ck = os.path.join(NCM_ROOT, "checkpoints", "tinystories")
        cfg = OmegaConf.load(f"{ck}/config.yaml").tokenizer.pre_tokenizer
        cfg.vocab_file_path = f"{ck}/bpe_vocab_8192.pickle"
        cfg.merges_file_path = f"{ck}/bpe_merges_8192.pickle"
        _tok = BPETokenizer.from_config(cfg)
    return _tok


def eos_pad():
    t = tokenizer()
    return t.eos_token_id, t.pad_token_id


def decode(ids):
    return tokenizer().decode(ids, skip_special_tokens=True)


def full_tokens(chunks):
    """All chunks joined, cut at the first EOS, PAD removed (sample_eval.py's full_tokens)."""
    EOS, PAD = eos_pad()
    flat = [i for c in chunks for i in c]
    if EOS in flat:
        flat = flat[:flat.index(EOS)]
    return [i for i in flat if i != PAD]


def chunk_texts(chunks):
    """Per-chunk texts up to the first EOS, and whether the story ended on its own (error_types.py)."""
    EOS, PAD = eos_pad()
    out, ended = [], False
    for c in chunks:
        ids = [i for i in c if i != PAD]
        if EOS in ids:
            ids, ended = ids[:ids.index(EOS)], True
        out.append(decode(ids))
        if ended:
            break
    return out, ended


def load_samples(path):
    """(seeds, list of (seed, chunks)) from a *_samples.json file."""
    d = json.load(open(os.path.expanduser(path)))
    return d.get("seeds"), [(s, ch) for s, ch in d["samples"]]


def heldout_windows(n, M=4):
    """(n, M, C) source token windows of heldout_M{M}.npy's first n stories, same order (prepare_chunks.py
    --source valid). Use these, not decoded codes, to tell story from padding: the decoder does not output PAD
    reliably (positions after EOS decode to arbitrary tokens)."""
    from blockncm.load import token_docs
    EOS, PAD = eos_pad()
    off, tok = token_docs(os.path.join(NCM_ROOT, "data", "tinystories"), "valid", EOS)
    elig = np.nonzero(np.diff(off) > (M - 1) * C)[0][:n]
    out = np.full((len(elig), M * C), PAD, dtype=np.int64)
    for r, i in enumerate(elig):
        ids = [int(x) for x in tok[off[i]:off[i + 1]]]
        if len(ids) < M * C and ids[-1] != EOS:
            ids.append(EOS)
        ids = ids[:M * C]
        out[r, :len(ids)] = ids
    return out.reshape(len(elig), M, C)


def real_stories(n=512, M=4, source="valid", seed=0):
    """n real stories longer than (M-1)*C tokens, cut into M chunks of C tokens like the training data.

    source="valid": valid_bpe_8192.bin, never seen by the released NCM or our runs (preferred).
    source="train": every 100th long story of the training file (block-ncm's validation split).
    Returned as (seed=-1, chunks) pairs, same format as load_samples.
    """
    from blockncm.load import token_docs
    EOS, PAD = eos_pad()
    off, tok = token_docs(os.path.join(NCM_ROOT, "data", "tinystories"), source, EOS)
    lengths = np.diff(off)
    elig = np.nonzero(lengths > (M - 1) * C)[0]
    if source == "train":
        elig = elig[elig % 100 == 0]
    out = []
    for i in elig:
        ids = [int(x) for x in tok[off[i]:off[i + 1]]][:M * C]
        ids = ids + [PAD] * (M * C - len(ids))
        out.append((-1, [ids[k:k + C] for k in range(0, M * C, C)]))
    random.Random(seed).shuffle(out)
    return out[:n]
