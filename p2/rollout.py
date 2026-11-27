"""Coarse-conditioned rollouts: keep a real story's codes for levels 1..keep, let a trained model generate the rest.

The draft follows the real story's plan but contains the model's own fine-level mistakes, which is the training
data planned for the correction step (docs/paper2_plan.md, Sec. 3c). `generate_forced` is block-ncm's
FastChunkedNCM.generate (pinned commit) with one change: after each forced level is sampled, the sample is
replaced by the real codes, so every later level is conditioned on the real coarse codes.
"""
import numpy as np
import torch

from blockncm.fast import vq_ctx  # noqa: F401  (same context manager generate uses)


@torch.no_grad()
def generate_forced(model, codes, G, keep, temperature=1.0, top_k=0, top_p=1.0):
    """codes: (B, M, L) real codes (level 0 + all target levels). Levels 0..keep come from `codes`
    (level 0 = model.l0 codes; target levels 1..keep = codes at offsets[0..keep-1]); levels keep+1..K are sampled.
    Returns (B, M, L) codes. keep = 0 forces only level 0; keep = K reproduces `codes`."""
    B, M, L = codes.shape
    dev = codes.device
    flat = codes.reshape(B * M, L)
    D = model.vqvae.embed_dim
    idx = flat[:, :model.l0]
    z_hat = torch.zeros(B * M, model.chunk_len, D, device=dev)
    out = [idx]
    pre = model._prefix_emb(B, dev)
    past = model._prefix_past(pre, dev) if G == 0 else None
    for t, l in enumerate(model.lt):
        with vq_ctx():
            zq = model.vq.embeddings[t](idx)
        with vq_ctx():
            z_hat, x = model.vq.get_next_autoregressive_input(t, z_hat, zq)
        cid = torch.arange(M, device=dev).repeat_interleave(l)
        h = model._embed(x.reshape(B, M * l, D).to(pre.dtype), t, cid)
        if t < G:
            mask, rope = model._gen_layout(M, G, t, dev)
            want = t == G - 1 and G < model.K
            hs, kvs = model._run_cached(torch.cat([pre, h], 1), mask, rope, want_kv=want)
            if want:
                past = kvs
            logits = model.net.lm_heads[t](hs[:, model.P:]).float()
        else:
            logits = model.net.lm_heads[t](model._local_pass(h, t, M, past, dev)).float()
        if t < keep:  # forced: the real codes of target level t + 1
            o = model.l0 + model.offsets[t]
            idx = flat[:, o:o + l]
        else:
            idx = model.net._sample_level_raw(logits, temperature, top_k, top_p).reshape(B * M, l)
        out.append(idx)
    return torch.cat(out, 1).view(B, M, -1)


def levenshtein(a, b):
    """Edit distance between two int sequences (numpy, one vectorised row at a time)."""
    a, b = np.asarray(a), np.asarray(b)
    if len(a) == 0 or len(b) == 0:
        return max(len(a), len(b))
    j = np.arange(len(b) + 1)
    prev = j.copy()
    for i in range(1, len(a) + 1):
        sub = prev[:-1] + (b != a[i - 1])            # substitution / match
        cur = np.empty_like(prev)
        cur[0] = i
        cur[1:] = np.minimum(sub, prev[1:] + 1)       # deletion
        cur = np.minimum.accumulate(cur - j) + j      # insertion: cur[k] = min over m <= k of cur[m] + (k - m)
        prev = cur
    return int(prev[-1])
