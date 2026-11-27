"""Test-time search for Block-NCM: generation that also returns each story's own likelihood.

`generate_scored` is block-ncm's FastChunkedNCM.generate (pinned commit, plan search included) plus the log
probability of every sampled code, so candidates can be ranked by the model itself (no outside judge).
"""
import torch

from blockncm.fast import vq_ctx


@torch.no_grad()
def generate_scored(model, B, M, G, temperature=1.0, n_plans=1, device="cuda"):
    """Returns codes (B, M, L) and the mean log probability per sampled code (B,), all levels, at the sampling
    temperature. n_plans > 1: as generate(), the most likely plan (levels 1..G) of n candidates is kept."""
    n = n_plans if 0 < G < model.K else 1
    Bn = B * n
    D = model.vqvae.embed_dim
    idx = model.sample_l0(Bn * M, device)
    z_hat = torch.zeros(Bn * M, model.chunk_len, D, device=device)
    out = [idx]
    pre = model._prefix_emb(Bn, device)
    past = model._prefix_past(pre, device) if G == 0 else None
    logp = torch.zeros(Bn, device=device)
    n_codes = 0
    for t, l in enumerate(model.lt):
        if t == G and n > 1:  # keep the most likely plan per story (same rule as generate)
            best = logp.view(B, n).argmax(1) + torch.arange(B, device=device) * n
            rows = (best[:, None] * M + torch.arange(M, device=device)).reshape(-1)
            out = [o[rows] for o in out]
            idx, z_hat = out[-1], z_hat[rows]
            past = [(k[best], v[best]) for k, v in past]
            pre, Bn, logp = pre[best], B, logp[best]
        with vq_ctx():
            zq = model.vq.embeddings[t](idx)
        with vq_ctx():
            z_hat, x = model.vq.get_next_autoregressive_input(t, z_hat, zq)
        cid = torch.arange(M, device=device).repeat_interleave(l)
        h = model._embed(x.reshape(Bn, M * l, D).to(pre.dtype), t, cid)
        if t < G:
            mask, rope = model._gen_layout(M, G, t, device)
            keep = t == G - 1 and G < model.K
            hs, kvs = model._run_cached(torch.cat([pre, h], 1), mask, rope, want_kv=keep)
            if keep:
                past = kvs
            logits = model.net.lm_heads[t](hs[:, model.P:]).float()
        else:
            logits = model.net.lm_heads[t](model._local_pass(h, t, M, past, device)).float()
        idx = model.net._sample_level_raw(logits, temperature, 0, 1.0).reshape(Bn * M, l)
        # global passes give (Bn, M*l, V), local passes (Bn*M, l, V); both are story-major, chunk-major
        lp = torch.log_softmax(logits / max(temperature, 1e-6), -1).reshape(Bn, M * l, -1)
        logp += lp.gather(-1, idx.view(Bn, M * l, 1)).view(Bn, -1).sum(1)
        n_codes += M * l
        out.append(idx)
    return torch.cat(out, 1).view(B, M, -1), logp / n_codes
