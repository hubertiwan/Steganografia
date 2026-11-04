"""Text generation with a Hugging Face model, with or without Unigram-Watermark."""
import numpy as np
import torch


def next_token_distribution(logits, wm=None, temperature=1.0, top_p=0.9):
    """Same order as Hugging Face generate(), which the authors used:
    1) watermark: +delta to green logits (logits processor),
    2) temperature, 3) nucleus (top-p) sampling, the paper's default decoding."""
    logits = logits.float().cpu().numpy().astype(np.float64)
    if wm is not None:
        logits = wm.watermark_logits(logits)
    logits = logits / temperature
    p = np.exp(logits - logits.max())
    p /= p.sum()
    if top_p is not None and top_p < 1.0:
        order = np.argsort(-p)
        cum = np.cumsum(p[order])
        keep = order[: int(np.searchsorted(cum, top_p) + 1)]  # smallest set with mass >= top_p
        q = np.zeros_like(p)
        q[keep] = p[keep]
        p = q / q.sum()
    return p


@torch.no_grad()
def generate(model, tokenizer, prompt, wm=None, max_new_tokens=200,
             temperature=1.0, top_p=0.9, seed=0, use_chat_template=True):
    """wm=None -> plain model (no watermark); wm=UnigramWatermark -> watermarked model.
    Returns (new_token_ids, decoded_text)."""
    rng = np.random.default_rng(seed)
    device = next(model.parameters()).device

    if use_chat_template and getattr(tokenizer, "chat_template", None):
        ids = tokenizer.apply_chat_template([{"role": "user", "content": prompt}],
                                            add_generation_prompt=True, return_tensors="pt")
        if not torch.is_tensor(ids):  # newer transformers versions return a dict
            ids = ids["input_ids"]
    else:
        ids = tokenizer(prompt, return_tensors="pt").input_ids
    ids = ids.to(device)

    new_tokens, past, inp = [], None, ids
    for _ in range(max_new_tokens):
        out = model(input_ids=inp, past_key_values=past, use_cache=True)
        past = out.past_key_values
        p = next_token_distribution(out.logits[0, -1], wm, temperature, top_p)
        nxt = int(rng.choice(len(p), p=p))
        if nxt == tokenizer.eos_token_id:
            break
        new_tokens.append(nxt)
        inp = torch.tensor([[nxt]], device=device)
    return new_tokens, tokenizer.decode(new_tokens, skip_special_tokens=True)


def tpr_at_fpr(wm_scores, unwm_scores, fpr=0.01):
    """As in the paper: threshold set so that the FPR on unwatermarked texts equals fpr.
    Returns (tpr, threshold)."""
    thr = np.quantile(np.asarray(unwm_scores), 1 - fpr)
    return float((np.asarray(wm_scores) > thr).mean()), float(thr)


def random_edits(tokens, frac, vocab_size, rng):
    """Apply int(frac * len) random insertions / deletions / replacements."""
    t = list(tokens)
    for _ in range(int(frac * len(t))):
        op, pos = rng.integers(3), int(rng.integers(len(t)))
        if op == 0:
            t.insert(pos, int(rng.integers(vocab_size)))
        elif op == 1 and len(t) > 1:
            t.pop(pos)
        else:
            t[pos] = int(rng.integers(vocab_size))
    return t
