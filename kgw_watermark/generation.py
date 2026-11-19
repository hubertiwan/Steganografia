"""Text generation with a Hugging Face model, with or without the KGW watermark."""
import numpy as np
import torch


def next_token_distribution(logits, wm=None, prev_token=None, temperature=1.0, top_p=0.9):
    """Same order as Hugging Face generate():
    1) watermark: +delta to green logits (green list from prev_token),
    2) temperature, 3) nucleus (top-p) sampling."""
    logits = logits.float().cpu().numpy().astype(np.float64)
    if wm is not None:
        logits = wm.watermark_logits(logits, prev_token)
    logits = logits / temperature

    p = np.exp(logits - logits.max())
    p /= p.sum()

    if top_p is not None and top_p < 1.0:
        order = np.argsort(-p)
        cum = np.cumsum(p[order])
        keep_count = int(np.searchsorted(cum, top_p)) + 1  # smallest set with mass >= top_p
        keep = order[:keep_count]
        q = np.zeros_like(p)
        q[keep] = p[keep]
        p = q / q.sum()
    return p


@torch.no_grad()
def generate(model, tokenizer, prompt, wm=None, max_new_tokens=200,
             temperature=1.0, top_p=0.9, seed=0, use_chat_template=True):
    """wm=None -> plain model (no watermark); wm=KGWWatermark -> watermarked model.
    Returns (new_token_ids, decoded_text)."""
    rng = np.random.default_rng(seed)
    device = next(model.parameters()).device

    if use_chat_template and getattr(tokenizer, "chat_template", None):
        messages = [{"role": "user", "content": prompt}]
        ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
        if not torch.is_tensor(ids):  # newer transformers versions return a dict
            ids = ids["input_ids"]
    else:
        ids = tokenizer(prompt, return_tensors="pt").input_ids
    ids = ids.to(device)

    prev_token = int(ids[0, -1])  # the first new token uses the last prompt token
    new_tokens = []
    past = None
    inp = ids
    for _ in range(max_new_tokens):
        out = model(input_ids=inp, past_key_values=past, use_cache=True)
        past = out.past_key_values
        p = next_token_distribution(out.logits[0, -1], wm, prev_token, temperature, top_p)
        nxt = int(rng.choice(len(p), p=p))
        if nxt == tokenizer.eos_token_id:
            break
        new_tokens.append(nxt)
        prev_token = nxt
        inp = torch.tensor([[nxt]], device=device)
    return new_tokens, tokenizer.decode(new_tokens, skip_special_tokens=True)
