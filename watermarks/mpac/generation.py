"""Text generation with a Hugging Face model, with or without the MPAC watermark."""
import numpy as np
import torch


def next_token_distribution(logits, wm=None, context=None, digits=None, temperature=0.7, top_p=1.0):
    """Same order as Hugging Face generate():
    1) watermark: +delta to the colorlist of the message digit, 2) temperature, 3) top-p.
    The paper samples with temperature 0.7 and no top-p (top_p = 1.0)."""
    logits = logits.float().cpu().numpy().astype(np.float64)
    if wm is not None:
        logits = wm.watermark_logits(logits, context, digits)
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
def generate(model, tokenizer, prompt, wm=None, message=None, max_new_tokens=200,
             temperature=0.7, top_p=1.0, seed=0, use_chat_template=True):
    """wm=None -> plain model; wm=MPACWatermark and message='0110...' -> watermarked model.
    Returns (new_token_ids, decoded_text)."""
    digits = None
    if wm is not None:
        digits = wm.check_message(message)
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

    all_tokens = ids[0].tolist()  # prompt + generated tokens, the context for the seed
    new_tokens = []
    past = None
    inp = ids
    for _ in range(max_new_tokens):
        out = model(input_ids=inp, past_key_values=past, use_cache=True)
        past = out.past_key_values
        context = all_tokens[-wm.context_size:] if wm is not None else None
        p = next_token_distribution(out.logits[0, -1], wm, context, digits, temperature, top_p)
        nxt = int(rng.choice(len(p), p=p))
        if nxt == tokenizer.eos_token_id:
            break
        new_tokens.append(nxt)
        all_tokens.append(nxt)
        inp = torch.tensor([[nxt]], device=device)
    return new_tokens, tokenizer.decode(new_tokens, skip_special_tokens=True)
