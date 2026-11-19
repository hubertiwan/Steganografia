"""Text generation with a Hugging Face model, with or without the SynthID-Text watermark."""
import numpy as np
import torch


def next_token_distribution(logits, temperature=0.7, top_k=100):
    """Model distribution after temperature and top-k (numpy array over the vocabulary)."""
    logits = logits.float().cpu().numpy().astype(np.float64)
    logits = logits / temperature

    if top_k is not None and top_k < len(logits):
        keep = np.argsort(-logits)[:top_k]
        masked = np.full_like(logits, -np.inf)
        masked[keep] = logits[keep]
        logits = masked

    p = np.exp(logits - logits.max())
    p /= p.sum()
    return p


def encode_prompt(tokenizer, prompt, use_chat_template=True):
    """Prompt -> tensor of token ids with shape (1, length)."""
    if use_chat_template and getattr(tokenizer, "chat_template", None):
        messages = [{"role": "user", "content": prompt}]
        ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
        if not torch.is_tensor(ids):  # newer transformers versions return a dict
            ids = ids["input_ids"]
    else:
        ids = tokenizer(prompt, return_tensors="pt").input_ids
    return ids


@torch.no_grad()
def generate(model, tokenizer, prompt, wm=None, max_new_tokens=200,
             temperature=0.7, top_k=100, seed=0, use_chat_template=True):
    """wm=None -> plain sampling (no watermark); wm=TournamentWatermark -> watermarked text.
    Both use the same distribution (temperature + top-k), so the comparison is fair.
    Returns (new_token_ids, decoded_text)."""
    rng = np.random.default_rng(seed)
    device = next(model.parameters()).device

    ids = encode_prompt(tokenizer, prompt, use_chat_template).to(device)
    all_tokens = ids[0].tolist()  # prompt + generated tokens, used as the context window
    seen = set()                  # contexts already used in this text

    new_tokens = []
    past = None
    inp = ids
    for _ in range(max_new_tokens):
        out = model(input_ids=inp, past_key_values=past, use_cache=True)
        past = out.past_key_values
        p = next_token_distribution(out.logits[0, -1], temperature, top_k)

        if wm is None:
            nxt = int(rng.choice(len(p), p=p))
        else:
            context = tuple(all_tokens[-wm.context_size:])
            if context in seen:
                # repeated context -> plain sampling, otherwise the same token would be favoured again
                nxt = int(rng.choice(len(p), p=p))
            else:
                seen.add(context)
                nxt = wm.tournament_sample(p, context, rng)

        if nxt == tokenizer.eos_token_id:
            break
        new_tokens.append(nxt)
        all_tokens.append(nxt)
        inp = torch.tensor([[nxt]], device=device)

    return new_tokens, tokenizer.decode(new_tokens, skip_special_tokens=True)
