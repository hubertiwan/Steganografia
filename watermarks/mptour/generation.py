"""Text generation with a Hugging Face model, with or without the MPTour watermark."""
import numpy as np
import torch

from watermarks.synthid import encode_prompt, next_token_distribution


@torch.no_grad()
def generate(model, tokenizer, prompt, wm=None, message=None, max_new_tokens=200,
             temperature=0.7, top_k=100, seed=0, use_chat_template=True):
    """wm=None -> plain sampling; wm=MPTourWatermark and message='0110...' -> watermarked text.
    Both use the same distribution (temperature + top-k), as in SynthID.
    Returns (new_token_ids, decoded_text)."""
    digits = None
    if wm is not None:
        digits = wm.check_message(message)
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
                nxt = wm.tournament_sample(p, context, digits, rng)

        if nxt == tokenizer.eos_token_id:
            break
        new_tokens.append(nxt)
        all_tokens.append(nxt)
        inp = torch.tensor([[nxt]], device=device)

    return new_tokens, tokenizer.decode(new_tokens, skip_special_tokens=True)
