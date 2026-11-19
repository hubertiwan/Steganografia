"""A fake language model for fast tests of the application (not used by the app itself).

The next-token logits are spread * N(0, 1), fixed for the last `context` tokens, like the fake
logits in the algorithm tests. Token i is the word "w<i>", so the tokenizer is invertible.
It has the interface the watermark packages use: forward() with past_key_values, .logits, .loss.
"""
import hashlib
import re
from types import SimpleNamespace

import numpy as np
import torch

from app.model import LanguageModel


class FakeTokenizer:
    chat_template = None
    eos_token_id = None  # never stops early
    all_special_ids = []

    def __init__(self, vocab_size):
        self.vocab_size = vocab_size

    def encode(self, text):
        ids = []
        for word in text.split():
            m = re.fullmatch(r"w(\d+)", word)
            if m and int(m.group(1)) < self.vocab_size:
                ids.append(int(m.group(1)))
            else:  # words of a prompt
                ids.append(int.from_bytes(hashlib.sha256(word.encode()).digest()[:4], "little") % self.vocab_size)
        return ids

    def decode(self, ids, skip_special_tokens=True):
        return " ".join(f"w{int(i)}" for i in ids)

    def __call__(self, text, return_tensors=None, add_special_tokens=True):
        ids = self.encode(text)
        if return_tensors == "pt":
            return SimpleNamespace(input_ids=torch.tensor([ids or [0]], dtype=torch.long))
        return SimpleNamespace(input_ids=ids)


class FakeModel(torch.nn.Module):
    def __init__(self, vocab_size, spread, context, seed):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(1), requires_grad=False)  # for next(model.parameters())
        self.vocab_size, self.spread, self.context, self.seed = vocab_size, spread, context, seed

    def logits_for(self, history):
        data = f"{self.seed}|" + ",".join(str(t) for t in history[-self.context:])
        rng = np.random.default_rng(int.from_bytes(hashlib.sha256(data.encode()).digest()[:8], "little"))
        return rng.standard_normal(self.vocab_size) * self.spread

    def forward(self, input_ids=None, past_key_values=None, use_cache=True, labels=None, **kwargs):
        history = list(past_key_values or [])
        rows = []
        for tok in input_ids[0].tolist():
            history.append(int(tok))
            rows.append(self.logits_for(history))
        logits = torch.tensor(np.stack(rows), dtype=torch.float32)[None]
        loss = None
        if labels is not None:  # like Hugging Face: token t is predicted from position t-1
            loss = torch.nn.functional.cross_entropy(logits[0, :-1], labels[0, 1:].long(), ignore_index=-100)
        return SimpleNamespace(logits=logits, past_key_values=history, loss=loss)


def make_fake_lm(vocab_size=3000, spread=3.0, context=2, seed=0):
    model = FakeModel(vocab_size, spread, context, seed).eval()
    return LanguageModel("fake", model, FakeTokenizer(vocab_size), vocab_size, "cpu", False)
