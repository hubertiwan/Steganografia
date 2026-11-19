# KGW (soft red list)

A Python implementation of the **KGW** watermark (Algorithm 2, soft red list), from:

> John Kirchenbauer, Jonas Geiping, Yuxin Wen, Jonathan Katz, Ian Miers, Tom Goldstein. *A Watermark for Large Language Models.* ICML 2023, arXiv:2301.10226.

## How it works

1. **Green list from the previous token:** the previous token and a secret key are hashed into a seed. The seed splits the vocabulary into a *green list* (a fraction γ of all tokens) and a *red list*. Unlike Unigram-Watermark, the split changes at every position.
2. **Watermark:** the logits of green tokens get a bonus δ, then the next token is sampled as usual (temperature, top-p).
3. **Detect:** whoever has the key rebuilds the green list of every token from the token before it, counts the green tokens and computes `z = (|s|_G − γT) / sqrt(Tγ(1−γ))`. If `z > 4`, the text is flagged as watermarked.

Default parameters match the paper: γ = 0.25, δ = 2.0, threshold z = 4.

## Project structure

```
watermarks/kgw/
    core.py         # KGWWatermark class: green list from the previous token, watermark, z-score
    generation.py   # text generation with a Hugging Face model, with or without the watermark
tests/
    test_kgw.py     # checks: green list size and determinism, z ~ N(0, 1) without watermark, detection
demos/demo_kgw.py   # end-to-end experiment: detection, perplexity, random edits
```

## Usage

```bash
pytest -q
python -m demos.demo_kgw
python -m demos.demo_kgw --num-prompts 5 --max-new-tokens 100   # quicker
```

```python
from watermarks.kgw import KGWWatermark, generate

wm = KGWWatermark(key=42, vocab_size=model.get_output_embeddings().weight.shape[0])
tokens, text = generate(model, tokenizer, "Why is the sky blue?", wm=wm)
print(wm.detect(tokens))
```

## Notes

- Only the soft red list (Algorithm 2) is implemented. The hard red list (Algorithm 1) and beam search are not.
- The first generated token uses the last prompt token as its previous token. The detector does not see the prompt, so it scores tokens 2..T.
- Perplexity is scored by the same model that generated the text, so it is only a rough quality check.
