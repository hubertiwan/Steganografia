# SynthID-Text (Tournament sampling)

A Python implementation of the **SynthID-Text** watermark (Tournament sampling, Algorithm 1), from:

> Sumanth Dathathri et al. *Scalable watermarking for identifying large language model outputs.* Nature 634 (2024).

## How it works

1. **Seed:** the last H tokens and a secret key are hashed into a seed `r_t`. Each (seed, token, layer) gives a pseudorandom bit `g_l(x, r_t)`, the g-value.
2. **Tournament (Algorithm 1):** draw 2^m candidate tokens from the model's distribution, pair them up and in layer l keep the one with the larger `g_l` (tie -> random). After m layers one token is left. If the context was already used in this text, plain sampling is used instead.
3. **Detect:** without the watermark every g-value is a fair coin. Whoever has the key computes the z-score of the sum of g-values over all tokens and layers. Repeated contexts are counted once. If `z > 2.326` (false positive rate 1%), the text is flagged as watermarked.

Default parameters: H = 4, m = 8 (256 candidates), top-k = 100, temperature = 0.7.

## Project structure

```
watermarks/synthid/
    core.py         # TournamentWatermark class: seed, g-values, Tournament sampling, z-score
    generation.py   # text generation with a Hugging Face model, with or without the watermark
tests/
    test_synthid.py # checks: determinism, mean g = 3/4 for a flat distribution, z ~ N(0, 1) without watermark
demos/demo_synthid.py # end-to-end experiment: detection, perplexity, speed, random edits
```

## Usage

```bash
pytest -q
python -m demos.demo_synthid
python -m demos.demo_synthid --num-prompts 5 --max-new-tokens 100   # quicker
```

```python
from watermarks.synthid import TournamentWatermark, generate

wm = TournamentWatermark(key="secret")
tokens, text = generate(model, tokenizer, "Why is the sky blue?", wm=wm)
print(wm.detect(tokenizer(text, add_special_tokens=False).input_ids))
```

## Notes

- Detection runs on the decoded text, tokenized again, as in real use.
- Perplexity is scored by the same model that generated the text, so it is only a rough quality check (the paper uses a bigger model).
- Not reproduced: the Bayesian (learned) detector and speculative sampling integration from the paper; only the mean-score detector is used.
