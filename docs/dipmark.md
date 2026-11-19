# DiPmark (distribution-preserving watermark)

A Python implementation of **DiPmark**, from:

> Yihan Wu, Zhengmian Hu, Junfeng Guo, Hongyang Zhang, Heng Huang. *A Resilient and Accessible Distribution-Preserving Watermark for Large Language Models.* ICML 2024, arXiv:2310.07710.

## How it works

1. **Cipher:** the previous token (texture key, a = 1) and a secret key give a random permutation of the vocabulary.
2. **DiP-reweight (Definition 4.2):** the token probabilities are laid out on [0, 1] in the order of the permutation. The probability mass in [0, α] is moved to [1 − α, 1], so tokens at the end of the permutation become more likely. Averaged over all keys the distribution does not change (Theorem 4.3), so the text quality stays the same. A texture key that was already used gets plain sampling.
3. **Detect (Algorithm 2):** the green list is the second part of the permutation, `θ[⌈γN⌉:]`. Count green tokens `L_G` and compute `Φ = L_G/n − (1 − γ)`. The text is flagged as watermarked if `Φ ≥ 1.517/√n` (false positive rate 1%). The p-value bound comes from Theorem 5.2.

Default parameters match the paper: α = 0.45, γ = 0.5, a = 1.

## Project structure

```
watermarks/dipmark/
    core.py         # DiPmarkWatermark class: permutation, DiP-reweight, green list, Phi, p-value
    generation.py   # text generation with a Hugging Face model, with or without the watermark
tests/
    test_dipmark.py # checks: distribution preserved (Theorem 4.3), FPR, detection, repeated contexts
demos/demo_dipmark.py # end-to-end experiment: detection, perplexity, random edits
```

## Usage

```bash
pytest -q
python -m demos.demo_dipmark
python -m demos.demo_dipmark --num-prompts 5 --max-new-tokens 100   # quicker
```

```python
from watermarks.dipmark import DiPmarkWatermark, generate

wm = DiPmarkWatermark(key=42, vocab_size=model.get_output_embeddings().weight.shape[0])
tokens, text = generate(model, tokenizer, "Why is the sky blue?", wm=wm)
print(wm.detect(tokens))
```

## Notes

- The detector skips texture keys that were already seen in the text, the same way as generation does. Repeated (context, token) pairs would break the Binomial assumption behind the threshold.
- The demo reports `Φ·√n` as the score, so one threshold (1.517 for FPR 1%) works for texts of any length.
- The signal is weaker than KGW at the same length (the price of not changing the distribution). Longer texts are detected more reliably.
- Perplexity is scored by the same model that generated the text, so it is only a rough quality check.
