"""DiPmark demo with a real language model.

Generates texts with and without the watermark, runs detection, measures TPR at a
fixed FPR, quality (perplexity), robustness to random edits and saves a plot.

The detection score used for TPR / FPR and the plot is Phi * sqrt(n). The paper's
threshold Phi >= 1.517 / sqrt(n) (FPR 1%) is then simply score >= 1.517 for any length.

Usage:
    python demo_dipmark.py                                    # defaults (Qwen2.5-0.5B-Instruct)
    python demo_dipmark.py --model facebook/opt-1.3b --no-chat-template
    python demo_dipmark.py --num-prompts 5 --max-new-tokens 100   # quicker run
"""
import argparse
import math
import os

import numpy as np
import torch

from dipmark_watermark import DiPmarkWatermark, generate
from synthid_watermark import encode_prompt
from unigram_watermark import random_edits, tpr_at_fpr

PROMPTS = [
    "Why is the sky blue?", "How do airplanes stay in the air?", "Why do we need to sleep?",
    "How does the internet work?", "Why do cats purr?", "What causes earthquakes?",
    "How do vaccines work?", "Why is the ocean salty?", "How do magnets work?",
    "Why do leaves change color in autumn?", "How does a refrigerator keep food cold?",
    "Why do we have different time zones?", "How do bees make honey?", "What is inflation?",
    "Why do stars twinkle?", "How does a credit card work?", "Why do onions make you cry?",
    "How do computers store information?", "Why does ice float on water?", "How do rainbows form?",
]


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct", help="Hugging Face model name")
    ap.add_argument("--no-chat-template", action="store_true", help="use for base (non-chat) models, e.g. OPT, GPT-2")
    ap.add_argument("--key", type=int, default=15485863, help="secret watermark key")
    ap.add_argument("--alpha", type=float, default=0.45, help="reweight strength, at most 0.5 (paper: 0.45)")
    ap.add_argument("--gamma", type=float, default=0.5, help="green list separator for detection (paper: 0.5)")
    ap.add_argument("--context-size", type=int, default=1, help="a, tokens in the texture key (paper: 1)")
    ap.add_argument("--fpr", type=float, default=0.01, help="false positive rate of the detector")
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--max-new-tokens", type=int, default=200)
    ap.add_argument("--num-prompts", type=int, default=len(PROMPTS))
    ap.add_argument("--out", default="results", help="output folder for the plot")
    return ap.parse_args()


def load_model(name):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        dtype = torch.float16
    else:
        dtype = torch.float32
    tokenizer = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=dtype)
    model = model.to(device).eval()
    return model, tokenizer, device


@torch.no_grad()
def perplexity(model, tokenizer, prompt, new_tokens, use_chat_template):
    """Perplexity of the generated tokens given the prompt.
    shortcut: scored by the same model that generated the text (paper uses a bigger one)."""
    prompt_ids = encode_prompt(tokenizer, prompt, use_chat_template)[0].tolist()
    ids = torch.tensor([prompt_ids + new_tokens], device=model.device)
    labels = ids.clone()
    labels[0, :len(prompt_ids)] = -100  # do not score the prompt
    loss = model(ids, labels=labels).loss.item()
    return math.exp(loss)


def length_free_score(wm, tokens):
    """Phi * sqrt(n), comparable between texts of different length."""
    phi, green, n = wm.score(tokens)
    return phi * math.sqrt(n)


def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)
    use_chat = not args.no_chat_template
    gen_kw = dict(max_new_tokens=args.max_new_tokens, temperature=args.temperature,
                  top_p=args.top_p, use_chat_template=use_chat)
    score_threshold = math.sqrt(math.log(1 / args.fpr) / 2)  # 1.517 for FPR = 1%

    # ---- 1. Model and watermark ----
    model, tokenizer, device = load_model(args.model)
    vocab = model.get_output_embeddings().weight.shape[0]  # may be larger than len(tokenizer)
    wm = DiPmarkWatermark(key=args.key, vocab_size=vocab, alpha=args.alpha,
                          gamma=args.gamma, context_size=args.context_size)
    print(f"{device} | {args.model} | vocab = {vocab} | alpha = {wm.alpha} | gamma = {wm.gamma}\n")

    # ---- 2. Example: same prompt with and without watermark ----
    prompt = "Is it possible to fail several times and succeed once?"
    t_wm, txt_wm = generate(model, tokenizer, prompt, wm=wm, seed=1, **gen_kw)
    t_un, txt_un = generate(model, tokenizer, prompt, wm=None, seed=2, **gen_kw)
    examples = [("WATERMARKED", t_wm, txt_wm), ("NOT WATERMARKED", t_un, txt_un)]
    for title, toks, txt in examples:
        d = wm.detect(toks, fpr=args.fpr)
        print(f"=== {title} ===\n{txt}")
        print(f"-> scored tokens: {d['n']}, green: {d['green']} ({d['green_frac']:.0%}), "
              f"Phi = {d['phi']:.3f} (threshold {d['threshold']:.3f}), p-value <= {d['p_value']:.2g}, "
              f"watermarked: {d['watermarked']}\n")

    # ---- 3. Experiment: detection and quality ----
    prompts = PROMPTS[: args.num_prompts]
    res_wm = []
    res_un = []
    for i, q in enumerate(prompts):
        res_wm.append(generate(model, tokenizer, q, wm=wm, seed=i, **gen_kw)[0])
        res_un.append(generate(model, tokenizer, q, wm=None, seed=10_000 + i, **gen_kw)[0])
        print(f"generated {i + 1}/{len(prompts)}", end="\r")
    print()

    s_wm = []
    s_un = []
    for tokens in res_wm:
        s_wm.append(length_free_score(wm, tokens))
    for tokens in res_un:
        s_un.append(length_free_score(wm, tokens))
    s_wm = np.array(s_wm)
    s_un = np.array(s_un)

    print("== Detection (score = Phi * sqrt(n)) ==")
    for f in [0.01, 0.10]:
        tpr, thr = tpr_at_fpr(s_wm, s_un, f)
        print(f"TPR @ FPR={f:.0%}: {tpr:.2f} (threshold score = {thr:.2f})")
    tpr = np.mean(s_wm >= score_threshold)
    fpr = np.mean(s_un >= score_threshold)
    print(f"Paper threshold score >= {score_threshold:.3f}: TPR = {tpr:.2f}, FPR = {fpr:.2f}")
    print(f"mean score: watermarked = {s_wm.mean():.2f}, not watermarked = {s_un.mean():.2f}\n")

    print("== Quality (perplexity, median) ==")
    ppl_wm = []
    ppl_un = []
    for q, tokens in zip(prompts, res_wm):
        ppl_wm.append(perplexity(model, tokenizer, q, tokens, use_chat))
    for q, tokens in zip(prompts, res_un):
        ppl_un.append(perplexity(model, tokenizer, q, tokens, use_chat))
    print(f"watermarked = {np.median(ppl_wm):.2f}, not watermarked = {np.median(ppl_un):.2f}\n")

    # ---- 4. Robustness to random edits ----
    print("== Random edits ==")
    edit_rng = np.random.default_rng(123)
    for frac in [0.0, 0.1, 0.2, 0.3, 0.5]:
        scores = []
        for tokens in res_wm:
            edited = random_edits(tokens, frac, vocab, edit_rng)
            scores.append(length_free_score(wm, edited))
        scores = np.array(scores)
        print(f"edits {int(frac * 100):3d}% of tokens -> mean score = {scores.mean():5.2f}, "
              f"detected (score >= {score_threshold:.3f}): {np.mean(scores >= score_threshold):.2f}")

    # ---- 5. Plot ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.figure(figsize=(6, 4))
    plt.hist(s_un, bins=15, alpha=0.7, label="not watermarked")
    plt.hist(s_wm, bins=15, alpha=0.7, label="DiPmark")
    plt.axvline(score_threshold, color="k", ls="--", label=f"threshold = {score_threshold:.3f}")
    plt.xlabel("Phi * sqrt(n)")
    plt.ylabel("number of texts")
    plt.title("DiPmark score distribution")
    plt.legend()
    plt.tight_layout()
    plot_path = os.path.join(args.out, "dipmark_scores.png")
    plt.savefig(plot_path, dpi=150)
    print(f"\nPlot saved to {plot_path}")


if __name__ == "__main__":
    main()
