"""SynthID-Text (Tournament sampling) demo with a real language model.

Generates texts with and without the watermark, runs detection on the decoded text
(re-tokenized, as in real use), measures TPR / FPR, quality (perplexity), speed and
robustness to random edits, and saves a plot to the output folder.

Usage:
    python demo_synthid.py                                  # defaults (Qwen2.5-0.5B-Instruct)
    python demo_synthid.py --model Qwen/Qwen2.5-1.5B --no-chat-template
    python demo_synthid.py --num-prompts 5 --max-new-tokens 100   # quicker run
"""
import argparse
import math
import os
import time

import numpy as np
import torch

from synthid_watermark import TournamentWatermark, encode_prompt, generate
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
    ap.add_argument("--no-chat-template", action="store_true", help="use for base (non-chat) models")
    ap.add_argument("--key", default="secret-watermark-key", help="secret watermark key")
    ap.add_argument("--context-size", type=int, default=4, help="H, sliding window for the seed")
    ap.add_argument("--num-layers", type=int, default=8, help="m, tournament layers (2^m candidates)")
    ap.add_argument("--threshold", type=float, default=2.326, help="z-score threshold (FPR = 1%%)")
    ap.add_argument("--top-k", type=int, default=100)
    ap.add_argument("--temperature", type=float, default=0.7)
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


def text_z_score(wm, tokenizer, text):
    """The detector gets only the text, so it has to tokenize it again."""
    tokens = tokenizer(text, add_special_tokens=False).input_ids
    return wm.z_score(tokens)[0]


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


def auc(pos, neg):
    """ROC AUC = P(z_watermarked > z_plain), ties count 1/2 (Mann-Whitney statistic)."""
    wins = 0.0
    for p in pos:
        for q in neg:
            if p > q:
                wins += 1.0
            elif p == q:
                wins += 0.5
    return wins / (len(pos) * len(neg))


def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)
    use_chat = not args.no_chat_template
    gen_kw = dict(max_new_tokens=args.max_new_tokens, temperature=args.temperature,
                  top_k=args.top_k, use_chat_template=use_chat)

    # ---- 1. Model and watermark ----
    model, tokenizer, device = load_model(args.model)
    vocab = model.get_output_embeddings().weight.shape[0]
    wm = TournamentWatermark(key=args.key, context_size=args.context_size, num_layers=args.num_layers)
    print(f"{device} | {args.model} | H = {wm.context_size} | m = {wm.num_layers} "
          f"| candidates = {wm.num_candidates}\n")

    # ---- 2. Example: same prompt with and without watermark ----
    prompt = "Is it possible to fail several times and succeed once?"
    t_wm, txt_wm = generate(model, tokenizer, prompt, wm=wm, seed=1, **gen_kw)
    t_un, txt_un = generate(model, tokenizer, prompt, wm=None, seed=2, **gen_kw)
    examples = [("WATERMARKED", txt_wm), ("NOT WATERMARKED", txt_un)]
    for title, txt in examples:
        tokens = tokenizer(txt, add_special_tokens=False).input_ids
        d = wm.detect(tokens, threshold=args.threshold)
        print(f"=== {title} ===\n{txt}")
        print(f"-> scored tokens: {d['n']}, z = {d['z']:.2f}, "
              f"watermarked (z > {args.threshold}): {d['watermarked']}\n")

    # ---- 3. Experiment: detection, quality, speed ----
    prompts = PROMPTS[: args.num_prompts]
    res_wm = []
    res_un = []
    time_wm = 0.0
    time_un = 0.0
    for i, q in enumerate(prompts):
        start = time.time()
        res_wm.append(generate(model, tokenizer, q, wm=wm, seed=i, **gen_kw))
        time_wm += time.time() - start

        start = time.time()
        res_un.append(generate(model, tokenizer, q, wm=None, seed=10_000 + i, **gen_kw))
        time_un += time.time() - start
        print(f"generated {i + 1}/{len(prompts)}", end="\r")
    print()

    z_wm = []
    z_un = []
    for tokens, text in res_wm:
        z_wm.append(text_z_score(wm, tokenizer, text))
    for tokens, text in res_un:
        z_un.append(text_z_score(wm, tokenizer, text))
    z_wm = np.array(z_wm)
    z_un = np.array(z_un)

    print("== Detection ==")
    print(f"AUC = {auc(z_wm, z_un):.3f}")
    for f in [0.01, 0.10]:
        tpr, thr = tpr_at_fpr(z_wm, z_un, f)
        print(f"TPR @ FPR={f:.0%}: {tpr:.2f} (threshold z = {thr:.2f})")
    tpr = np.mean(z_wm > args.threshold)
    fpr = np.mean(z_un > args.threshold)
    print(f"Threshold z = {args.threshold}: TPR = {tpr:.2f}, FPR = {fpr:.2f}")
    print(f"mean z: watermarked = {z_wm.mean():.2f}, not watermarked = {z_un.mean():.2f}\n")

    print("== Quality (perplexity, median) ==")
    ppl_wm = []
    ppl_un = []
    for q, (tokens, text) in zip(prompts, res_wm):
        ppl_wm.append(perplexity(model, tokenizer, q, tokens, use_chat))
    for q, (tokens, text) in zip(prompts, res_un):
        ppl_un.append(perplexity(model, tokenizer, q, tokens, use_chat))
    print(f"watermarked = {np.median(ppl_wm):.2f}, not watermarked = {np.median(ppl_un):.2f}\n")

    print("== Speed ==")
    n_wm = 0
    n_un = 0
    for tokens, text in res_wm:
        n_wm += len(tokens)
    for tokens, text in res_un:
        n_un += len(tokens)
    print(f"watermarked = {n_wm / time_wm:.0f} tok/s, not watermarked = {n_un / time_un:.0f} tok/s\n")

    # ---- 4. Robustness to random edits (on token ids) ----
    print("== Random edits ==")
    edit_rng = np.random.default_rng(123)
    for frac in [0.0, 0.1, 0.2, 0.3, 0.5]:
        zs = []
        for tokens, text in res_wm:
            edited = random_edits(tokens, frac, vocab, edit_rng)
            zs.append(wm.z_score(edited)[0])
        zs = np.array(zs)
        print(f"edits {int(frac * 100):3d}% of tokens -> mean z = {zs.mean():5.2f}, "
              f"detected (z > {args.threshold}): {np.mean(zs > args.threshold):.2f}")

    # ---- 5. Plot ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.figure(figsize=(6, 4))
    plt.hist(z_un, bins=15, alpha=0.7, label="not watermarked")
    plt.hist(z_wm, bins=15, alpha=0.7, label="SynthID Tournament")
    plt.axvline(args.threshold, color="k", ls="--", label=f"threshold z = {args.threshold}")
    plt.xlabel("z-score")
    plt.ylabel("number of texts")
    plt.title("z-score distribution")
    plt.legend()
    plt.tight_layout()
    plot_path = os.path.join(args.out, "synthid_z_scores.png")
    plt.savefig(plot_path, dpi=150)
    print(f"\nPlot saved to {plot_path}")


if __name__ == "__main__":
    main()
