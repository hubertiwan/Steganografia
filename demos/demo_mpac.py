"""MPAC (multi-bit watermark) demo with a real language model.

Embeds a random b-bit message in every watermarked text, decodes it back and reports bit
accuracy, detection (TPR / FPR), quality (perplexity) and robustness to random edits.

Usage:
    python -m demos.demo_mpac                                    # defaults (Qwen2.5-0.5B-Instruct)
    python -m demos.demo_mpac --bits 16 --max-new-tokens 300
    python -m demos.demo_mpac --num-prompts 5 --max-new-tokens 100   # quicker run
"""
import argparse
import math
import os

import numpy as np
import torch

from watermarks.mpac import MPACWatermark, bit_accuracy, generate
from watermarks.synthid import encode_prompt
from watermarks.unigram import random_edits, tpr_at_fpr

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
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct", help="Hugging Face model name or local folder")
    ap.add_argument("--no-chat-template", action="store_true", help="use for base (non-chat) models, e.g. OPT, GPT-2")
    ap.add_argument("--key", type=int, default=15485863, help="secret watermark key")
    ap.add_argument("--bits", type=int, default=8, help="message length in bits (paper: 8-32)")
    ap.add_argument("--gamma", type=float, default=0.25, help="colorlist fraction, radix r = floor(1/gamma) (paper: 0.25)")
    ap.add_argument("--delta", type=float, default=2.0, help="watermark strength (paper: 2.0)")
    ap.add_argument("--context-size", type=int, default=1, help="h, tokens hashed into the seed (paper: 1)")
    ap.add_argument("--alpha", type=float, default=0.001, help="false positive rate of the detector")
    ap.add_argument("--temperature", type=float, default=0.7, help="paper: 0.7")
    ap.add_argument("--top-p", type=float, default=1.0)
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


def random_message(bits, rng):
    return "".join(rng.choice(["0", "1"], size=bits))


def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)
    use_chat = not args.no_chat_template
    gen_kw = dict(max_new_tokens=args.max_new_tokens, temperature=args.temperature,
                  top_p=args.top_p, use_chat_template=use_chat)

    # ---- 1. Model and watermark ----
    model, tokenizer, device = load_model(args.model)
    vocab = model.get_output_embeddings().weight.shape[0]  # may be larger than len(tokenizer)
    wm = MPACWatermark(key=args.key, vocab_size=vocab, message_bits=args.bits, gamma=args.gamma,
                       delta=args.delta, context_size=args.context_size)
    print(f"{device} | {args.model} | vocab = {vocab} | radix r = {wm.radix} | "
          f"{args.bits}-bit message -> {wm.positions} positions\n")

    # ---- 2. Example: same prompt with and without watermark ----
    prompt = "Is it possible to fail several times and succeed once?"
    message = random_message(args.bits, np.random.default_rng(0))
    t_wm, txt_wm = generate(model, tokenizer, prompt, wm=wm, message=message, seed=1, **gen_kw)
    t_un, txt_un = generate(model, tokenizer, prompt, wm=None, seed=2, **gen_kw)
    print(f"embedded message: {message}\n")
    for title, toks, txt in [("WATERMARKED", t_wm, txt_wm), ("NOT WATERMARKED", t_un, txt_un)]:
        d = wm.detect(toks, alpha=args.alpha)
        print(f"=== {title} ===\n{txt}")
        print(f"-> tokens: {d['n']}, decoded message: {d['message']} "
              f"(bit accuracy {bit_accuracy(message, d['message']):.2f}), z = {d['z']:.2f}, "
              f"p = {d['p_value']:.2e}, watermarked (p <= {args.alpha}): {d['watermarked']}\n")

    # ---- 3. Experiment: bit accuracy, detection ----
    prompts = PROMPTS[: args.num_prompts]
    msg_rng = np.random.default_rng(1)
    messages = [random_message(args.bits, msg_rng) for _ in prompts]
    res_wm = []
    res_un = []
    for i, q in enumerate(prompts):
        res_wm.append(generate(model, tokenizer, q, wm=wm, message=messages[i], seed=i, **gen_kw)[0])
        res_un.append(generate(model, tokenizer, q, wm=None, seed=10_000 + i, **gen_kw)[0])
        print(f"generated {i + 1}/{len(prompts)}", end="\r")
    print()

    det_wm = [wm.detect(t, alpha=args.alpha) for t in res_wm]
    det_un = [wm.detect(t, alpha=args.alpha) for t in res_un]
    acc = np.array([bit_accuracy(m, d["message"]) for m, d in zip(messages, det_wm)])
    print("== Message ==")
    print(f"bit accuracy = {acc.mean():.3f} (bit error rate {1 - acc.mean():.3f}), "
          f"whole message correct in {np.mean(acc == 1.0):.2f} of texts")

    z_wm = np.array([d["z"] for d in det_wm])
    z_un = np.array([d["z"] for d in det_un])
    print("\n== Detection ==")
    print(f"exact p-value <= {args.alpha}: TPR = {np.mean([d['watermarked'] for d in det_wm]):.2f}, "
          f"FPR = {np.mean([d['watermarked'] for d in det_un]):.2f}")
    for f in [0.01, 0.10]:
        tpr, thr = tpr_at_fpr(z_wm, z_un, f)
        print(f"TPR @ FPR={f:.0%} (paper's z, empirical threshold): {tpr:.2f} (threshold z = {thr:.2f})")
    print(f"mean z: watermarked = {z_wm.mean():.2f}, not watermarked = {z_un.mean():.2f} "
          f"(z is > 0 without a watermark, see docs/mpac.md)\n")

    print("== Quality (perplexity, median) ==")
    ppl_wm = [perplexity(model, tokenizer, q, t, use_chat) for q, t in zip(prompts, res_wm)]
    ppl_un = [perplexity(model, tokenizer, q, t, use_chat) for q, t in zip(prompts, res_un)]
    print(f"watermarked = {np.median(ppl_wm):.2f}, not watermarked = {np.median(ppl_un):.2f}\n")

    # ---- 4. Robustness to random edits ----
    print("== Random edits ==")
    edit_rng = np.random.default_rng(123)
    fractions = [0.0, 0.1, 0.2, 0.3, 0.5]
    acc_edits = []
    for frac in fractions:
        accs, detected = [], []
        for m, tokens in zip(messages, res_wm):
            d = wm.detect(random_edits(tokens, frac, vocab, edit_rng), alpha=args.alpha)
            accs.append(bit_accuracy(m, d["message"]))
            detected.append(d["watermarked"])
        acc_edits.append(np.mean(accs))
        print(f"edits {int(frac * 100):3d}% of tokens -> bit accuracy = {np.mean(accs):.3f}, "
              f"detected: {np.mean(detected):.2f}")

    # ---- 5. Plot ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].hist(z_un, bins=15, alpha=0.7, label="not watermarked")
    axes[0].hist(z_wm, bins=15, alpha=0.7, label="MPAC")
    axes[0].set_xlabel("z-score (Eq. 1)")
    axes[0].set_ylabel("number of texts")
    axes[0].set_title("z-score distribution")
    axes[0].legend()
    axes[1].plot([100 * f for f in fractions], acc_edits, "o-")
    axes[1].set_ylim(0.4, 1.02)
    axes[1].set_xlabel("edited tokens [%]")
    axes[1].set_ylabel("bit accuracy")
    axes[1].set_title(f"{args.bits}-bit message after random edits")
    fig.tight_layout()
    plot_path = os.path.join(args.out, "mpac_results.png")
    fig.savefig(plot_path, dpi=150)
    print(f"\nPlot saved to {plot_path}")


if __name__ == "__main__":
    main()
