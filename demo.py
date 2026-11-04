"""Unigram-Watermark demo with a real language model.

Generates texts with and without the watermark, runs detection, measures TPR at a
fixed FPR, tests robustness to random edits and saves plots to the output folder.

Usage:
    python demo.py                                   # defaults (Qwen2.5-0.5B-Instruct)
    python demo.py --model facebook/opt-1.3b --no-chat-template
    python demo.py --num-prompts 10 --max-new-tokens 100   # quicker run
"""
import argparse
import html
import os

import numpy as np
import torch

from unigram_watermark import UnigramWatermark, generate, random_edits, tpr_at_fpr

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
    ap.add_argument("--gamma", type=float, default=0.5, help="green list fraction (paper: 0.5)")
    ap.add_argument("--delta", type=float, default=2.0, help="watermark strength (paper: 2.0)")
    ap.add_argument("--tau", type=float, default=6.0, help="z-score detection threshold (paper: 6.0)")
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--max-new-tokens", type=int, default=200)
    ap.add_argument("--num-prompts", type=int, default=len(PROMPTS))
    ap.add_argument("--out", default="results", help="output folder for plots and HTML")
    return ap.parse_args()


def load_model(name):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(
        name, torch_dtype=torch.float16 if device == "cuda" else torch.float32).to(device).eval()
    return model, tokenizer, device


def colorize_html(tokens, tokenizer, wm):
    """Green tokens in green, red tokens in red (like Table 1 in the paper)."""
    spans = []
    for t in tokens:
        piece = html.escape(tokenizer.decode([t])).replace("\n", "<br>")
        color = "#1a9641" if wm.green_mask[t] else "#d7191c"
        spans.append(f'<span style="color:{color}">{piece}</span>')
    return "<div style='font-family:monospace; line-height:1.6'>" + "".join(spans) + "</div>"


def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)
    gen_kw = dict(max_new_tokens=args.max_new_tokens, temperature=args.temperature,
                  top_p=args.top_p, use_chat_template=not args.no_chat_template)

    # ---- 1. Model and watermark ----
    model, tokenizer, device = load_model(args.model)
    vocab = model.get_output_embeddings().weight.shape[0]  # may be larger than len(tokenizer)
    wm = UnigramWatermark(key=args.key, vocab_size=vocab, gamma=args.gamma, delta=args.delta)
    print(f"{device} | {args.model} | vocab = {vocab} | green tokens = {wm.green_size}\n")

    # ---- 2. Example: same prompt with and without watermark ----
    prompt = "Is it possible to fail several times and succeed once?"  # prompt from Table 1
    t_wm, txt_wm = generate(model, tokenizer, prompt, wm=wm, seed=1, **gen_kw)
    t_un, txt_un = generate(model, tokenizer, prompt, wm=None, seed=2, **gen_kw)
    for title, toks, txt in [("WATERMARKED", t_wm, txt_wm), ("NOT WATERMARKED", t_un, txt_un)]:
        d = wm.detect(toks, tau=args.tau)
        print(f"=== {title} ===\n{txt}")
        print(f"-> tokens: {d['n']}, green: {d['green']} ({d['green_frac']:.0%}), z = {d['z']:.2f}, "
              f"watermarked (z > {args.tau}): {d['watermarked']}, "
              f"Unique: {wm.detect_unique(toks)['watermarked']}\n")
    html_path = os.path.join(args.out, "example_colored.html")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write("<meta charset='utf-8'><b>Watermarked:</b>" + colorize_html(t_wm, tokenizer, wm)
                + "<br><b>Not watermarked:</b>" + colorize_html(t_un, tokenizer, wm))
    print(f"Colored tokens saved to {html_path}\n")

    # ---- 3. Experiment: z-score distribution and TPR @ FPR ----
    prompts = PROMPTS[: args.num_prompts]
    res_wm, res_un = [], []
    for i, q in enumerate(prompts):
        res_wm.append(generate(model, tokenizer, q, wm=wm, seed=i, **gen_kw)[0])
        res_un.append(generate(model, tokenizer, q, wm=None, seed=10_000 + i, **gen_kw)[0])
        print(f"generated {i + 1}/{len(prompts)}", end="\r")
    print()

    z_wm = np.array([wm.z_score(t)[0] for t in res_wm])
    z_un = np.array([wm.z_score(t)[0] for t in res_un])
    for f in [0.01, 0.10]:
        tpr, thr = tpr_at_fpr(z_wm, z_un, f)
        print(f"TPR @ FPR={f:.0%}: {tpr:.2f} (threshold z = {thr:.2f})")
    print(f"Threshold z = {args.tau}: TPR = {np.mean(z_wm > args.tau):.2f}, FPR = {np.mean(z_un > args.tau):.2f}")
    print(f"Unique (alpha=1%): TPR = {np.mean([wm.detect_unique(t)['watermarked'] for t in res_wm]):.2f}, "
          f"FPR = {np.mean([wm.detect_unique(t)['watermarked'] for t in res_un]):.2f}\n")

    # ---- 4. Robustness to random edits ----
    edit_rng = np.random.default_rng(123)
    for frac in [0.0, 0.1, 0.2, 0.3, 0.5, 0.7]:
        zs = np.array([wm.z_score(random_edits(t, frac, vocab, edit_rng))[0] for t in res_wm])
        print(f"edits {int(frac * 100):3d}% of tokens -> mean z = {zs.mean():5.2f}, "
              f"detected (z > {args.tau}): {np.mean(zs > args.tau):.2f}, detected (z > 4): {np.mean(zs > 4):.2f}")

    # ---- 5. Plots ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].hist(z_un, bins=15, alpha=0.7, label="not watermarked")
    ax[0].hist(z_wm, bins=15, alpha=0.7, label="Unigram-Watermark")
    ax[0].axvline(args.tau, color="k", ls="--", label=f"threshold z = {args.tau}")
    ax[0].set_xlabel("z-score"); ax[0].set_ylabel("number of texts"); ax[0].legend()
    ax[0].set_title("z-score distribution (cf. Fig. 1a)")

    lengths = [n for n in [10, 25, 50, 100, 150, 200] if n <= args.max_new_tokens]
    ax[1].plot(lengths, [np.mean([wm.z_score(t[:n])[0] for t in res_wm]) for n in lengths], "o-", label="watermarked")
    ax[1].plot(lengths, [np.mean([wm.z_score(t[:n])[0] for t in res_un]) for n in lengths], "o-", label="not watermarked")
    ax[1].axhline(args.tau, color="k", ls="--")
    ax[1].set_xlabel("number of tokens"); ax[1].set_ylabel("mean z-score"); ax[1].legend()
    ax[1].set_title("z-score grows like sqrt(n) for watermarked text")
    plt.tight_layout()
    plot_path = os.path.join(args.out, "z_scores.png")
    plt.savefig(plot_path, dpi=150)
    print(f"\nPlot saved to {plot_path}")


if __name__ == "__main__":
    main()
