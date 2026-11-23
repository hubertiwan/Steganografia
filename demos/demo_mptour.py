"""MPTour (own multi-bit watermark: MPAC positions + SynthID Tournament) demo with a real language model.

Embeds a random b-bit message in every watermarked text, decodes it back and reports bit
accuracy, detection (TPR / FPR), quality (perplexity) and robustness to random edits.
The same prompts and messages are also generated with MPAC (paper defaults) for comparison.

Usage:
    python -m demos.demo_mptour                                    # defaults (Qwen2.5-0.5B-Instruct)
    python -m demos.demo_mptour --bits 16 --max-new-tokens 300
    python -m demos.demo_mptour --num-prompts 5 --max-new-tokens 100   # quicker run
"""
import argparse
import os

import numpy as np

from demos.demo_mpac import PROMPTS, load_model, perplexity, random_message
from watermarks import mpac
from watermarks.mptour import MPTourWatermark, bit_accuracy, generate
from watermarks.unigram import random_edits, tpr_at_fpr


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct", help="Hugging Face model name or local folder")
    ap.add_argument("--no-chat-template", action="store_true", help="use for base (non-chat) models, e.g. OPT, GPT-2")
    ap.add_argument("--key", type=int, default=15485863, help="secret watermark key")
    ap.add_argument("--bits", type=int, default=8, help="message length in bits")
    ap.add_argument("--radix", type=int, default=4, help="r, values of one message digit (as MPAC with gamma=0.25)")
    ap.add_argument("--context-size", type=int, default=4, help="H, tokens hashed into the seed (SynthID: 4)")
    ap.add_argument("--num-layers", type=int, default=8, help="m, tournament layers, 2^m candidates (SynthID: 8)")
    ap.add_argument("--alpha", type=float, default=0.001, help="false positive rate of the detector")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-k", type=int, default=100)
    ap.add_argument("--max-new-tokens", type=int, default=200)
    ap.add_argument("--num-prompts", type=int, default=len(PROMPTS))
    ap.add_argument("--out", default="results", help="output folder for the plot")
    return ap.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)
    use_chat = not args.no_chat_template
    gen_kw = dict(max_new_tokens=args.max_new_tokens, temperature=args.temperature,
                  top_k=args.top_k, use_chat_template=use_chat)

    # ---- 1. Model and watermark ----
    model, tokenizer, device = load_model(args.model)
    vocab = model.get_output_embeddings().weight.shape[0]  # may be larger than len(tokenizer)
    wm = MPTourWatermark(key=args.key, message_bits=args.bits, radix=args.radix,
                         context_size=args.context_size, num_layers=args.num_layers)
    print(f"{device} | {args.model} | vocab = {vocab} | radix r = {wm.radix} | "
          f"{args.bits}-bit message -> {wm.positions} positions | {wm.num_candidates} candidates\n")

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
    base = mpac.MPACWatermark(key=args.key, vocab_size=vocab, message_bits=args.bits)  # paper defaults
    res_wm, res_un, res_mpac = [], [], []
    for i, q in enumerate(prompts):
        res_wm.append(generate(model, tokenizer, q, wm=wm, message=messages[i], seed=i, **gen_kw)[0])
        res_un.append(generate(model, tokenizer, q, wm=None, seed=10_000 + i, **gen_kw)[0])
        res_mpac.append(mpac.generate(model, tokenizer, q, wm=base, message=messages[i], seed=i,
                                      max_new_tokens=args.max_new_tokens, temperature=args.temperature,
                                      use_chat_template=use_chat)[0])
        print(f"generated {i + 1}/{len(prompts)}", end="\r")
    print()

    det_wm = [wm.detect(t, alpha=args.alpha) for t in res_wm]
    det_un = [wm.detect(t, alpha=args.alpha) for t in res_un]
    acc = np.array([bit_accuracy(m, d["message"]) for m, d in zip(messages, det_wm)])
    acc_mpac = np.array([bit_accuracy(m, base.detect(t)["message"]) for m, t in zip(messages, res_mpac)])
    print("== Message ==")
    print(f"MPTour: bit accuracy = {acc.mean():.3f}, whole message correct in {np.mean(acc == 1.0):.2f} of texts")
    print(f"MPAC:   bit accuracy = {acc_mpac.mean():.3f}, whole message correct in {np.mean(acc_mpac == 1.0):.2f} of texts")

    z_wm = np.array([d["z"] for d in det_wm])
    z_un = np.array([d["z"] for d in det_un])
    print("\n== Detection ==")
    print(f"exact p-value <= {args.alpha}: TPR = {np.mean([d['watermarked'] for d in det_wm]):.2f}, "
          f"FPR = {np.mean([d['watermarked'] for d in det_un]):.2f}")
    for f in [0.01, 0.10]:
        tpr, thr = tpr_at_fpr(z_wm, z_un, f)
        print(f"TPR @ FPR={f:.0%} (z, empirical threshold): {tpr:.2f} (threshold z = {thr:.2f})")
    print(f"mean z: watermarked = {z_wm.mean():.2f}, not watermarked = {z_un.mean():.2f} "
          f"(z is > 0 without a watermark, see docs/mptour.md)\n")

    print("== Quality (perplexity, median) ==")
    ppl = {name: np.median([perplexity(model, tokenizer, q, t, use_chat) for q, t in zip(prompts, res)])
           for name, res in [("MPTour", res_wm), ("MPAC", res_mpac), ("not watermarked", res_un)]}
    print(", ".join(f"{name} = {v:.2f}" for name, v in ppl.items()) + "\n")

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
    axes[0].hist(z_wm, bins=15, alpha=0.7, label="MPTour")
    axes[0].set_xlabel("z-score")
    axes[0].set_ylabel("number of texts")
    axes[0].set_title("z-score distribution")
    axes[0].legend()
    axes[1].plot([100 * f for f in fractions], acc_edits, "o-")
    axes[1].set_ylim(0.4, 1.02)
    axes[1].set_xlabel("edited tokens [%]")
    axes[1].set_ylabel("bit accuracy")
    axes[1].set_title(f"{args.bits}-bit message after random edits")
    fig.tight_layout()
    plot_path = os.path.join(args.out, "mptour_results.png")
    fig.savefig(plot_path, dpi=150)
    print(f"\nPlot saved to {plot_path}")


if __name__ == "__main__":
    main()
