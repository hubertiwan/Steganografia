"""Comparison of MPTour with MPAC.

Variant A of MPTour (MPAC colorlists as g-functions) was removed after this comparison; it is in
the git history (commit f2a00d5).

All variants get the same prompts, messages and seeds. Reported per variant: bit accuracy,
whole message correct, TPR / FPR (exact p-value <= alpha), z of the known message (verify,
for the tournament variants), perplexity, bit accuracy after 10% random edits and time per token.

Usage:
    python -m demos.compare_mptour                         # Qwen2.5-0.5B-Instruct, 10 prompts
    python -m demos.compare_mptour --num-prompts 4 --max-new-tokens 100
"""
import argparse
import time

import numpy as np

from demos.demo_mpac import PROMPTS, load_model, perplexity, random_message
from watermarks import mpac, mptour
from watermarks.unigram import random_edits


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--no-chat-template", action="store_true")
    ap.add_argument("--key", type=int, default=15485863)
    ap.add_argument("--bits", type=int, default=8)
    ap.add_argument("--alpha", type=float, default=0.001)
    ap.add_argument("--max-new-tokens", type=int, default=200)
    ap.add_argument("--num-prompts", type=int, default=10)
    return ap.parse_args()


def main():
    args = parse_args()
    use_chat = not args.no_chat_template
    model, tokenizer, device = load_model(args.model)
    vocab = model.get_output_embeddings().weight.shape[0]

    tour_kw = dict(temperature=0.7, top_k=100, max_new_tokens=args.max_new_tokens, use_chat_template=use_chat)
    variants = [  # name, watermark, generate function, its sampling arguments
        ("MPTour B (hash, r=4)", mptour.MPTourWatermark(args.key, args.bits, radix=4), mptour.generate, tour_kw),
        ("MPAC (delta=2, gamma=0.25)", mpac.MPACWatermark(args.key, vocab, args.bits), mpac.generate,
         dict(temperature=0.7, top_p=1.0, max_new_tokens=args.max_new_tokens, use_chat_template=use_chat)),
    ]
    prompts = PROMPTS[: args.num_prompts]
    msg_rng = np.random.default_rng(1)
    messages = [random_message(args.bits, msg_rng) for _ in prompts]
    plain = [mptour.generate(model, tokenizer, q, wm=None, seed=10_000 + i, **tour_kw)[0] for i, q in enumerate(prompts)]
    ppl_plain = np.median([perplexity(model, tokenizer, q, t, use_chat) for q, t in zip(prompts, plain)])
    print(f"{device} | {args.model} | {len(prompts)} prompts | {args.bits}-bit messages | "
          f"perplexity without watermark = {ppl_plain:.2f}\n")

    edit_rng = np.random.default_rng(123)
    header = f"{'variant':30s} {'bit acc':>7s} {'msg ok':>6s} {'TPR':>5s} {'FPR':>5s} {'z(msg)':>6s} " \
             f"{'ppl':>5s} {'acc@10%':>7s} {'ms/tok':>6s}"
    rows = []
    for name, wm, gen, kw in variants:
        start, tokens = time.perf_counter(), []
        for i, q in enumerate(prompts):
            tokens.append(gen(model, tokenizer, q, wm=wm, message=messages[i], seed=i, **kw)[0])
            print(f"{name}: generated {i + 1}/{len(prompts)}", end="\r")
        ms = 1000 * (time.perf_counter() - start) / sum(len(t) for t in tokens)
        det = [wm.detect(t, alpha=args.alpha) for t in tokens]
        acc = np.array([mpac.bit_accuracy(m, d["message"]) for m, d in zip(messages, det)])
        fpr = np.mean([wm.detect(t, alpha=args.alpha)["watermarked"] for t in plain])
        z_msg = np.mean([wm.verify(t, m)[0] for t, m in zip(tokens, messages)]) if hasattr(wm, "verify") else np.nan
        ppl = np.median([perplexity(model, tokenizer, q, t, use_chat) for q, t in zip(prompts, tokens)])
        acc10 = np.mean([mpac.bit_accuracy(m, wm.detect(random_edits(t, 0.1, vocab, edit_rng))["message"])
                         for m, t in zip(messages, tokens)])
        rows.append(f"{name:30s} {acc.mean():7.3f} {np.mean(acc == 1):6.2f} "
                    f"{np.mean([d['watermarked'] for d in det]):5.2f} {fpr:5.2f} {z_msg:6.2f} "
                    f"{ppl:5.2f} {acc10:7.3f} {ms:6.1f}")
        print(" " * 80, end="\r")
    print(header)
    print("\n".join(rows))


if __name__ == "__main__":
    main()
