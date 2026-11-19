"""A tiny Hugging Face model built locally for tests (no download).

GPT-2 with 2 layers and random weights, a word-level tokenizer over a few hundred English
words and a chat template, so the same code path as for Qwen (apply_chat_template, KV cache,
loss for perplexity) is exercised. The generated text is random words, not real language.
"""
WORDS = ("the a an of to in and is are was it that this for on with as by at from be have not "
         "watermark detector token text model language green red list key secret score test "
         "message bit user sentence word random sample probability vocabulary seed position "
         "color sky blue sun light water ocean salt cat dog bird tree leaf autumn winter summer "
         "spring why how what when where who because so very more most many some any every "
         "computer internet network data memory store information money credit card bank "
         "airplane wing air flight engine star night moon earth quake vaccine body cell "
         "magnet field iron ice float rainbow rain drop honey bee onion cry time zone clock "
         "sleep brain dream fail succeed once several times people often think can will "
         "make use work help learn read write see know find give tell ask try call keep").split()

CHAT_TEMPLATE = ("{% for m in messages %}<|im_start|>{{ m['role'] }} {{ m['content'] }} <|im_end|> "
                 "{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant {% endif %}")


def build_tiny_model(path, chat_template=True, seed=0):
    import torch
    from tokenizers import Tokenizer, models, pre_tokenizers, decoders
    from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

    specials = ["<|endoftext|>", "<|im_start|>", "<|im_end|>", "[UNK]", "user", "assistant"]
    words = sorted(set(WORDS) - set(specials))
    vocab = {w: i for i, w in enumerate(specials + words)}
    raw = Tokenizer(models.WordLevel(vocab=vocab, unk_token="[UNK]"))
    raw.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    raw.decoder = decoders.WordPiece(prefix="##")  # joins words with spaces
    tok = PreTrainedTokenizerFast(tokenizer_object=raw, eos_token="<|im_end|>", unk_token="[UNK]",
                                  additional_special_tokens=["<|endoftext|>", "<|im_start|>"])
    if chat_template:
        tok.chat_template = CHAT_TEMPLATE
    tok.save_pretrained(path)
    torch.manual_seed(seed)
    cfg = GPT2Config(vocab_size=len(tok) + 6, n_positions=1024, n_embd=64, n_layer=2, n_head=2,
                     bos_token_id=0, eos_token_id=tok.eos_token_id, pad_token_id=0)
    GPT2LMHeadModel(cfg).save_pretrained(path)  # vocab_size > len(tokenizer), like Qwen
    return path
