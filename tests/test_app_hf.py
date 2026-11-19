"""The application with a real Hugging Face pipeline (transformers + tokenizers).

The model is a tiny GPT-2 with random weights built locally (tests/tiny_model.py), with a chat
template, so the same code path as for Qwen is used without downloading anything. Its text is
random words; only the integration is checked here, not text quality.
"""
import pytest

from app import algorithms, experiments
from app.errors import AppError
from app.model import load_model


@pytest.fixture(scope="module")
def tiny_lm(tiny_model_dir):
    return load_model(str(tiny_model_dir))


def test_model_loading(tiny_lm, tiny_model_dir):
    assert tiny_lm.use_chat_template  # the tokenizer has a chat template
    assert tiny_lm.vocab_size > len(tiny_lm.tokenizer)  # output layer larger than the tokenizer, like Qwen
    ids = tiny_lm.encode_prompt("why is the sky blue")
    assert tiny_lm.tokenizer.decode(ids).startswith("<|im_start|> user why is the sky blue")
    assert not load_model(str(tiny_model_dir), use_chat_template=False).use_chat_template


def test_bad_model_name():
    with pytest.raises(AppError) as e:
        load_model("/no/such/model")
    assert e.value.hint
    with pytest.raises(AppError):
        load_model("   ")


@pytest.mark.parametrize("algo_id", list(algorithms.ALGORITHMS))
def test_demo_with_transformers(algo_id, tiny_lm):
    r = experiments.run_demo(tiny_lm, algo_id, algorithms.get(algo_id).defaults(), "why is the sky blue", 60, seed=0)
    wm, plain = r["watermarked"], r["plain"]
    assert wm["tokens"] and isinstance(wm["text"], str)
    assert "".join(wm["pieces"]) == wm["text"]
    assert all(t < tiny_lm.vocab_size for t in wm["tokens"])
    assert wm["perplexity"] is not None and wm["perplexity"] > 1
    # random weights -> high entropy -> a clear signal
    assert wm["detection"]["score"] > plain["detection"]["score"]


def test_test_series_with_transformers(tiny_lm):
    s = {"prompts": ["why is the sky blue", "how do bees make honey"], "max_new_tokens": 40,
         "detect_on": "text", "edit_fractions": [0, 0.3]}
    r = experiments.run_test(tiny_lm, "mpac", algorithms.get("mpac").defaults(), s)
    assert r["summary"]["n"] == 2 and r["summary"]["ppl_wm"] is not None


def test_token_pieces_with_multibyte_characters():
    """Byte-level BPE (as in Qwen) splits Polish letters and emoji into several tokens.
    The pieces used for coloring must still give back the exact text."""
    from tokenizers import ByteLevelBPETokenizer
    from transformers import PreTrainedTokenizerFast

    from app.model import LanguageModel
    bpe = ByteLevelBPETokenizer()
    bpe.train_from_iterator(["the sky is blue"] * 10, vocab_size=280, min_frequency=1)
    tok = PreTrainedTokenizerFast(tokenizer_object=bpe)
    lm = LanguageModel("bpe", None, tok, len(tok), "cpu", False)
    for text in ["Zażółć gęślą jaźń 😀 the sky", "ąę"]:
        ids = tok(text).input_ids
        assert "".join(lm.pieces(ids)) == text
        assert "".join(lm.pieces(ids[:-1])) == tok.decode(ids[:-1])  # cut inside a character
