"""Loading a Hugging Face language model and a few helpers that need it."""
import math
from dataclasses import dataclass

from .errors import AppError

DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"  # the model used in all demos
SUGGESTED_MODELS = [DEFAULT_MODEL, "Qwen/Qwen2.5-1.5B-Instruct", "facebook/opt-1.3b"]


@dataclass
class LanguageModel:
    name: str
    model: object
    tokenizer: object
    vocab_size: int
    device: str
    use_chat_template: bool

    def describe(self):
        return {"name": self.name, "vocab_size": self.vocab_size, "device": self.device,
                "use_chat_template": self.use_chat_template}

    def tokenize(self, text):
        """Text -> token ids, the way a detector sees a text (no special tokens)."""
        return list(self.tokenizer(text, add_special_tokens=False).input_ids)

    def decode(self, tokens):
        return self.tokenizer.decode(list(tokens), skip_special_tokens=True)

    def pieces(self, tokens):
        """Text of every token, for coloring. Decoding tokens one by one breaks characters
        that span several tokens (e.g. Polish letters or emoji in byte-level BPE), so the
        text is decoded as a growing prefix and each token gets the new part. A token that
        ends in the middle of a character gets an empty piece."""
        tokens = list(tokens)
        out = []
        prev = ""
        for i in range(len(tokens)):
            text = self.decode(tokens[:i + 1])
            if text.endswith("�"):  # incomplete character, finished by a later token
                out.append("")
            elif text.startswith(prev):
                out.append(text[len(prev):])
                prev = text
            else:  # should not happen, but never lose text
                out.append(self.decode([tokens[i]]))
                prev = text
        joined = "".join(out)
        full = self.decode(tokens)
        if out and full != joined and full.startswith(joined):  # text ends with an unfinished character
            out[-1] += full[len(joined):]
        return out

    def encode_prompt(self, prompt):
        """Same prompt encoding as generate() in the watermark packages."""
        import torch
        if self.use_chat_template and getattr(self.tokenizer, "chat_template", None):
            messages = [{"role": "user", "content": prompt}]
            ids = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
            if not torch.is_tensor(ids):  # newer transformers versions return a dict
                ids = ids["input_ids"]
        else:
            ids = self.tokenizer(prompt, return_tensors="pt").input_ids
        return ids[0].tolist()

    def perplexity(self, prompt, new_tokens):
        """Perplexity of the generated tokens given the prompt (same as in the demos).
        Scored by the same model that generated the text, so it is only a rough check."""
        import torch
        if not new_tokens:
            return None
        prompt_ids = self.encode_prompt(prompt)
        device = next(self.model.parameters()).device
        ids = torch.tensor([prompt_ids + list(new_tokens)], device=device)
        labels = ids.clone()
        labels[0, :len(prompt_ids)] = -100  # do not score the prompt
        with torch.no_grad():
            loss = self.model(ids, labels=labels).loss.item()
        return math.exp(loss) if math.isfinite(loss) else None


def load_model(name, use_chat_template=True):
    """Same settings as load_model() in the demos: GPU with float16 if available, else CPU."""
    name = (name or "").strip()
    if not name:
        raise AppError("Nie podano nazwy modelu.", f"Wpisz np. {DEFAULT_MODEL} albo wskaż folder z modelem.")
    try:
        import torch
        import transformers
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise AppError(f"Brakuje biblioteki {exc.name}.", "Zainstaluj zależności: pip install -r requirements.txt") from exc

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    major, minor = (int(x) for x in transformers.__version__.split(".")[:2])
    dtype_arg = "dtype" if (major, minor) >= (4, 56) else "torch_dtype"  # renamed in transformers 4.56
    try:
        tokenizer = AutoTokenizer.from_pretrained(name)
        model = AutoModelForCausalLM.from_pretrained(name, **{dtype_arg: dtype}).to(device).eval()
    except OSError as exc:
        raise AppError(f"Nie udało się wczytać modelu „{name}”.",
                       "Sprawdź nazwę i połączenie z internetem (za pierwszym razem model jest pobierany "
                       "z huggingface.co, Qwen2.5-0.5B to ok. 1 GB) albo wskaż folder z pobranym modelem.\n"
                       f"Szczegóły: {str(exc).splitlines()[0]}") from exc
    except Exception as exc:
        raise AppError(f"Błąd podczas ładowania modelu „{name}”: {type(exc).__name__}: {exc}",
                       "Spróbuj mniejszego modelu albo sprawdź, ile jest wolnej pamięci.") from exc
    vocab = int(model.get_output_embeddings().weight.shape[0])  # can be larger than len(tokenizer)
    use_chat = bool(use_chat_template and getattr(tokenizer, "chat_template", None))
    return LanguageModel(name, model, tokenizer, vocab, device, use_chat)
