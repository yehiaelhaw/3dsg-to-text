"""Local prompt-token counting for supported Ollama responder families."""

from __future__ import annotations

import gzip
import json
import re
import urllib.request
from pathlib import Path
from typing import Callable, Optional

_CACHE_DIR = Path(__file__).resolve().parent / ".token_cache"
_DEFAULT_HOST = "http://localhost:11434"
_TIMEOUT = 180  # the verbose payload is ~4 MB per model over an SSH forward

# tokenizer.ggml.pre selects the pretokenizer regex; the wrong regex can silently change token counts.
_PRE_REGEX = {
    "qwen2": r"(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])"
             r"|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*"
             r"|\s*[\r\n]+|\s+(?!\S)|\s+",
    "tekken": r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*"
              r"[\p{Ll}\p{Lm}\p{Lo}\p{M}]+"
              r"|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+"
              r"[\p{Ll}\p{Lm}\p{Lo}\p{M}]*|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n/]*"
              r"|\s*[\r\n]+|\s+(?!\S)|\s+",
}


# --------------------------------------------------------------------------- #
# Template rendering — supported families
# --------------------------------------------------------------------------- #

def _render_qwen_chatml(prompt: str, system: str) -> str:
    head = f"<|im_start|>system\n{system}<|im_end|>\n" if system else ""
    return head + f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"


def _render_deepseek_r1(prompt: str, system: str) -> str:
    # Jinja template: {{bos_token}}{{system}}<|User|>…<|Assistant|>. The BOS is
    # counted via add_bos_token rather than emitted as text here.
    return f"{system}<｜User｜>{prompt}<｜Assistant｜>"


def _render_mistral(prompt: str, system: str) -> str:
    body = f"{system}\n\n{prompt}" if system else prompt
    return f"[INST]{body}[/INST]"


# Model-tag prefix -> (renderer, expected tokenizer.ggml.pre). Keep specific prefixes before broader ones (qwen2.5 before qwen2).
_FAMILIES: list[tuple[str, Callable[[str, str], str], str]] = [
    ("deepseek-r1",  _render_deepseek_r1, "qwen2"),
    ("qwen2.5",      _render_qwen_chatml, "qwen2"),
    ("qwen2",        _render_qwen_chatml, "qwen2"),
    ("mistral-nemo", _render_mistral,     "tekken"),
]


def _family_for(model: str):
    tag = model.split(":", 1)[0].lower()
    for prefix, render, pre in _FAMILIES:
        if tag.startswith(prefix):
            return render, pre
    return None, None


# --------------------------------------------------------------------------- #
# Ollama metadata, cached by served blob digest
# --------------------------------------------------------------------------- #

def _post(host: str, path: str, payload: dict) -> dict:
    req = urllib.request.Request(
        host.rstrip("/") + path,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=_TIMEOUT) as r:
        return json.loads(r.read().decode("utf-8"))


def _digest(host: str, model: str) -> str:
    """Return the served model digest used as the tokenizer-cache key."""
    with urllib.request.urlopen(host.rstrip("/") + "/api/tags", timeout=30) as r:
        tags = json.loads(r.read().decode("utf-8"))
    for m in tags.get("models", []):
        if m.get("name") == model or m.get("model") == model:
            return (m.get("digest") or "")[:16]
    raise LookupError(f"model {model!r} not present on {host}")


def _cache_path(model: str, digest: str) -> Path:
    safe = re.sub(r"[^A-Za-z0-9._-]", "_", model)
    return _CACHE_DIR / f"{safe}__{digest}.json.gz"


def _spec(host: str, model: str) -> dict:
    """Fetch and cache tokenizer metadata required for local counting."""
    digest = _digest(host, model)
    path = _cache_path(model, digest)
    if path.exists():
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            return json.load(fh)

    show = _post(host, "/api/show", {"model": model, "verbose": True})
    info = show.get("model_info") or {}
    spec = {
        "tokens":     info.get("tokenizer.ggml.tokens"),
        "merges":     info.get("tokenizer.ggml.merges"),
        "token_type": info.get("tokenizer.ggml.token_type") or [],
        "pre":        info.get("tokenizer.ggml.pre"),
        "add_bos":    bool(info.get("tokenizer.ggml.add_bos_token")),
        "system":     show.get("system") or "",
    }
    if not spec["tokens"] or not spec["merges"]:
        raise ValueError(f"{model}: /api/show returned no tokenizer arrays")

    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with gzip.open(tmp, "wt", encoding="utf-8") as fh:
        json.dump(spec, fh)
    tmp.replace(path)
    return spec


def _build(spec: dict, expected_pre: str):
    from tokenizers import AddedToken, Regex, Tokenizer
    from tokenizers import decoders, models, pre_tokenizers

    pre = spec["pre"]
    if pre != expected_pre:
        raise ValueError(f"pretokenizer {pre!r} != expected {expected_pre!r}")

    tokens = spec["tokens"]
    vocab = {t: i for i, t in enumerate(tokens)}
    merges = [tuple(m.split(" ", 1)) for m in spec["merges"]]

    tok = Tokenizer(models.BPE(vocab=vocab, merges=merges, fuse_unk=False,
                               byte_fallback=False, ignore_merges=True))
    tok.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Split(Regex(_PRE_REGEX[pre]), behavior="isolated", invert=False),
        pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
    ])
    tok.decoder = decoders.ByteLevel()

    # CONTROL (3) / USER_DEFINED (4) must match as single units, or template
    # markers like <|im_start|> would be split into their component bytes.
    specials = [AddedToken(tokens[i], special=True, normalized=False)
                for i, t in enumerate(spec["token_type"]) if t in (3, 4)]
    if specials:
        tok.add_special_tokens(specials)
    return tok


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #

_SIZERS: dict[tuple[str, str, str], Callable[[str], Optional[int]]] = {}

#: Set by make_sizer() so callers can report which path they are on.
UNSUPPORTED_REASON: dict[tuple[str, str, str], str] = {}


def make_sizer(backend: str, model: str, options: dict) -> Callable[[str], Optional[int]]:
    """Return an exact prompt-token sizer when supported, otherwise a sizer that returns None. Setup failures are non-fatal."""
    host = (options or {}).get("host", _DEFAULT_HOST)
    key = (backend, model, host)
    if key in _SIZERS:
        return _SIZERS[key]

    def unsupported(reason: str):
        UNSUPPORTED_REASON[key] = reason
        _SIZERS[key] = lambda _prompt: None
        return _SIZERS[key]

    if backend != "ollama":
        return unsupported(f"backend {backend!r} is not ollama")

    render, expected_pre = _family_for(model)
    if render is None:
        return unsupported(f"{model!r} is not a replay-validated family")

    try:
        spec = _spec(host, model)
        tok = _build(spec, expected_pre)
    except Exception as exc:                      # noqa: BLE001 — never fatal
        return unsupported(f"{type(exc).__name__}: {exc}")

    system, bos = spec["system"], (1 if spec["add_bos"] else 0)

    def sizer(prompt: str) -> int:
        return len(tok.encode(render(prompt, system)).ids) + bos

    _SIZERS[key] = sizer
    return sizer
