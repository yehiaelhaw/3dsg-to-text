"""token_count.py — exact prompt token counts for Ollama responders.

The runner needs to know, *before* spending a generation, whether a prompt fits
the responder's context window. The old answer was `len(prompt) // 4`: a true
lower bound (measured 0.597-0.914 of the real count over 2,103 cells, never
above it), but loose enough that a prompt had to be ~36% past the window before
it tripped. Everything between 100% and 136% was sent anyway and had to be
caught after the fact — and the post-call guard cannot be relied on there,
because an overflowing prompt is exactly the case where this server misreports
`prompt_eval_count` (the 16,386 signature).

This module closes that gap by tokenizing the prompt locally with the *same*
tokenizer the server uses, so the pre-call number is not an estimate.

How the tokenizer is obtained
-----------------------------
`POST /api/show {"verbose": true}` returns the tokenizer straight out of the
GGUF blob being served — full `tokenizer.ggml.tokens` / `.merges` / `.token_type`
arrays, plus the pretokenizer id, the BOS policy, the chat template and the
model's default system prompt. That is the authoritative source: not tiktoken,
not a HuggingFace download (Mistral-Nemo and Llama-3.1 are gated repos anyway),
and not a fitted ratio. Rebuilding from those arrays reproduces the server's
tokenization rather than approximating it.

What Ollama adds after the prompt leaves the runner
---------------------------------------------------
More than you would guess, and it is all reproduced here:
  * the model's chat template (`<|im_start|>user` … etc.);
  * a *default system prompt* baked into the Modelfile — qwen2.5 ships
    "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.",
    roughly 30 tokens on every single call;
  * BOS, whose presence varies by model (qwen2.5 no, deepseek-r1/mistral yes).

Note `/api/generate` renders the prompt as a one-message chat: of the models
here only qwen2.5 even has a bare `.Prompt` template branch, and for it both
branches emit byte-identical text, so the single rendering below is correct
either way.

Scope
-----
Deliberately narrow. `_FAMILIES` is an allowlist of the responder families whose
counts were verified by replay against 13,320 recorded cells (max delta 0). An
unrecognised model returns None and the caller falls back to the heuristic —
a wrong exact count would be worse than an honest estimate, so anything
unvalidated is not guessed at. `evaluation/tests/test_token_count_replay.py`
re-proves the equality and is what should gate adding a family here.
"""

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

# Pretokenizer regexes from llama.cpp's llm_tokenizer_bpe, keyed by
# tokenizer.ggml.pre. These decide where the BPE is allowed to merge, so they
# are load-bearing: the wrong one still produces plausible-looking counts.
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
# Template rendering — one function per validated family
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


# model-tag prefix -> (renderer, expected tokenizer.ggml.pre). Matched longest
# first, so "deepseek-r1" wins over a bare family check.
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
    """Served blob digest — the cache key. Changes whenever the model is
    re-pulled, which is also when template/system/vocab could change."""
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
    """Everything needed to tokenize, fetched once and cached on disk.

    Keyed by digest, so a re-pull silently invalidates rather than serving a
    stale vocab. Gzipped: the raw arrays are ~4 MB per model.
    """
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
    """Return `prompt -> exact token count`, or `prompt -> None` if unsupported.

    Built once per (backend, model, host) and memoised: constructing the
    tokenizer costs ~0.2 s and encoding a 186 KB context ~0.09 s, so a run pays
    the build once and nothing measurable per cell.

    Never raises. Any failure — server unreachable, `tokenizers` missing, an
    unrecognised family — degrades to the None-sizer and the caller keeps its
    heuristic.
    """
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
