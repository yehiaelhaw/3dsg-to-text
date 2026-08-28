"""judging.py — the LLM-judge scoring layer: prompts, scores, verdict parsing."""

from __future__ import annotations

import re

from evaluation.core import KeyFact
from evaluation.llm.base import LLMProvider

_CORRECTNESS_PROMPT = """\
You are a correctness judge for a 3D scene-graph QA system.

QUESTION:
{question}

GROUND TRUTH ANSWER:
{ground_truth}

MODEL ANSWER:
{answer}

Task: Score how correct the MODEL ANSWER is relative to the GROUND TRUTH on a scale from 0.0 to 1.0,
where 0.0 means completely wrong and 1.0 means fully correct.

First identify what the QUESTION explicitly asks for. Then check whether the MODEL ANSWER correctly provides that.
Award partial credit proportional to the fraction of explicitly-requested facts answered correctly.
The GROUND TRUTH may contain supporting evidence beyond what the question requires — do not penalize for omitting details that the question did not ask for.
Order of items does not matter unless the question explicitly asks for a ranking.
Ignore stylistic differences.
Respond with ONLY a single decimal number between 0.0 and 1.0. No explanation."""


def answer_correctness(
    question: str,
    answer: str,
    ground_truth: str,
    judge: LLMProvider,
) -> float:
    prompt = _CORRECTNESS_PROMPT.format(
        question=question.strip(),
        ground_truth=ground_truth.strip(),
        answer=answer.strip(),
    )
    result = judge.generate(prompt)
    return _parse_score(result.text)


def _parse_score(text: str) -> float:
    # Prefer an explicit SCORE tag before falling back to the first numeric score.
    tagged = re.search(r"SCORE:\s*([01](?:\.\d+)?|\.\d+)", text, re.IGNORECASE)
    if tagged:
        return round(min(max(float(tagged.group(1)), 0.0), 1.0), 4)
    match = re.search(r"\b([01](?:\.\d+)?|\.\d+)\b", text.strip())
    if match:
        return round(min(max(float(match.group(1)), 0.0), 1.0), 4)
    raise ValueError(f"Could not parse judge score from output: {text!r}")


_RUBRIC_PROMPT = """\
You are a fact-checking judge.

QUESTION: {question}
MODEL ANSWER: {answer}

For each numbered fact below, answer YES if the MODEL ANSWER contains or clearly implies it, or NO if it does not.
A fact that names a specific object, ID, or room requires that same specific object, ID, or room to
appear in the ANSWER -- if the ANSWER instead says it lacks the information and offers only generic,
hypothetical, or categorical speculation (e.g. "it could be a chair or a cabinet" without naming
which one), that does NOT satisfy a fact about a specific named object, even if the general category
is mentioned.
A fact that requires multiple separate conditions to ALL hold (for example: belonging to several
named groups or categories, satisfying several stated criteria, or being true in more than one
respect) is YES only if the ANSWER supports every one of those conditions individually. Check each
condition one at a time before answering. If your own reasoning finds that even one required
condition is unsupported or not mentioned, your verdict for that fact must be NO -- never answer YES
when your stated reasoning identifies a missing or unsupported condition.
If the ANSWER explores several candidate objects, relations, or paths while reasoning but then states
a specific final, complete conclusion (a final list, a single named answer, a definitive count), only
credit a fact against that final conclusion. A fact about an item that appears only in an
intermediate or exploratory step, and is not included in the final conclusion, is NOT supported --
even if that item was mentioned alongside others that did make the final conclusion.
Start each line with the number and YES or NO, then add a brief reason. Example:
1. YES — the answer explicitly states Room 22.0
2. NO — no distance value is mentioned

Facts:
{numbered_facts}"""


# Core tier (weight > 1) is what the question asks for and is the only thing answer_correctness scores; detail tier is diagnostic only, never folded in -- so a wrong core answer can't be masked by volunteered detail.
_CORE_MIN_WEIGHT = 1.0


def rubric_correctness(
    question: str,
    answer: str,
    key_facts: list[KeyFact],
    judge: LLMProvider,
) -> tuple[float, float | None, str]:
    """Return (core, detail, judge_text).

    core   = Sum(w of YES core facts)   / Sum(w of all core facts)     -- the primary AC
    detail = Sum(w of YES detail facts) / Sum(w of all detail facts)   -- diagnostic,
             or None when the question has no detail (weight <= 1) facts.
    """
    numbered = "\n".join(f"{i + 1}. {kf.fact}" for i, kf in enumerate(key_facts))
    prompt = _RUBRIC_PROMPT.format(
        question=question.strip(), answer=answer.strip(), numbered_facts=numbered
    )
    text = judge.generate(prompt).text
    present = _parse_rubric(text, len(key_facts))

    def tier_score(facts: list[tuple[KeyFact, bool]]) -> float | None:
        total = sum(kf.weight for kf, _ in facts)
        if total <= 0:
            return None
        return round(sum(kf.weight for kf, p in facts if p) / total, 4)

    scored = list(zip(key_facts, present))
    core   = [(kf, p) for kf, p in scored if kf.weight > _CORE_MIN_WEIGHT]
    detail = [(kf, p) for kf, p in scored if kf.weight <= _CORE_MIN_WEIGHT]

    core_score = tier_score(core)
    if core_score is None:
        # No core (weight > 1) fact -- an authoring error the loader should have
        # caught (dataset.load validates this). Fail loudly rather than silently
        # scoring the primary metric off detail facts.
        raise ValueError(
            "rubric_correctness: question has no core (weight > 1) key fact; "
            "the primary answer_correctness is undefined"
        )
    return core_score, tier_score(detail), text


# Accept YES/NO only when it is syntactically a verdict, not part of a fact restatement.
_NUM_VERDICT = re.compile(r"^[\-\s]*(\d+)[.)]\s*(YES|NO)\b(.*)$", re.IGNORECASE)
_REASON_SEP = re.compile(r"^\s*(?:[-–—:;,.!]|$)")
_SUB_VERDICT = re.compile(r"^[\-•\*\s]*(YES|NO)\b", re.IGNORECASE)

# Addressed but non-YES verdicts score NO; completely unaddressed facts raise below.
_NUM_ANY = re.compile(r"^[\-\s]*(\d+)[.)]\s*(.+)$")
_SEP_SPLIT = re.compile(r"\s*[-–—:;,.!]\s*")
_MAX_VERDICT_WORDS = 3


def _parse_rubric(text: str, n: int) -> list[bool]:
    results = [False] * n
    seen: set[int] = set()
    # Numbered fact awaiting a verdict on a following indented line.
    pending: int | None = None
    for line in text.splitlines():
        # Strip common Markdown emphasis before parsing.
        clean = line.replace("*", "").replace("`", "").replace("_", "").strip()
        m = _NUM_VERDICT.match(clean)
        if m:
            idx = int(m.group(1)) - 1
            if 0 <= idx < n:
                if _REASON_SEP.match(m.group(3)):
                    results[idx] = m.group(2).upper() == "YES"
                    seen.add(idx)
                    pending = None
                else:
                    # Restatement, not a verdict -- wait for the real one below.
                    pending = idx
            continue
        numbered = _NUM_ANY.match(clean)
        if numbered:
            idx = int(numbered.group(1)) - 1
            parts = _SEP_SPLIT.split(numbered.group(2), 1)
            phrase = parts[0].strip()
            addressed = (len(parts) > 1 and phrase
                         and len(phrase.split()) <= _MAX_VERDICT_WORDS)
            if 0 <= idx < n and idx not in seen and addressed:
                results[idx] = False   # considered, not affirmed -> NO
                seen.add(idx)
                pending = None
            else:
                pending = idx if 0 <= idx < n and idx not in seen else None
            continue
        # Indented continuation under a numbered fact that has no verdict yet.
        if pending is not None and pending not in seen and line[:1] in (" ", "\t", "*", "-"):
            sub = _SUB_VERDICT.match(clean)
            if sub:
                results[pending] = sub.group(1).upper() == "YES"
                seen.add(pending)
                pending = None
    if n > 0 and len(seen) < n:
        # Missing verdicts are errors, not implicit NOs, so the record can be re-judged.
        missing = [i + 1 for i in range(n) if i not in seen]
        raise ValueError(
            f"Judge gave no verdict at all for fact(s) {missing} of {n}: {text!r}"
        )
    return results
