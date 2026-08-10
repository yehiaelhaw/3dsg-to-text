"""context_based.py — Metrics that require the source context (scene graph text)."""

from __future__ import annotations

import re

from evaluation.core import KeyFact
from evaluation.llm.base import LLMProvider

_FAITHFULNESS_PROMPT = """\
You are a strict faithfulness judge for a 3D scene-graph QA system.

CONTEXT (scene graph):
{context}

QUESTION:
{question}

ANSWER:
{answer}

Task: Score how well the ANSWER is supported by the CONTEXT on a scale from 0.0 to 1.0,
where 1.0 means every claim is directly supported and 0.0 means the answer contradicts or invents information not present.
Award partial credit proportional to the fraction of claims that are supported.

Pay special attention to room-object assignments: if the answer states that a specific object is in a specific room, verify that the context lists that object under that room — not just that the object exists somewhere in the scene.

Respond with ONLY a single decimal number between 0.0 and 1.0. No explanation."""


def faithfulness(
    question: str,
    context: str,
    answer: str,
    judge: LLMProvider,
) -> float:
    prompt = _FAITHFULNESS_PROMPT.format(
        context=context.strip(),
        question=question.strip(),
        answer=answer.strip(),
    )
    result = judge.generate(prompt)
    return _parse_score(result.text)


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
    # prefer explicit SCORE: tag to avoid grabbing numbers from mid-explanation
    tagged = re.search(r"SCORE:\s*([01](?:\.\d+)?|\.\d+)", text, re.IGNORECASE)
    if tagged:
        return round(min(max(float(tagged.group(1)), 0.0), 1.0), 4)
    match = re.search(r"\b([01](?:\.\d+)?|\.\d+)\b", text.strip())
    if match:
        return round(min(max(float(match.group(1)), 0.0), 1.0), 4)
    raise ValueError(f"Could not parse faithfulness score from judge output: {text!r}")


_RUBRIC_PROMPT = """\
You are a fact-checking judge.

QUESTION: {question}
MODEL ANSWER: {answer}

For each numbered fact below, answer YES if the MODEL ANSWER contains or clearly implies it, or NO if it does not.
Start each line with the number and YES or NO, then add a brief reason. Example:
1. YES — the answer explicitly states Room 22.0
2. NO — no distance value is mentioned

Facts:
{numbered_facts}"""


# Facts split into two tiers by weight: the *core* tier (weight > 1) is what the
# question explicitly asks for and is the ONLY thing the primary answer_correctness
# scores; the *detail* tier (weight <= 1) is supporting information the question did
# not ask for -- scored separately as a diagnostic (answer_correctness_detail), never
# folded into the primary. Splitting here (rather than one Sum(weights)/Sum(all))
# means a wrong core answer can no longer be masked by volunteered supporting detail.
# The threshold is > 1 (not == 3) so a future intermediate weight still counts as core
# and numeric weighting is preserved *within* the core tier.
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


# A verdict word only counts when the line ends there or continues into a reason.
# Without that guard a fact RESTATEMENT whose text begins with "No"/"Yes" -- e.g.
# "4. No other table is included in this shape group" -- parses as a verdict, and
# silently records the OPPOSITE of the verdict the judge gives on the next line.
# Nothing raises, so the row still looks scored. Seen on gemini-2.5-flash, which
# restates each fact on the numbered line and puts YES/NO on an indented bullet.
_NUM_VERDICT = re.compile(r"^[\-\s]*(\d+)[.)]\s*(YES|NO)\b(.*)$", re.IGNORECASE)
_REASON_SEP = re.compile(r"^\s*(?:[-–—:;,.!]|$)")
_SUB_VERDICT = re.compile(r"^[\-•\*\s]*(YES|NO)\b", re.IGNORECASE)

# Only an affirmation earns the fact. A numbered line carrying a short verdict
# phrase that is NOT "YES" -- "MAYBE", "PARTIALLY YES", "IMPLIED YES" -- means the
# judge considered the fact and declined to affirm it, which scores as NO. This is
# deliberately vocabulary-free: no list of hedge words to maintain, and hedges
# nobody has seen yet resolve the same way. The prompt asks for YES or NO, so a
# reply that is neither has not met the bar the prompt sets.
#
# NOT the same as a fact the judge never mentioned -- that still raises below.
# "evaluated but not affirmed" and "never evaluated" are different claims, and
# scoring the second as absent is the silent-understatement bug the fail-closed
# rule exists to catch (it caught a truncated verdict list at 4/10632 rows).
#
# The separator requirement is what keeps a fact RESTATEMENT from being read as a
# verdict: "3. Bath cabinet [16]" has no reason clause, so it stays pending and
# waits for the real verdict on the following line.
_NUM_ANY = re.compile(r"^[\-\s]*(\d+)[.)]\s*(.+)$")
_SEP_SPLIT = re.compile(r"\s*[-–—:;,.!]\s*")
_MAX_VERDICT_WORDS = 3


def _parse_rubric(text: str, n: int) -> list[bool]:
    results = [False] * n
    seen: set[int] = set()
    # Index of the most recent numbered fact still awaiting a verdict, so a
    # verdict on the following indented line can be attributed to it.
    pending: int | None = None
    for line in text.splitlines():
        # Tolerate markdown emphasis / bullets the judge sometimes adds, e.g.
        # "1.  **YES** - ..." or "- 1) `NO`": strip emphasis chars before matching.
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
        # EVERY fact must be ADDRESSED. A fact the judge skipped entirely is a
        # format failure, not an implicit NO: defaulting it to absent silently
        # understates the answer and biases AC downward, and it does so invisibly
        # because the record still looks scored. Raise so the record errors and is
        # re-judged on resume -- the same fail-closed contract the rest of the
        # pipeline follows. Both the no-verdicts-at-all and the truncated-list
        # cases land here. (An addressed-but-hedged fact does NOT: see _NUM_ANY.)
        missing = [i + 1 for i in range(n) if i not in seen]
        raise ValueError(
            f"Judge gave no verdict at all for fact(s) {missing} of {n}: {text!r}"
        )
    return results
