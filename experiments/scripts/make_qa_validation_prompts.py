"""Generate QA-validation prompts for the 3RScan primary sets.

Reads experiments/scripts/3rscan_0cac762f/keyfact-qa.jsonl and
experiments/scripts/3rscan_1d2f8518/keyfact-qa.jsonl and writes, to
qa_validation_prompts/ at the repo root:

  - one self-contained prompt per scene, covering all of that scene's
    questions in a single pass (so scene_contexts/<scene>/ only needs to be
    attached once per scene, not once per question)
  - one short no-attachment follow-up prompt for the direct scene-A-vs-scene-B
    templating comparison that a per-scene split can't do on its own

Rules content is transcribed from logs/QA_REVIEW_PROMPT_3rscan_primaries.md
(the already-vetted audit prompt for these same two scenes, which reviews
both together) and docs/QA_DESIGN.md. Local-only, no LLM calls.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "qa_validation_prompts"

SCENES = ["3rscan_0cac762f", "3rscan_1d2f8518"]

SCOPE_TABLE = """\
| `question_type` | documents that will be asked |
|---|---|
| `object_relation` | prose, relations_flat, relations_subject, relations_predicate, synthesis, both JSON |
| `relation_structure` | all of the above **plus relations_tree and relations_digest** |
| `relation_aggregate` | all of the `object_relation` row **plus relations_digest** (not relations_tree) |
| `aggregation` | inventory, prose, synthesis, both JSON (**no** relations_* document) |
| `planning` | inventory, prose, relations_flat, relations_subject, relations_predicate, synthesis, both JSON |

(The bare contents list, `inventory`, is additionally asked every spatial question as a
no-information control; that is deliberate and is not a defect to report.)

Concretely, per type:

- `relation_structure` -> answerable from **relations_tree** (draws every support/containment
  edge, nothing else) **and** from **relations_digest** (only its printed summary: deepest-nesting
  depth + printed chain list, printed receptacle counts, top-8 proximity clusters + stated total,
  attribute groups, relation census). If the needed fact is a support edge relations_digest does
  not actually print, the question is out of scope for it and must be `object_relation` instead.
- `relation_aggregate` -> answerable from **relations_digest**'s printed text. Its cluster list is
  capped at the largest 8 (the header total is reliable, membership beyond the printed 8 is not).
  relations_tree carries none of this.
- `object_relation` -> answerable from the raw triple views. **relations_subject caps each
  object's comparative/shared-attribute lists at 3 entries plus "(+N more)"**, so a question
  resting on a truncated comparison may be unanswerable there even though relations_flat prints
  it -- check comparative facts against relations_subject specifically.
- `aggregation` -> answerable from the **bare contents list alone**. No relation document is
  asked, so a question of this type must not need a relation.
- `planning` -> answerable from the raw triple views. Must not require summary-only content
  (cluster membership, census counts, receptacle rankings)."""

RULES_BLOCK = """\
**Ground truth**

- Every key fact must be verifiable in the attached files. Facts are derived from the files by
  hand, never inferred from what a room "would" contain.
- A fact naming *the* largest / deepest / busiest / most-common item must actually be unique. If
  two items tie, a weight-3 fact that names only one of them is a defect: the answer that names
  the other one is correct and will be scored wrong.

**Fact atomicity and weights**

- **One claim per fact -- never bundle.** A fact pairing a conclusion with its supporting numbers
  ("X is peripheral: nearest 5.1 m vs 2.9-3.5 m") is scored NO whenever the answer states the
  conclusion without reciting the figures. Split it.
- **Weight 3 = core** = anything the question explicitly asks for. If the question asks for two
  things ("which object, and how many does it carry?"), both are weight 3, including a count. The
  test is "did the question ask for this", not what category the fact belongs to.
- **Weight 1 = detail** = a supporting fact the question did not ask for. Detail facts cannot
  dilute the score, so they need not be rationed -- but every question must carry **at least one
  weight-3 fact**.
- A **solicited exact count** is weight 3. A magnitude offered only to qualify an identity is
  weight 1.

**Fact phrasing**

- **Each fact must be a standalone checkable proposition** -- true or false on its own, without
  the question or the sibling facts for context. The checking model sees `(answer, fact)` pairs in
  isolation.
- **Facts must be answer-shaped, not document-shaped.** A fact must be derivable from a correct
  *answer about the room*. A fact that describes how one of the documents lays its content out --
  what is "drawn in the tree", what is "listed separately", what a heading says, what rank a
  summary prints -- is not a fact about the room, and only the document that happens to render it
  that way can produce it. Same for the question stem.
- **Avoid exhaustiveness at weight 3.** "X, Y and Z are the only ..." / "no other object ..." is
  the checker's weakest case: it confirms the listed items and misses a wrong added one (false
  YES). Keep universals at weight 1 and make the positive identification the weight-3 fact, or
  de-universalise the phrasing.
- ASCII only; `->` not an arrow character.

**Matched pairs**

Questions sharing a `pair_id` within this scene are two stems for **one** information request,
asked in two registers. The manipulated variable is wording alone:

- the two members must carry **identical `key_facts` lists, verbatim** -- only `text` and
  `question_style` may differ;
- they must ask the **same information request**. Same-answer-by-accident is not enough. A stem
  that hands the reader an object its counterpart makes the reader *find*, or that omits a count
  its counterpart asks for, is a defect;
- `question_style: constructed` means the stem uses the vocabulary the summary documents print in
  their own output -- chain, depth, nesting, receptacle, cluster, clique, census, ranking, group,
  membership, support/containment, directional, comparative;
- `question_style: natural` means a person standing in the room, who has never read any document
  about it, would ask it that way. A "natural" stem still carrying that analysis vocabulary, or
  referring to records/data/files at all, is mis-tagged.

**Planning questions**

- A planning question states a **goal** and the model must infer which objects that goal needs,
  then locate them. Naming the object is a leak -- **and so is naming a use that maps one-to-one
  onto the object** ("somewhere to wash up" for the sink, "a decorative piece" for the decoration).
- A goal whose answer is recoverable from stereotype alone is declined: if the answer is the
  room's only object of the goal's obvious type, a model given no information at all scores full
  marks and the question measures priors rather than the document."""

CHECK_STEPS = """\
For **each** question in the set below, in order:

1. **Restate.** In your own words, what is this question asking for, and which parts are weight-3
   (explicitly solicited) vs weight-1 (supporting detail)?
2. **Ground truth.** For every key fact, quote the proving line(s) from the attached files. A fact
   you cannot prove from an attached file is a finding, not something to reason about from general
   expectations.
3. **Uniqueness.** For every superlative or "the" in a weight-3 fact, confirm nothing ties.
4. **Atomicity and weights.** Bundled facts; weight-3 facts the question did not ask for; solicited
   counts sitting at weight 1; missing weight-3.
5. **Standalone phrasing and answer-shape.** Facts that need their siblings to be meaningful;
   facts (or stems) about a document's layout rather than the room.
6. **Exhaustiveness.** Universals sitting at weight 3.
7. **Type/scope.** Walk the row of the scope table for that question's type and confirm every
   listed document could produce the weight-3 facts. Name any document that could not, and say
   which type the question should carry instead.
8. **Matched-pair integrity** (questions with a `pair_id`): identical facts, same information
   request, register correctly tagged.
9. **Planning rules** (planning questions only): goal-only framing, no one-to-one use leak, not
   stereotype-answerable.
10. **Discrimination.** Would a model shown *no* information about the room answer this correctly
    from general expectations about rooms? If yes, say so -- the question measures priors, not the
    document.
11. **ASCII.**

Then, once per set:

12. **Templating vs the frozen scene.** Compare this scene's questions against the frozen
    `3rscan_02b33dfb` stem list in the appendix (a third scene in the same group, already
    committed). Report every stem that reads as a template instance of one of those rather than
    derived from this scene's own structure: near-identical wording, or a question shape that
    would apply unchanged to any room. Say whether the underlying *fact* is genuinely scene-specific
    even though the wording pattern is shared, since those are different problems. (The matching
    check against the *other* new scene, `{other_scene}`, is a separate follow-up pass once both
    scenes are reviewed -- do not attempt it here, you don't have that scene's files.)"""

OUTPUT_FORMAT = """\
**Section A -- verdict table.** One row per question: `id | type | pair_id | verdict | one-line
reason`. Verdict is `OK`, `FIX` (rule violated, fix proposed in section B), or `FLAG` (judgement
call for the maintainer, no fix proposed).

**Section B -- findings.** One block per finding, most severe first:

- question id(s)
- which rule above, named
- the evidence: quoted lines from the attached files, with the filename
- why it changes a score
- a **complete replacement JSONL line** for the question (or the exact fact-level edit), keeping
  the same `id`, `scene_id`, `question_type` and `pair_id` unless the finding is that one of those
  is wrong. If a pair member changes, give both members.

**Section C -- weight-3 verification appendix.** Every weight-3 fact in the set, with the file and
line that proves it. A fact with no citation is treated as unverified, not as verified.

**Section D -- templating assessment vs the frozen scene**, per check 12.

Do not summarise, do not congratulate, and do not report a rule as satisfied in section B --
section A already carries the OK verdicts. If you cannot verify something from the attached files,
say so explicitly rather than reasoning about what the room probably contains."""

FROZEN_APPENDIX = """\
The third primary scene in the same group, already frozen. Stems only, no key facts -- it is
there for the templating check, not as ground truth.

```
1   relation_structure constructed  What is the longest chain of objects resting on or fixed to one another that ends at the floor?
2   relation_structure constructed  Which single object has the most other objects resting on, standing on, or fixed to it, and roughly how many?
3   object_relation    natural      How shallow is the object stacking in this room? Apart from the floor and the walls, does any single object hold more than one thing?
4   object_relation    natural      The room has two bath cabinets. What is on or in each one?
5   object_relation    natural      What object is resting on the bathtub [11]?
6   object_relation    natural      What is lying on top of the trash can [27]?
7   object_relation    natural      Which objects are built into something else, and what are they built into?
8   object_relation    natural      Three things in the room hang on something else. What are they, and what does each hang on?
9   object_relation    natural      How many objects stand directly on the floor, and name them?
10  object_relation    natural      The toilet paper [29] and the shower curtain [17] are close to each other. What is each attached to?
11  object_relation    natural      What is the ceiling [6] attached to?
12  relation_structure natural      The room has four towels. One is not attached to a wall. Which, and what does it hang on?
13  relation_aggregate constructed  Several fixtures are clustered close together around the bathtub. Which objects make up that group?
14  relation_aggregate constructed  Across all recorded relationships, are most about physical support or something else?
15  aggregation                     How many objects in total does the room contain?
16  aggregation                     Which type of object is the most common, and what is the second most common?
17  aggregation                     Besides walls and towels, is there any object type that appears more than once?
18  planning                        You want to place something next to the bathing area without putting it on the bathing fixture.
19  planning                        You want to store something near where you wash your hands, not near the bathtub.
20  relation_structure constructed  Among support chains passing through a bath cabinet before reaching the floor, which ends at an object ...
21  relation_structure constructed  Restricting attention to walls, which wall carries the most directly supported objects?
22  relation_aggregate constructed  The proximity graph is split into several disconnected groups. How many, and which two ...
23  relation_aggregate constructed  Compare the shared-material clique with the wall-based shared-texture clique.
24  planning                        Two people want to dry off at the same time without either moving from where they stand.
25  planning                        You need to wipe up a small spill next to the bathing area without stepping away from it.
26  relation_structure natural      The towel [5] is not fixed directly to a wall. What is it hanging from, and what is that hanging from?
27  relation_structure natural      Before repainting I need to strip whichever surface has the most fixed to it, the floor aside.
28  relation_aggregate natural      Apart from the crowded area around the main fittings, what two pairs of items sit close together elsewhere?
29  relation_aggregate natural      The bathtub [11] and sink [15] appear to have the same finish. What other item matches them?
30  relation_structure natural      One thing is propped further from the floor than anything else -- hanging off something, which hangs off ...
31  relation_structure natural      I am about to clear this room out. A lot of what I lift comes off one single surface -- which one?
32  relation_structure constructed  Of the four towels, which one's support chain does not terminate at a wall?
33  relation_structure natural      Sink [15] has to come out, and then whatever it is mounted in. What is that unit?
34  relation_structure natural      I am re-tiling and want to start with the wall that has most on it. Which wall, and how much?
35  relation_structure constructed  Walking up the support chain from towel [5], what are the next two links before the floor?
36  relation_structure constructed  In the receptacle ranking, which entry sits immediately below floor [1], and what is its object count?
37  relation_aggregate natural      I want one photograph that captures the busiest corner of this bathroom. Which fittings end up in it?
38  relation_aggregate natural      Thinking about how the things in this bathroom relate, is it mostly a matter of what rests on what?
39  relation_aggregate natural      The things in this room sit in a few separate groups. How many such groups?
40  relation_aggregate natural      Which walls are both made of the same material and finished the same way, and how many is that?
41  relation_aggregate constructed  Which proximity clusters have exactly two members, and what are those members?
42  relation_aggregate constructed  The same-texture attribute group containing bathtub [11] and sink [15] has a third member. Which?
```"""


def load_scene(scene_id):
    path = ROOT / "experiments" / "scripts" / scene_id / "keyfact-qa.jsonl"
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def build_scene_prompt(scene_id, questions, other_scene):
    lines = []
    lines.append(f"# QA validation prompt: {scene_id} ({len(questions)} questions)\n")
    lines.append(
        "You are auditing a set of questions-and-answers that will be used to evaluate large "
        "language models. This scene is a single 3D-scanned room, written out as ten different "
        "text documents (a bare contents list, prose, flat relation triples, triples grouped by "
        "object, triples grouped by relation type, a drawn support tree, a precomputed summary, "
        "two JSON serializations, and a merged document). A language model is shown **one** of "
        "those documents plus **one** question, and answers. A second model then checks the answer "
        "against that question's `key_facts` list, one fact at a time, YES or NO. The score is the "
        "weighted fraction of YES verdicts over the **weight-3** facts only; weight-1 facts are a "
        "diagnostic and never affect that score.\n"
    )
    lines.append(
        "You are **reviewing**, not authoring: do not propose new questions, do not rebalance the "
        "question-type counts, do not rewrite a question that is merely not to your taste. Report "
        "defects against the rules below, with evidence.\n"
    )

    lines.append(f"## Questions under review ({len(questions)})\n")
    lines.append(
        "One JSON object per line, exactly as in `experiments/scripts/" + scene_id + "/keyfact-qa.jsonl`.\n"
    )
    lines.append("```jsonl")
    for q in questions:
        lines.append(json.dumps(q, ensure_ascii=True))
    lines.append("```\n")

    lines.append(f"## Files to attach from `scene_contexts/{scene_id}/`\n")
    lines.append(
        "Required -- these five are the complete ground truth for every question in this set: "
        "`relations_flat.txt`, `relations_tree.txt`, `relations_digest.txt`, "
        "`relations_subject.txt`, `inventory.txt`.\n\n"
        "Optional, add if the context budget allows (answerability checks, no new ground truth): "
        "`prose.txt`, `relations_predicate.txt`, `synthesis.txt`.\n\n"
        "Do not attach `json_mini.json` / `json_pretty.json` -- they are in scope for everything "
        "so they discriminate nothing, and every fact in this set is already covered by the five "
        "required files. Attach only if a specific finding needs the raw source as a tiebreaker.\n"
    )

    lines.append("## The scope model (memorise this before checking anything)\n")
    lines.append(
        "`question_type` is not a label -- it selects which documents get asked the question. If a "
        "question requires information one of those documents does not contain, that document is "
        "scored zero through no fault of its own and the comparison is contaminated.\n"
    )
    lines.append(SCOPE_TABLE + "\n")

    lines.append("## The authoring rules this set must satisfy\n")
    lines.append(RULES_BLOCK + "\n")

    lines.append("## What to check, in order\n")
    lines.append(CHECK_STEPS.format(other_scene=other_scene) + "\n")

    lines.append("## Output format\n")
    lines.append(OUTPUT_FORMAT + "\n")

    lines.append("## Appendix: `3rscan_02b33dfb` stems (for check 12)\n")
    lines.append(FROZEN_APPENDIX + "\n")

    return "\n".join(lines)


def build_followup_prompt(scene_a, qa_a, scene_b, qa_b):
    def stems(qs):
        out = []
        for q in qs:
            style = q.get("question_style", "-")
            pid = q.get("pair_id", "-")
            out.append(f"{q['id']:>3}  {q['question_type']:<18} {style:<11} pair={pid:<3}  {q['text']}")
        return "\n".join(out)

    lines = []
    lines.append("# QA validation follow-up: cross-scene templating (" + scene_a + " vs " + scene_b + ")\n")
    lines.append(
        "Run this **after** both per-scene reviews are done. No file attachments needed -- this "
        "checks question wording, not ground truth.\n"
    )
    lines.append(
        "Each scene's questions are supposed to be derived from *that scene's own* structure, not "
        "copied from the other scene with the identifiers swapped. Compare the two stem lists "
        "below. Report every stem in either list that reads as a template instance of a stem in "
        "the other list rather than independently scene-derived: near-identical wording, or a "
        "question shape that would apply unchanged to any room. For each match, say whether the "
        "underlying *fact* is genuinely scene-specific even though the wording pattern is shared "
        "(that is fine) or whether the question itself was templated (that is a defect).\n"
    )
    lines.append(f"## {scene_a} stems\n")
    lines.append("```")
    lines.append(stems(qa_a))
    lines.append("```\n")
    lines.append(f"## {scene_b} stems\n")
    lines.append("```")
    lines.append(stems(qa_b))
    lines.append("```\n")
    lines.append(
        "## Output format\n\nOne block per finding (empty if none): the two question ids, both "
        "stems quoted, and your verdict on whether the wording is templated and whether the "
        "underlying fact is still scene-specific."
    )
    return "\n".join(lines)


def main():
    OUT_DIR.mkdir(exist_ok=True)

    # clear any stale per-question prompts from a previous granularity
    for stale in OUT_DIR.glob("*_q[0-9][0-9].md"):
        stale.unlink()

    scene_questions = {s: load_scene(s) for s in SCENES}

    for scene_id in SCENES:
        other = [s for s in SCENES if s != scene_id][0]
        prompt = build_scene_prompt(scene_id, scene_questions[scene_id], other)
        (OUT_DIR / f"{scene_id}.md").write_text(prompt, encoding="utf-8")

    followup = build_followup_prompt(
        SCENES[0], scene_questions[SCENES[0]], SCENES[1], scene_questions[SCENES[1]]
    )
    (OUT_DIR / "cross_scene_templating_followup.md").write_text(followup, encoding="utf-8")

    print(f"Wrote {len(SCENES)} scene prompts + 1 follow-up prompt to {OUT_DIR}")


if __name__ == "__main__":
    main()
