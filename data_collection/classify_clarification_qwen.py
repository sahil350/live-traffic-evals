"""
Step 5 prep: classify each of the 84 working-set records as ANSWERABLE or
NEEDS_CLARIFICATION -- whether a careful deep-research agent's appropriate first
move is to proceed directly, or to ask a clarifying question before proceeding.

Local Qwen2.5 7B via Ollama, reason-before-label (same pattern as
classify_candidates_qwen.py). The worked examples were taken from the project's own working set
(omitted from this release), not invented -- same practice as every other
rubric in this project.

v2 (see outputs/clarification_classification_v1.csv, archived): v1
over-flagged NEEDS_CLARIFICATION at 78.6% (66/84) despite the rubric's own
"don't treat breadth as underspecification" instruction -- manual review
found a systematic pattern of demanding depth/audience-level/scope-boundary/
evaluation-criteria parameters a careful agent could reasonably choose
itself, and of judging turn 1 in isolation even when turn 2 already resolved
an apparent ambiguity. v2 added an explicit "these do NOT count" list and two
worked examples targeting those two failure modes --
dropped the rate to 57.1% (48/84), correctly fixing both worked-example
targets, but manual review found the same "criteria/scope/timeframe" pattern
still surviving on non-identical cases -- the model applied
the new rule to lookalikes of the worked examples, not as a general
principle.

v3 (this version, see outputs/clarification_classification_v2.csv,
archived): adds a forced self-check step (MISSING -> CHECK against the
do-NOT-count list -> REASON -> LABEL) so the model has to explicitly test its
own stated gap against the list before committing to a label, plus a 5th
worked example targeting the exact recurring "criteria for
identifying X" failure phrase.

Writes incrementally (one row at a time, flushed) and is resumable -- rerunning
skips case_ids already present in the output CSV.

Usage:
    python classify_clarification_qwen.py
"""

import csv
import os
import sys
import time

import requests

OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "../outputs")
OLLAMA_MODEL = "qwen2.5:7b"
OLLAMA_URL = "http://localhost:11434/api/chat"
MAX_RETRIES = 3

RUBRIC = """You are classifying a user query for a deep-research evaluation set.
Judge whether the query is sufficiently specified to act on directly,
or whether a careful research agent's appropriate first move would be
to ask a clarifying question before proceeding.

Definitions:
- ANSWERABLE: The query gives enough information (topic, scope, and any
necessary parameters) for a careful research agent to proceed directly --
either it's fully specified, or any gaps can be filled with reasonable,
low-stakes assumptions that wouldn't materially change what a good answer
looks like.
- NEEDS_CLARIFICATION: Answering well genuinely requires information the
query doesn't provide and can't reasonably be assumed -- e.g. the query is
ambiguous between substantively different interpretations that would produce
different answers, or omits a parameter/goal/audience that changes the shape
of a good response.

Do not treat breadth, ambition, or difficulty as underspecification.
Only flag NEEDS_CLARIFICATION if answering well genuinely requires
information the query doesn't provide and can't reasonably be assumed.

The following do NOT by themselves count as needing clarification -- a
careful agent can make a reasonable default choice and proceed:
- Missing depth, detail level, audience level, or desired length. Default
  to a thorough, general-audience treatment.
- Missing exact scope boundaries on an inherently broad topic. Default to
  the most natural, common reading of the topic.
- Missing evaluation criteria, categorization rules, or parameters the
  agent could reasonably define itself as part of doing the work (e.g.
  "what criteria count as a good niche" is part of the analysis the agent
  is being asked to produce, not a prerequisite for starting it).
- An apparent ambiguity in turn 1 that a follow-up turn 2 already narrows
  or resolves. Judge the conversation as a whole, not turn 1 in isolation.

Only flag NEEDS_CLARIFICATION when the query is ambiguous between
substantively different interpretations that would change the shape of a
good answer and no reasonable default exists (e.g. a term with several
unrelated meanings and no context to pick one), or when proceeding
genuinely requires a private fact only the user has (their budget, their
specific dataset, their organization's constraints) that cannot be
reasonably guessed.

Before you answer, work through this checklist:
1. Name the specific information gap, if any.
2. Check that gap against the "do NOT count" list above. If it matches any
   item on that list, the query is ANSWERABLE regardless of how broad or
   thin it looks -- do not flag NEEDS_CLARIFICATION for a gap that's on
   that list.
3. Only if the gap does NOT match anything on that list -- it's a genuine
   ambiguity between substantively different interpretations, or a private
   fact only the user has -- is NEEDS_CLARIFICATION correct.

Examples:

[Worked examples are omitted from this public release. They were excerpts of real conversations from a gated dataset (LMSYS-Chat-1M and Chatbot Arena), which cannot be redistributed. Write your own, in the same shape as the output format below, before running. Without them the labels will differ from those reported in the paper.]

Respond in EXACTLY this format, nothing else:
MISSING: name the specific information gap, or "none"
CHECK: does it match the do-NOT-count list? state which item, or "no match"
REASON: one sentence, cite the specific ambiguity or confirm specificity
LABEL: ANSWERABLE or NEEDS_CLARIFICATION
"""


def classify(prompt, turn2, retries=MAX_RETRIES):
    user_msg = f"Query (turn 1): {prompt}\n"
    user_msg += f"Follow-up (turn 2, if present): {turn2}\n" if turn2 else "Follow-up (turn 2, if present): [none]\n"

    for attempt in range(retries):
        try:
            resp = requests.post(
                OLLAMA_URL,
                json={
                    "model": OLLAMA_MODEL,
                    "messages": [
                        {"role": "system", "content": RUBRIC},
                        {"role": "user", "content": user_msg},
                    ],
                    "stream": False,
                    "options": {"temperature": 0},
                },
                timeout=120,
            )
            resp.raise_for_status()
            content = resp.json()["message"]["content"].strip()
            label_line = next((l for l in content.splitlines() if l.upper().startswith("LABEL:")), "LABEL: PARSE_ERROR")
            reason_line = next((l for l in content.splitlines() if l.upper().startswith("REASON:")), "REASON: (none)")
            label = "NEEDS_CLARIFICATION" if "NEEDS_CLARIFICATION" in label_line.upper() else "ANSWERABLE"
            reason = reason_line.split(":", 1)[1].strip() if ":" in reason_line else ""
            return label, reason
        except Exception as e:
            if attempt == retries - 1:
                return "ERROR", str(e)
            time.sleep(3)


def load_working_set(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def load_already_done(out_path):
    if not os.path.exists(out_path):
        return set()
    with open(out_path) as f:
        return {r["case_id"] for r in csv.DictReader(f)}


if __name__ == "__main__":
    in_path = os.path.join(OUTPUT_DIR, "working_set_151sample.csv")
    out_path = os.path.join(OUTPUT_DIR, "clarification_classification.csv")

    rows = load_working_set(in_path)
    done = load_already_done(out_path)
    todo = [r for r in rows if r["case_id"] not in done]

    print(f"{len(rows)} working-set rows, {len(done)} already classified, {len(todo)} remaining")

    write_header = not os.path.exists(out_path)
    with open(out_path, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["case_id", "source", "prompt", "turn2", "clarification_label", "clarification_reason"])
            f.flush()

        t0 = time.time()
        for i, row in enumerate(todo):
            label, reason = classify(row["prompt"], row.get("turn2", ""))
            writer.writerow([row["case_id"], row["source"], row["prompt"], row.get("turn2", ""), label, reason])
            f.flush()

            if (i + 1) % 10 == 0 or (i + 1) == len(todo):
                elapsed = time.time() - t0
                rate = (i + 1) / elapsed if elapsed > 0 else 0
                remaining = (len(todo) - i - 1) / rate if rate > 0 else float("inf")
                print(f"  {i+1}/{len(todo)} done ({rate:.2f} rows/s, ~{remaining/60:.1f} min remaining)")
                sys.stdout.flush()

    print("\nDone.")
