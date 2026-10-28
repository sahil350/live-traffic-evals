"""
Taxonomy pass v2, replacing the ad-hoc v1 taxonomy (classify_taxonomy_qwen.py,
archived under outputs/taxonomy_v1_adhoc_2026-08-31/). v1's "A3 self-directed
learning" category was catching single-turn factoid/riddle/explanation
requests that happened to mention an academic-sounding topic -- e.g. a
step-by-step riddle solve, "explain CRISPR ethics," a sentence-rephrasing
task -- because v1's definition ("orienting to a field to build personal
expertise") didn't require any actual synthesis-across-sub-topics signal.

v2 fixes this by grounding Level 1 in an established, citable taxonomy
instead of an invented one: Gary Marchionini's Lookup / Learn / Investigate
distinction (Marchionini, G. "Exploratory search: from finding to
understanding." Communications of the ACM 49.4 (2006): 41-46). This is the
same intellectual lineage as Kuhlthau's ISP (both treat information-seeking
as a process, not a single query), so it's a more defensible methodology for
the paper than the ad-hoc six-way split, not just a bug fix.

Level 2 (domain/purpose) keeps the categories from v1 that held up well on
manual review (B/C/D/E mapped cleanly onto real patterns): Academic/
scientific, Business/market, Self-directed skill-building, Personal/
consumer, Technical/implementation, Generic/templated, Other.

Runs over the same 2,144 Stage-2 positives as v1. Writes to
lmsys_taxonomy_v2.csv / arena_taxonomy_v2.csv (NOT the v1 filenames --
different rubric, keep outputs distinguishable, same discipline as
archiving the pre-rubric-tightening pilot data earlier in this project).

Usage:
    python classify_taxonomy_v2_marchionini.py
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

TAXONOMY_RUBRIC = """You are categorizing a two-turn snippet of a real chat conversation using \
Gary Marchionini's taxonomy of information-seeking activity (Marchionini, 2006, "Exploratory \
search: from finding to understanding").

Assign exactly ONE Level 1 category -- this is about the DEPTH OF SEARCH PROCESS, not the topic:

LOOKUP -- a single, well-defined query that has ONE satisfying answer. This includes long or \
detailed answers, multi-step reasoning toward one correct answer, single concept explanations, \
single edits/rewrites/translations, and single recommendations -- as long as one response from \
the assistant settles the request. Do NOT be fooled by academic-sounding topics or step-by-step \
reasoning chains: solving a logic puzzle "step by step," explaining one concept in depth, or \
rephrasing a paragraph are all LOOKUP even if the subject matter sounds scholarly, because they \
don't require synthesizing across multiple distinct sub-topics or sources. This also applies when \
the prompt wraps a closed-form task (a riddle, a coding exercise, a puzzle) in roleplay language \
like "act as a researcher and investigate the flaws in these answers" or "act as a resolver and \
improve the answer" -- generating candidate answers to ONE puzzle and critiquing/refining them is \
still LOOKUP. The words "researcher"/"investigate"/"analyze" in the prompt's own roleplay framing \
do NOT by themselves indicate INVESTIGATE-level activity -- judge the underlying task (is there \
one puzzle/problem with a checkable answer?), not the vocabulary the prompt uses to describe the \
role the assistant should play.

LEARN -- the user is building their own understanding of a topic by exploring/synthesizing \
across MULTIPLE distinct sub-topics or facets, often iteratively narrowing or expanding, for \
personal understanding or skill-building -- not to produce a formal output for someone else.

INVESTIGATE -- exploratory engagement explicitly aimed at a formal outcome: a research project, \
thesis, business decision, published output, or plan. Involves critical assessment or synthesis \
FOR that stated purpose, not just curiosity.

Then assign a Level 2 domain (applies regardless of Level 1, for completeness):
Academic/scientific | Business/market | Self-directed skill-building | Personal/consumer | \
Technical/implementation | Generic/templated | Other

Examples:

[Worked examples are omitted from this public release. They were excerpts of real conversations from a gated dataset (LMSYS-Chat-1M and Chatbot Arena), which cannot be redistributed. Write your own, in the same shape as the output format below, before running. Without them the labels will differ from those reported in the paper.]

Respond in EXACTLY this format, nothing else:
REASON: one sentence
LEVEL1: LOOKUP, LEARN, or INVESTIGATE
LEVEL2: Academic/scientific, Business/market, Self-directed skill-building, Personal/consumer, \
Technical/implementation, Generic/templated, or Other
"""

VALID_LEVEL1 = {"LOOKUP", "LEARN", "INVESTIGATE"}
VALID_LEVEL2 = {
    "academic/scientific", "business/market", "self-directed skill-building",
    "personal/consumer", "technical/implementation", "generic/templated", "other",
}
LEVEL2_CANONICAL = {
    "academic/scientific": "Academic/scientific",
    "business/market": "Business/market",
    "self-directed skill-building": "Self-directed skill-building",
    "personal/consumer": "Personal/consumer",
    "technical/implementation": "Technical/implementation",
    "generic/templated": "Generic/templated",
    "other": "Other",
}


def classify(turn1, turn2, retries=MAX_RETRIES):
    user_msg = f"Turn 1 (user): {turn1}\n"
    user_msg += f"Turn 2 (user): {turn2}\n" if turn2 else "Turn 2 (user): [no follow-up turn]\n"

    for attempt in range(retries):
        try:
            resp = requests.post(
                OLLAMA_URL,
                json={
                    "model": OLLAMA_MODEL,
                    "messages": [
                        {"role": "system", "content": TAXONOMY_RUBRIC},
                        {"role": "user", "content": user_msg},
                    ],
                    "stream": False,
                    "options": {"temperature": 0},
                },
                timeout=120,
            )
            resp.raise_for_status()
            content = resp.json()["message"]["content"].strip()
            reason_line = next((l for l in content.splitlines() if l.upper().startswith("REASON:")), "REASON: (none)")
            level1_line = next((l for l in content.splitlines() if l.upper().startswith("LEVEL1:")), "LEVEL1: LOOKUP")
            level2_line = next((l for l in content.splitlines() if l.upper().startswith("LEVEL2:")), "LEVEL2: Other")

            level1 = level1_line.split(":", 1)[1].strip().upper()
            if level1 not in VALID_LEVEL1:
                level1 = "LOOKUP"

            level2_raw = level2_line.split(":", 1)[1].strip().lower()
            level2 = LEVEL2_CANONICAL.get(level2_raw, "Other")

            reason = reason_line.split(":", 1)[1].strip() if ":" in reason_line else ""
            return level1, level2, reason
        except Exception as e:
            if attempt == retries - 1:
                return "ERROR", "ERROR", str(e)
            time.sleep(3)


def load_positives(path):
    with open(path) as f:
        return [r for r in csv.DictReader(f) if r["llm_label"] == "POSITIVE"]


def load_already_done(out_path):
    if not os.path.exists(out_path):
        return set()
    with open(out_path) as f:
        return {r["prompt"] for r in csv.DictReader(f)}


def run_source(source_name, in_path, out_path):
    positives = load_positives(in_path)
    done = load_already_done(out_path)
    todo = [r for r in positives if r["prompt"] not in done]

    print(f"[{source_name}] {len(positives)} positives, {len(done)} already categorized, "
          f"{len(todo)} remaining")

    write_header = not os.path.exists(out_path)
    with open(out_path, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["prompt", "turn2_preview", "level1", "level2", "taxonomy_reason"])
            f.flush()

        t0 = time.time()
        for i, row in enumerate(todo):
            level1, level2, reason = classify(row["prompt"], row.get("turn2_preview", ""))
            writer.writerow([row["prompt"], row.get("turn2_preview", ""), level1, level2, reason])
            f.flush()

            if (i + 1) % 25 == 0 or (i + 1) == len(todo):
                elapsed = time.time() - t0
                rate = (i + 1) / elapsed if elapsed > 0 else 0
                remaining = (len(todo) - i - 1) / rate if rate > 0 else float("inf")
                print(f"  [{source_name}] {i+1}/{len(todo)} done "
                      f"({rate:.2f} rows/s, ~{remaining/60:.0f} min remaining)")
                sys.stdout.flush()


if __name__ == "__main__":
    run_source("LMSYS-Chat-1M",
               os.path.join(OUTPUT_DIR, "lmsys_llm_classified.csv"),
               os.path.join(OUTPUT_DIR, "lmsys_taxonomy_v2.csv"))
    run_source("Chatbot Arena Conversations",
               os.path.join(OUTPUT_DIR, "arena_llm_classified.csv"),
               os.path.join(OUTPUT_DIR, "arena_taxonomy_v2.csv"))
    print("\nDone.")
