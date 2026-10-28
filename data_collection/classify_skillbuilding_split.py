"""
Third pass: sub-splits the "Self-directed skill-building" Level 2 domain from
classify_taxonomy_v2_marchionini.py into two kinds, since it currently lumps
together things that shouldn't count the same way toward "research-planning
intent":

  - KNOWLEDGE_FIELD: orienting to a body of knowledge/field a research agent
    would meaningfully help scope (ML, AI, physics, data science, a new
    academic/technical discipline).
  - PRACTICAL_SKILL: acquiring a practical/personal skill (a language, an
    instrument, cooking, fitness) -- closer to the "teach me Spanish"
    exclusion already established in the Stage-2 binary rubric, not research
    orientation.

Runs only over rows with level2 == 'Self-directed skill-building' in the v2
taxonomy output (both LEARN and INVESTIGATE level1, though INVESTIGATE rows
in this domain are rare).

Usage:
    python classify_skillbuilding_split.py
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

RUBRIC = """You are sub-categorizing a chat conversation already identified as \
"self-directed skill-building" -- the user wants to build their own capability in something. \
Decide which of two kinds it is:

KNOWLEDGE_FIELD -- orienting to a body of knowledge or technical/academic field: machine \
learning, AI, data science, a scientific discipline, a technical domain like software \
architecture or a programming paradigm. The kind of thing a research agent would meaningfully \
help someone scope out and survey.

PRACTICAL_SKILL -- acquiring a practical or personal skill: a spoken language, a musical \
instrument, cooking, a sport, a craft, a fitness routine, an everyday life skill. This is \
capability-building through practice, not orientation to a knowledge field.

Examples:

[Worked examples are omitted from this public release. They were excerpts of real conversations from a gated dataset (LMSYS-Chat-1M and Chatbot Arena), which cannot be redistributed. Write your own, in the same shape as the output format below, before running. Without them the labels will differ from those reported in the paper.]

Respond in EXACTLY this format, nothing else:
REASON: one sentence
LABEL: KNOWLEDGE_FIELD or PRACTICAL_SKILL
"""


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
            reason_line = next((l for l in content.splitlines() if l.upper().startswith("REASON:")), "REASON: (none)")
            label_line = next((l for l in content.splitlines() if l.upper().startswith("LABEL:")), "LABEL: PRACTICAL_SKILL")
            label = "KNOWLEDGE_FIELD" if "KNOWLEDGE_FIELD" in label_line.upper() else "PRACTICAL_SKILL"
            reason = reason_line.split(":", 1)[1].strip() if ":" in reason_line else ""
            return label, reason
        except Exception as e:
            if attempt == retries - 1:
                return "ERROR", str(e)
            time.sleep(3)


def load_skillbuilding(path):
    with open(path) as f:
        return [r for r in csv.DictReader(f) if r["level2"] == "Self-directed skill-building"]


def load_already_done(out_path):
    if not os.path.exists(out_path):
        return set()
    with open(out_path) as f:
        return {r["prompt"] for r in csv.DictReader(f)}


def run_source(source_name, in_path, out_path):
    rows = load_skillbuilding(in_path)
    done = load_already_done(out_path)
    todo = [r for r in rows if r["prompt"] not in done]

    print(f"[{source_name}] {len(rows)} skill-building rows, {len(done)} already split, "
          f"{len(todo)} remaining")

    write_header = not os.path.exists(out_path)
    with open(out_path, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["prompt", "turn2_preview", "level1", "skill_kind", "skill_reason"])
            f.flush()

        t0 = time.time()
        for i, row in enumerate(todo):
            kind, reason = classify(row["prompt"], row.get("turn2_preview", ""))
            writer.writerow([row["prompt"], row.get("turn2_preview", ""), row["level1"], kind, reason])
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
               os.path.join(OUTPUT_DIR, "lmsys_taxonomy_v2.csv"),
               os.path.join(OUTPUT_DIR, "lmsys_skillbuilding_split.csv"))
    run_source("Chatbot Arena Conversations",
               os.path.join(OUTPUT_DIR, "arena_taxonomy_v2.csv"),
               os.path.join(OUTPUT_DIR, "arena_skillbuilding_split.csv"))
    print("\nDone.")
