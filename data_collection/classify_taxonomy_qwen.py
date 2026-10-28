"""
Second pass: categorize the confirmed positives from classify_candidates_qwen.py
into an intent taxonomy, instead of continuing to patch a binary rubric with
more exclusion bullets. False positives found so far (personal/consumer
decisions, technical/implementation decisions, project-planning, templated
meta-prompts) aren't noise to exclude -- they're categories we hadn't named.
Naming them lets the model pick the best-fitting label instead of checking
against a growing NOT-list, and gives an actual count per category instead of
a single binary precision estimate.

Runs only over rows with llm_label == 'POSITIVE' in the Stage-2 output CSVs
(2,144 rows total, not the full 72K sample) -- cheap, ~50 min at the observed
~0.7 rows/sec rate.

Usage:
    python classify_taxonomy_qwen.py
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

TAXONOMY_RUBRIC = """You are categorizing a two-turn snippet of a real chat conversation into an \
intent taxonomy for a research study on how people orient themselves to topics before diving in.

Assign exactly ONE Level 1 category:

A. Research/investigative orientation -- the user is scoping an academic, scientific, or \
investigative question: a literature survey, thesis/research proposal scoping, or broad \
landscape question about a field of study.
B. Project/product planning -- the user is scoping a business, app, product, or creative \
project (not academic/scientific research). E.g. "structure for my app's MVP", "governance \
model for a company", "business strategy for X".
C. Personal/consumer decision-making -- shopping, travel planning, personal life decisions, \
dressed in orientation-sounding language. E.g. "help me decide on a dress", "plan my trip \
itinerary".
D. Technical/implementation decision -- choosing between technical approaches, tools, \
libraries, or architectures for a concrete build. E.g. "should I use X or Y language/library", \
"module system or pattern matching first".
E. Generic/templated request -- meta-prompts, copy-pasted community prompt templates ("act as \
X", "become my Y"), or ultra-short generic asks with no real framing ("tell me about X", "what \
is the future of X").
F. Other/unclear -- doesn't fit any category above, or too ambiguous to place confidently.

If (and only if) you assign category A, ALSO assign a Level 2 sub-category:
A1. Academic/scientific research -- thesis, paper, literature review, formal research project
A2. Business/market research -- market landscape, industry analysis, competitive research
A3. Self-directed learning research -- "how do I learn field X", "roadmap to learn ML", \
orienting to a field to build personal expertise (research-oriented, not task-execution)
A4. General exploratory/curiosity-driven topic orientation -- broad "what's known about X" \
survey requests without a stated formal project

Examples:

[Worked examples are omitted from this public release. They were excerpts of real conversations from a gated dataset (LMSYS-Chat-1M and Chatbot Arena), which cannot be redistributed. Write your own, in the same shape as the output format below, before running. Without them the labels will differ from those reported in the paper.]

Respond in EXACTLY this format, nothing else:
REASON: one sentence
LEVEL1: A, B, C, D, E, or F
LEVEL2: A1, A2, A3, A4, or NONE (NONE if LEVEL1 is not A)
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
            level1_line = next((l for l in content.splitlines() if l.upper().startswith("LEVEL1:")), "LEVEL1: F")
            level2_line = next((l for l in content.splitlines() if l.upper().startswith("LEVEL2:")), "LEVEL2: NONE")
            level1 = level1_line.split(":", 1)[1].strip().upper()[:1]
            if level1 not in ("A", "B", "C", "D", "E", "F"):
                level1 = "F"
            level2_raw = level2_line.split(":", 1)[1].strip().upper()
            level2 = level2_raw if level2_raw in ("A1", "A2", "A3", "A4") else "NONE"
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
               os.path.join(OUTPUT_DIR, "lmsys_taxonomy.csv"))
    run_source("Chatbot Arena Conversations",
               os.path.join(OUTPUT_DIR, "arena_llm_classified.csv"),
               os.path.join(OUTPUT_DIR, "arena_taxonomy.csv"))
    print("\nDone.")
