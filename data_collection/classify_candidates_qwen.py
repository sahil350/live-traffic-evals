"""
Classify Stage-1 embedding candidates with a local Qwen2.5 7B model (via Ollama)
against the Kuhlthau initiation/selection-stage rubric. Runs against
lmsys_initiation_candidates.csv and arena_initiation_candidates.csv (is_candidate
rows only) produced by sanity_check_query_volume.py.

Rubric validated on 14 hand-labeled examples from manual review: 100% on
unambiguous cases (4/4 clear positive, 5/5 clear negative including a pivot
case), 1/4 on genuinely ambiguous boundary cases -- those are accepted as
legitimate judgment calls, not classifier errors (see conversation notes).

Writes incrementally (one row at a time, flushed) and is resumable -- rerunning
skips prompts already present in the output CSV, so an interrupted ~3hr run
can pick back up without restarting from zero.

Usage:
    python classify_candidates_qwen.py
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

RUBRIC = """You are classifying a two-turn snippet of a real chat conversation for a research \
study. Decide whether the CONVERSATION shows evidence of Kuhlthau ISP "initiation/selection-stage \
research-planning" intent -- i.e., whether the user was, at some point in this exchange, orienting \
themselves to a topic before narrowing down: scoping a research project, thesis, or investigation; \
asking for a broad survey/landscape of a field; asking how to structure or approach researching \
something; explicitly planning to conduct or produce research or investigative work.

You are judging overall conversational intent, not selecting which turn to extract for later use.

POSITIVE if the conversation shows genuine orientation/scoping intent at any point -- including \
cases where turn 1 shows real orientation and turn 2 pivots to something else (e.g. turn 1 scopes \
a thesis, turn 2 asks to draft the abstract). The orientation evidence is what matters, even if \
the conversation moves on from it.

NEGATIVE if none of the turns show orientation/scoping intent, including:
- A single fact or concept lookup ("what is X", "explain X to me", "explain X better") with no \
broader research framing
- Creative writing, roleplay, or persona-based tasks
- Coding/technical help requests
- Games (e.g. 20 questions, guessing games)
- Personal advice unrelated to research
- Debate/opinion questions where the user just wants the model's take, with no stated intent to \
go do research and no exploration of the question's argument space
- Self-directed skill learning (e.g. "teach me Spanish") that is not academic/investigative research
- A bare request to produce a deliverable (abstract, report section, market analysis, proposal \
draft, SEO/ad copy, press release, a full article on a topic, a how-to/build guide) with no \
preceding orientation shown -- even if the topic sounds academic or research-flavored, producing \
a document is not the same as scoping one. This includes "please provide/give me a detailed guide \
on how to build/develop X" -- that's a request for instructions or a coding/technical build, not \
orientation to a research topic, even when X sounds like an ambitious research-y system.
- A bare reading-list request ("send me 5 papers on X") with no orientation framing. If paired \
with orientation language ("help me get started," "what should I read to understand X") or framed \
as a survey of a field, count it POSITIVE instead.

Some cases are genuinely ambiguous (e.g. a debate framed around a real research question, a broad \
concept explanation that borders on a field survey). Reason through these on their merits. A \
debate/opinion question only counts as POSITIVE when the user explicitly states research, \
writing, or decision-making intent (e.g. "I'm writing a paper on X," "for a debate class \
assignment," "trying to decide for my thesis") -- don't infer that intent from the topic sounding \
substantive or high-stakes on its own. Without that explicit framing, treat it as a bare opinion \
request. If you are still genuinely torn after reasoning it through, default to NEGATIVE.

Examples:

[Worked examples are omitted from this public release. They were excerpts of real conversations from a gated dataset (LMSYS-Chat-1M and Chatbot Arena), which cannot be redistributed. Write your own, in the same shape as the output format below, before running. Without them the labels will differ from those reported in the paper.]

Respond in EXACTLY this format, nothing else:
REASON: one sentence
LABEL: POSITIVE or NEGATIVE
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
            label_line = next((l for l in content.splitlines() if l.upper().startswith("LABEL:")), "LABEL: PARSE_ERROR")
            reason_line = next((l for l in content.splitlines() if l.upper().startswith("REASON:")), "REASON: (none)")
            label = "POSITIVE" if "POSITIVE" in label_line.upper() else "NEGATIVE"
            reason = reason_line.split(":", 1)[1].strip() if ":" in reason_line else ""
            return label, reason
        except Exception as e:
            if attempt == retries - 1:
                return "ERROR", str(e)
            time.sleep(3)


def load_candidates(path):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    return [r for r in rows if r.get("is_candidate") == "True"]


def load_already_done(out_path):
    if not os.path.exists(out_path):
        return set()
    with open(out_path) as f:
        return {r["prompt"] for r in csv.DictReader(f)}


def run_source(source_name, in_path, out_path):
    candidates = load_candidates(in_path)
    done = load_already_done(out_path)
    todo = [r for r in candidates if r["prompt"] not in done]

    print(f"[{source_name}] {len(candidates)} candidates, {len(done)} already classified, "
          f"{len(todo)} remaining")

    write_header = not os.path.exists(out_path)
    with open(out_path, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["prompt", "turn2_preview", "max_sim_to_initiation_exemplar",
                              "trajectory_score", "llm_label", "llm_reason"])
            f.flush()

        t0 = time.time()
        for i, row in enumerate(todo):
            label, reason = classify(row["prompt"], row.get("turn2_preview", ""))
            writer.writerow([row["prompt"], row.get("turn2_preview", ""),
                              row.get("max_sim_to_initiation_exemplar", ""),
                              row.get("trajectory_score", ""), label, reason])
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
               os.path.join(OUTPUT_DIR, "lmsys_initiation_candidates.csv"),
               os.path.join(OUTPUT_DIR, "lmsys_llm_classified.csv"))
    run_source("Chatbot Arena Conversations",
               os.path.join(OUTPUT_DIR, "arena_initiation_candidates.csv"),
               os.path.join(OUTPUT_DIR, "arena_llm_classified.csv"))
    print("\nDone.")
