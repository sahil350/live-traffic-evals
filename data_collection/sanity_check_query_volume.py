"""
Sanity check: how many initiation/selection-stage (Kuhlthau ISP) research-planning
queries survive filtering from LMSYS-Chat-1M and Chatbot Arena Conversations?

Pipeline:
  1. Load the first exchange (up to 2 user turns + 2 assistant turns) from each source
  2. Embed everything locally (free, offline, sentence-transformers), in chunks of
     EMBED_CHUNK_SIZE (default 10,000), flushing a checkpoint CSV after each chunk --
     an interrupted run resumes by skipping rows already in the checkpoint instead
     of re-embedding from scratch (important at SAMPLE_N=200,000; not needed for
     small pilots but harmless there too)
  3. Semantic search against a small hand-curated set of exemplar queries
     (this is the real recall mechanism -- catches paraphrases regex would miss)
  4. Cluster the full embedding space (HDBSCAN) to find topic neighborhoods;
     keep clusters whose centroid sits near the exemplar anchors
     (this doubles as the non-redundancy check Step 1's Selection criteria need)
  5. Regex patterns are kept ONLY as a source for exemplar sentences, not as
     the filter itself -- regex has poor recall on paraphrase-heavy language
  6. For candidates only, score conversation TRAJECTORY (not just the isolated
     first message): does turn 2 narrow the topic, stay on-topic, or pivot
     away entirely? Does the assistant respond with a survey/list (the kind of
     answer an initiation-stage question elicits)? Kuhlthau's stages describe
     a *process*, so a single opening line is often ambiguous on its own --
     this is a cheap (no LLM) proxy for that missing trajectory signal, meant
     to sit alongside the embedding score, not replace it.

Run locally (requires HF access + a HF token approved for LMSYS-Chat-1M, which
is gated -- request access on the dataset page first).

    pip install datasets sentence-transformers hdbscan scikit-learn pandas numpy

Model choice: bge-large-en-v1.5 (strong for retrieval/semantic-search tasks,
runs fine on CPU for this volume, ~1.3GB download, fully offline after that).
Swap MODEL_NAME for all-mpnet-base-v2 if you want something smaller/faster.

Usage:
    python sanity_check_query_volume.py
"""

import csv
import json
import os
import re

import numpy as np
import pandas as pd
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
import hdbscan
from sklearn.metrics.pairwise import cosine_similarity

MODEL_NAME = "BAAI/bge-large-en-v1.5"
SIM_THRESHOLD = 0.55  # cosine similarity to nearest exemplar to count as a candidate
MIN_CLUSTER_SIZE = 15  # HDBSCAN param -- tune down if corpus sample is small
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "../outputs")
SAMPLE_N = int(os.environ.get("SAMPLE_N", 200_000))  # prompts pulled per source; lower for a quick pilot
EMBED_CHUNK_SIZE = int(os.environ.get("EMBED_CHUNK_SIZE", 10_000))  # rows per checkpoint flush
PIVOT_SIM_PERCENTILE = 20  # bottom 20% of observed turn1-vs-turn2 similarity reads as a topic pivot.
# A fixed absolute cosine cutoff isn't safe here -- bge-large-en-v1.5's baseline
# similarity floor between two unrelated English sentences sits well above 0
# (observed min ~0.27, median ~0.62 on the pilot sample), so a guessed constant
# like 0.30 barely ever fires. Percentile-based instead: self-calibrating to
# whatever the actual distribution looks like, and to the embedding model in use.

# ---------------------------------------------------------------------------
# 1. Load the first exchange (up to 2 user turns + 2 assistant turns) from
#    each source -- we need turn 2+ for the trajectory heuristic below, not
#    just the opening line.
# ---------------------------------------------------------------------------

def _extract_exchange(conv, max_user_turns=2):
    user_turns = [m["content"] for m in conv if m["role"] == "user"][:max_user_turns]
    assistant_turns = [m["content"] for m in conv if m["role"] == "assistant"][:max_user_turns]
    return user_turns, assistant_turns


def load_lmsys_first_turns(n=200_000):
    ds = load_dataset("lmsys/lmsys-chat-1m", split="train")
    ds = ds.filter(lambda x: x["language"] == "English")
    records = []
    for row in ds.select(range(min(n, len(ds)))):
        conv = row["conversation"]
        if conv and conv[0]["role"] == "user":
            user_turns, assistant_turns = _extract_exchange(conv)
            records.append({"prompt": user_turns[0], "user_turns": user_turns,
                             "assistant_turns": assistant_turns})
    return records


def load_chatbot_arena_conversations(n=200_000):
    ds = load_dataset("lmsys/chatbot_arena_conversations", split="train")
    records = []
    for row in ds.select(range(min(n, len(ds)))):
        conv = row["conversation_a"]
        if conv and conv[0]["role"] == "user":
            user_turns, assistant_turns = _extract_exchange(conv)
            records.append({"prompt": user_turns[0], "user_turns": user_turns,
                             "assistant_turns": assistant_turns})
    return records


# ---------------------------------------------------------------------------
# 2. Exemplar seed set -- Kuhlthau initiation/selection-stage queries.
#    Hand-written to match the definition, not mined from regex hits, so they
#    stay a clean, deliberate anchor rather than inheriting regex's biases.
#    Expand this list with real regex-caught examples once you have some,
#    but keep the two sources distinguishable in your methodology notes.
# ---------------------------------------------------------------------------

INITIATION_EXEMPLARS = [
    "Help me understand the current landscape of research on X before I dive in.",
    "What are the main approaches or schools of thought in this field, and how do they differ?",
    "I'm trying to get oriented on a new topic -- where should I start?",
    "What are the open questions or debates that researchers are still working on in this area?",
    "How should I structure my approach to researching this topic?",
    "What should I read to get a solid overview before narrowing down my focus?",
    "Can you help me scope out a research project on this subject?",
    "What are the key subtopics I should be aware of if I want to explore this area?",
    "I'm not sure yet what angle to take on this topic -- can you help me think it through?",
    "Give me a broad survey of what's known and unknown about this subject.",
]

# Confirmed true positives mined from the 2026-08-29 LMSYS pilot (SAMPLE_N=5000),
# manually reviewed against the definition above. Kept as a visibly separate
# block from the hand-written set (not merged silently) per the methodology
# note above -- these carry LMSYS's own phrasing quirks, the hand-written ones
# don't.
CONFIRMED_TRUE_POSITIVE_EXEMPLARS = []  # real prompts mined from the pilot are omitted from the public release

INITIATION_EXEMPLARS = INITIATION_EXEMPLARS + CONFIRMED_TRUE_POSITIVE_EXEMPLARS

# Formulation/collection-stage contrast set -- narrow, fact-seeking. Used only
# to sanity-check that the two neighborhoods actually separate in embedding
# space (report this separation as validation of the exemplar set).
FORMULATION_EXEMPLARS = [
    "What was the exact date this event occurred?",
    "How many people were involved in this specific incident?",
    "Who was the president of this country in that year?",
    "What is the precise value of this constant?",
    "When did this specific policy take effect?",
]

# ---------------------------------------------------------------------------
# Trajectory heuristic (no LLM) -- a cheap proxy for the fact that Kuhlthau's
# stages describe a *process*, not a single isolated line. Applied only to
# candidates that already passed the embedding threshold, so it's layered on
# top of semantic search, not a replacement for it.
# ---------------------------------------------------------------------------

NARROWING_PATTERNS = re.compile(
    r"\b(more specifically|in particular|narrow(ing)? (it |this )?down|"
    r"to narrow (it |this )?down|for example|such as|what about|"
    r"one (aspect|area|part)|zoom in|dive deeper|deep dive|focus on|"
    r"drill down|specifically,?\s+(i|I))\b"
)

# A numbered or bulleted assistant reply looks like a survey/overview answer
# -- the kind of response an initiation-stage question typically elicits.
ASSISTANT_LIST_PATTERN = re.compile(r"(\n\s*\d+[\.\)]\s)|(\n\s*[-*]\s)")


def compute_trajectory_features(candidate_records, cand_prompt_emb, model, source_name=""):
    n = len(candidate_records)
    turn_count = np.zeros(n, dtype=int)
    narrowing_hit = np.zeros(n, dtype=bool)
    assistant_list_response = np.zeros(n, dtype=bool)
    topic_pivot = np.zeros(n, dtype=bool)
    sim_turn1_turn2 = np.full(n, np.nan)
    turn2_preview = [""] * n

    followup_idx, followup_texts = [], []
    for i, rec in enumerate(candidate_records):
        user_turns = rec["user_turns"]
        assistant_turns = rec["assistant_turns"]
        turn_count[i] = len(user_turns)
        if assistant_turns and ASSISTANT_LIST_PATTERN.search(assistant_turns[0] or ""):
            assistant_list_response[i] = True
        if len(user_turns) >= 2:
            followup_idx.append(i)
            followup_texts.append(user_turns[1])
            turn2_preview[i] = user_turns[1][:150]
            if NARROWING_PATTERNS.search(user_turns[1]):
                narrowing_hit[i] = True

    if followup_texts:
        followup_emb = model.encode(followup_texts, normalize_embeddings=True, show_progress_bar=False)
        sims = [float(np.dot(cand_prompt_emb[i], followup_emb[j])) for j, i in enumerate(followup_idx)]
        pivot_cutoff = float(np.percentile(sims, PIVOT_SIM_PERCENTILE)) if len(sims) >= 5 else -1.0
        print(f"  [{source_name}] pivot cutoff (bottom {PIVOT_SIM_PERCENTILE}% of "
              f"{len(sims)} turn1-turn2 similarities): {pivot_cutoff:.3f}")
        for j, i in enumerate(followup_idx):
            sim_turn1_turn2[i] = sims[j]
            if sims[j] < pivot_cutoff:
                topic_pivot[i] = True

    has_followup = turn_count >= 2
    trajectory_score = (
        has_followup.astype(int)
        + narrowing_hit.astype(int)
        + assistant_list_response.astype(int)
        - topic_pivot.astype(int)
    )

    return pd.DataFrame({
        "turn_count": turn_count,
        "sim_turn1_turn2": sim_turn1_turn2,
        "narrowing_hit": narrowing_hit,
        "assistant_list_response": assistant_list_response,
        "topic_pivot": topic_pivot,
        "trajectory_score": trajectory_score,
        "turn2_preview": turn2_preview,
    })


# ---------------------------------------------------------------------------
# 3. Embed + semantic search (chunked, resumable) + cluster
# ---------------------------------------------------------------------------

def embed_and_filter_chunked(records, init_emb, model, source_name, checkpoint_path):
    """Embed prompts in chunks, flushing to checkpoint_path after each chunk so
    an interrupted run can resume by skipping already-embedded rows instead of
    re-embedding from scratch. Only candidate rows get their turns persisted
    (as JSON) -- non-candidates never need them again, keeps the checkpoint
    file smaller for a 200k-row run.
    """
    already_done = 0
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path) as f:
            already_done = sum(1 for _ in csv.DictReader(f))
        if already_done > 0:
            print(f"  [{source_name}] resuming from checkpoint: {already_done} prompts "
                  f"already embedded, skipping to row {already_done}")

    write_header = not os.path.exists(checkpoint_path)
    total = len(records)
    with open(checkpoint_path, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["prompt", "max_sim_to_initiation_exemplar", "is_candidate",
                              "user_turns_json", "assistant_turns_json"])
            f.flush()

        for start in range(already_done, total, EMBED_CHUNK_SIZE):
            chunk = records[start:start + EMBED_CHUNK_SIZE]
            chunk_prompts = [r["prompt"] for r in chunk]
            chunk_emb = model.encode(chunk_prompts, batch_size=256, show_progress_bar=False,
                                      normalize_embeddings=True)
            sims = cosine_similarity(chunk_emb, init_emb).max(axis=1)
            candidate_mask = sims >= SIM_THRESHOLD

            for i, r in enumerate(chunk):
                is_cand = bool(candidate_mask[i])
                turns_json = json.dumps(r["user_turns"]) if is_cand else ""
                assistant_json = json.dumps(r["assistant_turns"]) if is_cand else ""
                writer.writerow([r["prompt"], float(sims[i]), is_cand, turns_json, assistant_json])
            f.flush()

            done = min(start + EMBED_CHUNK_SIZE, total)
            print(f"  [{source_name}] embedded {done}/{total} ({100 * done / total:.1f}%)")


def load_checkpoint(checkpoint_path):
    with open(checkpoint_path) as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["is_candidate"] = r["is_candidate"] == "True"
        r["max_sim_to_initiation_exemplar"] = float(r["max_sim_to_initiation_exemplar"])
        if r["is_candidate"]:
            r["user_turns"] = json.loads(r["user_turns_json"])
            r["assistant_turns"] = json.loads(r["assistant_turns_json"])
    return rows


def run_pipeline(records, source_name, model, checkpoint_path):
    records = [r for r in records if isinstance(r["prompt"], str) and 4 <= len(r["prompt"].split()) <= 200]
    print(f"  [{source_name}] {len(records)} prompts to embed (chunk size {EMBED_CHUNK_SIZE}, "
          f"checkpoint: {checkpoint_path})")

    init_emb = model.encode(INITIATION_EXEMPLARS, normalize_embeddings=True)
    form_emb = model.encode(FORMULATION_EXEMPLARS, normalize_embeddings=True)

    # Sanity check: exemplar sets should NOT be close to each other.
    cross_sim = cosine_similarity(init_emb, form_emb).mean()
    print(f"  [{source_name}] mean similarity between initiation and formulation "
          f"exemplars: {cross_sim:.3f} (want this low, e.g. <0.4 -- if it's high, "
          f"the exemplar sets aren't well separated and results below are less trustworthy)")

    embed_and_filter_chunked(records, init_emb, model, source_name, checkpoint_path)
    checkpoint_rows = load_checkpoint(checkpoint_path)

    prompts = [r["prompt"] for r in checkpoint_rows]
    sims_to_init = np.array([r["max_sim_to_initiation_exemplar"] for r in checkpoint_rows])
    candidate_mask = np.array([r["is_candidate"] for r in checkpoint_rows])

    df = pd.DataFrame({
        "prompt": prompts,
        "max_sim_to_initiation_exemplar": sims_to_init,
        "is_candidate": candidate_mask,
    })
    n_candidates = int(candidate_mask.sum())
    print(f"  [{source_name}] {n_candidates} candidates above similarity threshold "
          f"{SIM_THRESHOLD} ({100 * n_candidates / len(prompts):.2f}% of sample)")

    # Trajectory heuristic, candidates only -- see module docstring step 6.
    # Re-embed just the candidate subset (small, e.g. 20-40% of the sample) --
    # the checkpoint only persisted similarity scores, not full embedding
    # vectors, to keep it lean for a 200k-row run.
    if n_candidates > 0:
        candidate_records = [r for r in checkpoint_rows if r["is_candidate"]]
        cand_prompt_emb = model.encode([r["prompt"] for r in candidate_records],
                                        batch_size=256, show_progress_bar=False,
                                        normalize_embeddings=True)
        traj_df = compute_trajectory_features(candidate_records, cand_prompt_emb, model, source_name)
        for col in traj_df.columns:
            df.loc[candidate_mask, col] = traj_df[col].values

        score_counts = traj_df["trajectory_score"].value_counts().sort_index()
        print(f"  [{source_name}] trajectory_score distribution among candidates "
              f"(-1=pivot .. 3=strong): {dict(score_counts)}")
        n_strong = int((traj_df["trajectory_score"] >= 2).sum())
        print(f"  [{source_name}] {n_strong} candidates with trajectory_score >= 2 "
              f"(embedding match + follow-up behavior consistent with initiation/selection)")

    # Cluster the candidate subset for non-redundancy accounting.
    # (Clustering candidates only, not the full corpus, keeps this fast --
    # switch to clustering prompt_emb in full if you want cluster labels for
    # everything, e.g. to look for adjacent categories you didn't anchor on.)
    if n_candidates >= MIN_CLUSTER_SIZE:
        clusterer = hdbscan.HDBSCAN(min_cluster_size=MIN_CLUSTER_SIZE, metric="euclidean")
        labels = clusterer.fit_predict(cand_prompt_emb)
        df.loc[candidate_mask, "cluster"] = labels
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = int((labels == -1).sum())
        print(f"  [{source_name}] {n_clusters} distinct non-redundant clusters found "
              f"among candidates ({n_noise} unclustered/noise points)")
    else:
        print(f"  [{source_name}] too few candidates ({n_candidates}) to cluster "
              f"meaningfully -- treat as a small-N pilot signal, not a validated count")

    return df


# ---------------------------------------------------------------------------
# 4. Run
# ---------------------------------------------------------------------------

# Which source(s) to run: "lmsys", "arena", or "both" (default). Run them as
# separate `docker run` invocations (SOURCE=lmsys, then SOURCE=arena) when
# memory is tight -- each process fully exits and releases its memory before
# the next starts, instead of accumulating both sources' embeddings in one
# long-lived process.
SOURCE = os.environ.get("SOURCE", "both").lower()

if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Loading embedding model: {MODEL_NAME} (downloads once, then fully offline)")
    model = SentenceTransformer(MODEL_NAME)

    if SOURCE in ("lmsys", "both"):
        print(f"\nLoading LMSYS-Chat-1M (English, first {SAMPLE_N})...")
        lmsys_records = load_lmsys_first_turns(n=SAMPLE_N)
        lmsys_checkpoint = os.path.join(OUTPUT_DIR, "lmsys_embed_checkpoint.csv")
        lmsys_df = run_pipeline(lmsys_records, "LMSYS-Chat-1M", model, lmsys_checkpoint)
        lmsys_df.to_csv(os.path.join(OUTPUT_DIR, "lmsys_initiation_candidates.csv"), index=False)

    if SOURCE in ("arena", "both"):
        print(f"\nLoading Chatbot Arena Conversations (first {SAMPLE_N})...")
        arena_records = load_chatbot_arena_conversations(n=SAMPLE_N)
        arena_checkpoint = os.path.join(OUTPUT_DIR, "arena_embed_checkpoint.csv")
        arena_df = run_pipeline(arena_records, "Chatbot Arena Conversations", model, arena_checkpoint)
        arena_df.to_csv(os.path.join(OUTPUT_DIR, "arena_initiation_candidates.csv"), index=False)

    print("\nNOTE: this gives candidate volume + non-redundant cluster count + a "
          "no-LLM trajectory_score (-1 pivot .. 3 strong), which is the signal worth "
          "reporting -- not raw hit count (see funnel discussion). trajectory_score >= 2 "
          "is the best free proxy for precision we have right now; if that's not clean "
          "enough on manual spot-check, next step is LLM-classifying candidates against "
          "the Kuhlthau initiation/selection definition (using turn 1 + turn 2, not just "
          "turn 1) before any broader SME review.")
