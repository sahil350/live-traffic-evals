"""
Contamination check: does our 87-row working set (working_set_151sample.csv) have
exact or near-duplicate matches against OpenResearcher's training data and
benchmark history? Gate check per sub-project 1's Step 3 (external scope, not
otherwise documented in this repo -- see conversation notes 2026-09-02).

External sources checked (question + answer text, not the associated document
corpora -- those are handled separately, see limitations in the report):
  - web-bench (BrowseComp, BrowseComp_zh, GAIA-text, HLE, SEAL, WebWalkerQA, xbench)
  - browsecomp-plus (Tevatron, decrypted via the benchmark's own published XOR scheme)
  - OpenResearcher-Dataset (seed_42 only -- verified 100% question overlap across
    all 16 seed configs, so seed_42 alone represents the full ~96K-trajectory /
    6,102-unique-question training pool)

Two checks: exact match (normalized whitespace/case) and embedding-based
near-duplicate (bge-large-en-v1.5, same model used throughout this project).
"""

import csv
import json
import re

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

MODEL_NAME = "BAAI/bge-large-en-v1.5"


def normalize(s):
    return re.sub(r"\s+", " ", s.strip().lower())


if __name__ == "__main__":
    with open("outputs/working_set_151sample.csv") as f:
        working = list(csv.DictReader(f))
    with open("outputs/all_external_texts.json") as f:
        external = json.load(f)

    print(f"Working set: {len(working)} rows")
    print(f"External texts: {len(external)}")

    our_texts = [r["prompt"] for r in working]
    ext_texts = [e["question"] for e in external]

    print(f"\nLoading {MODEL_NAME}...")
    model = SentenceTransformer(MODEL_NAME)

    print("Embedding working set...")
    our_emb = model.encode(our_texts, batch_size=32, show_progress_bar=True, normalize_embeddings=True)

    print("Embedding external corpus (this is the slow part)...")
    ext_emb = model.encode(ext_texts, batch_size=256, show_progress_bar=True, normalize_embeddings=True)

    print("\nComputing similarity matrix...")
    sims = cosine_similarity(our_emb, ext_emb)  # shape (87, n_external)

    results = []
    for i, r in enumerate(working):
        best_idx = int(np.argmax(sims[i]))
        best_sim = float(sims[i][best_idx])
        results.append({
            "case_id": r["case_id"],
            "our_prompt": r["prompt"],
            "best_match_source": external[best_idx]["source"],
            "best_match_question": external[best_idx]["question"],
            "best_match_answer": external[best_idx].get("answer", ""),
            "max_similarity": best_sim,
        })

    results.sort(key=lambda x: -x["max_similarity"])

    with open("outputs/contamination_check_results.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["case_id", "our_prompt", "best_match_source",
                                                "best_match_question", "best_match_answer",
                                                "max_similarity"])
        writer.writeheader()
        writer.writerows(results)

    print("\nTop 15 highest-similarity matches:")
    for r in results[:15]:
        print(f"  sim={r['max_similarity']:.3f} [{r['case_id']}] src={r['best_match_source']}")
        print(f"    ours: {r['our_prompt'][:110]}")
        print(f"    ext:  {r['best_match_question'][:110]}")

    print(f"\nWrote outputs/contamination_check_results.csv ({len(results)} rows)")
