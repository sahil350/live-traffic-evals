"""
Follow-up to contamination_check.py: 3 of the original external sources
(web-bench's xbench, browsecomp, browsecomp_zh splits -- 1,655 rows) were
found to be encrypted with a scheme that couldn't be reversed with the
canary/method web-bench's own README documents. Correct plaintext was
recovered from each benchmark's true canonical source instead (see
METHODOLOGY_REPORT.md Section 16 for the full story). This script re-embeds
only those 1,655 corrected texts and recomputes the max-similarity result
per working-set record, taking the max against BOTH the original (already
verified correct) results and these new ones.
"""

import csv
import json

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

MODEL_NAME = "BAAI/bge-large-en-v1.5"

if __name__ == "__main__":
    with open("outputs/working_set_151sample.csv") as f:
        working = list(csv.DictReader(f))
    with open("outputs/corrected_previously_garbled.json") as f:
        corrected = json.load(f)
    with open("outputs/contamination_check_results.csv") as f:
        prior_results = {r["case_id"]: r for r in csv.DictReader(f)}

    print(f"Working set: {len(working)} rows")
    print(f"Corrected texts to re-embed: {len(corrected)}")

    model = SentenceTransformer(MODEL_NAME)

    our_texts = [r["prompt"] for r in working]
    our_emb = model.encode(our_texts, batch_size=32, show_progress_bar=True, normalize_embeddings=True)

    corr_texts = [c["question"] for c in corrected]
    corr_emb = model.encode(corr_texts, batch_size=256, show_progress_bar=True, normalize_embeddings=True)

    sims = cosine_similarity(our_emb, corr_emb)

    final_results = []
    for i, r in enumerate(working):
        best_idx = int(np.argmax(sims[i]))
        new_best_sim = float(sims[i][best_idx])
        new_best_source = corrected[best_idx]["source"]
        new_best_question = corrected[best_idx]["question"]
        new_best_answer = corrected[best_idx].get("answer", "")

        prior = prior_results[r["case_id"]]
        prior_sim = float(prior["max_similarity"])

        if new_best_sim > prior_sim:
            final_results.append({
                "case_id": r["case_id"], "our_prompt": r["prompt"],
                "best_match_source": new_best_source, "best_match_question": new_best_question,
                "best_match_answer": new_best_answer, "max_similarity": new_best_sim,
            })
        else:
            final_results.append({
                "case_id": r["case_id"], "our_prompt": r["prompt"],
                "best_match_source": prior["best_match_source"],
                "best_match_question": prior["best_match_question"],
                "best_match_answer": prior["best_match_answer"],
                "max_similarity": prior_sim,
            })

    final_results.sort(key=lambda x: -x["max_similarity"])

    with open("outputs/contamination_check_results_corrected.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["case_id", "our_prompt", "best_match_source",
                                                "best_match_question", "best_match_answer",
                                                "max_similarity"])
        writer.writeheader()
        writer.writerows(final_results)

    print("\nTop 10 highest-similarity matches (corrected, full 11,639-text corpus):")
    for r in final_results[:10]:
        print(f"  sim={r['max_similarity']:.3f} [{r['case_id']}] src={r['best_match_source']}")
        print(f"    ours: {r['our_prompt'][:110]}")
        print(f"    ext:  {r['best_match_question'][:110]}")

    print(f"\nWrote outputs/contamination_check_results_corrected.csv ({len(final_results)} rows)")
