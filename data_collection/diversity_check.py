"""
Internal diversity check on the 84-record working set itself -- not a
contamination check against external data (that's Section 16), this asks
whether the working set collapses into a few near-duplicate/templated shapes
or is genuinely spread out. Same embedding model as every other similarity
check in this project (bge-large-en-v1.5), same HDBSCAN clustering approach
as topic_diversity.py, applied to the 84 instead of the 329 pilot positives.

Outputs:
  - outputs/diversity_similarity_matrix.csv (84x84 cosine similarity, case_ids as row/col labels)
  - outputs/diversity_top_pairs.csv (top 20 most-similar off-diagonal pairs)
  - printed summary stats + cluster breakdown

Usage:
    python diversity_check.py
"""

import csv
import os

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import hdbscan

OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "../outputs")
MODEL_NAME = "BAAI/bge-large-en-v1.5"
MIN_CLUSTER_SIZE = int(os.environ.get("DIVERSITY_MIN_CLUSTER_SIZE", 3))

if __name__ == "__main__":
    with open(os.path.join(OUTPUT_DIR, "working_set_151sample.csv")) as f:
        rows = list(csv.DictReader(f))
    print(f"Working set: {len(rows)} rows")

    case_ids = [r["case_id"] for r in rows]
    prompts = [r["prompt"] for r in rows]

    print(f"Loading {MODEL_NAME}...")
    model = SentenceTransformer(MODEL_NAME)
    emb = model.encode(prompts, batch_size=32, show_progress_bar=True, normalize_embeddings=True)

    sims = cosine_similarity(emb)
    n = len(rows)

    # Write full matrix
    with open(os.path.join(OUTPUT_DIR, "diversity_similarity_matrix.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([""] + case_ids)
        for i, cid in enumerate(case_ids):
            writer.writerow([cid] + [round(float(x), 4) for x in sims[i]])

    # Off-diagonal stats
    mask = ~np.eye(n, dtype=bool)
    off_diag = sims[mask]
    print(f"\nOff-diagonal pairwise similarity (n={len(off_diag)} pairs):")
    print(f"  mean={off_diag.mean():.4f}  median={np.median(off_diag):.4f}  "
          f"min={off_diag.min():.4f}  max={off_diag.max():.4f}  std={off_diag.std():.4f}")

    # Top 20 most-similar off-diagonal pairs (each unordered pair once)
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            pairs.append((float(sims[i, j]), case_ids[i], case_ids[j], prompts[i], prompts[j]))
    pairs.sort(key=lambda x: -x[0])

    with open(os.path.join(OUTPUT_DIR, "diversity_top_pairs.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["similarity", "case_id_a", "case_id_b", "prompt_a", "prompt_b"])
        for sim, a, b, pa, pb in pairs[:20]:
            writer.writerow([round(sim, 4), a, b, pa, pb])

    print("\nTop 10 most-similar pairs:")
    for sim, a, b, pa, pb in pairs[:10]:
        print(f"  {sim:.4f}  {a} <-> {b}")
        print(f"    A: {pa[:90]}")
        print(f"    B: {pb[:90]}")

    # Clustering pass (same approach as topic_diversity.py)
    clusterer = hdbscan.HDBSCAN(min_cluster_size=MIN_CLUSTER_SIZE, metric="euclidean")
    labels = clusterer.fit_predict(emb)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = int((labels == -1).sum())
    print(f"\nHDBSCAN (min_cluster_size={MIN_CLUSTER_SIZE}): {n_clusters} clusters, "
          f"{n_noise} singleton/noise points out of {n} total")
    for cluster_id in sorted(set(labels)):
        if cluster_id == -1:
            continue
        idx = np.where(labels == cluster_id)[0]
        print(f"  --- Cluster {cluster_id} (n={len(idx)}) ---")
        for i in idx:
            print(f"    [{case_ids[i]}] {prompts[i][:90]}")

    print("\nWrote outputs/diversity_similarity_matrix.csv and outputs/diversity_top_pairs.csv")
