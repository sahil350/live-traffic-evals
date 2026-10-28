"""
How many distinct topics do the LLM-confirmed positives cover? Loads the
329 combined positives from lmsys_llm_classified.csv + arena_llm_classified.csv,
embeds them (bge-large-en-v1.5, same model as the main pipeline), and clusters
with HDBSCAN using a smaller min_cluster_size than Stage 1 (appropriate for a
few hundred points instead of thousands).

Usage:
    python topic_diversity.py
"""

import csv
import os

import numpy as np
from sentence_transformers import SentenceTransformer
import hdbscan

OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "../outputs")
MODEL_NAME = "BAAI/bge-large-en-v1.5"
MIN_CLUSTER_SIZE = int(os.environ.get("TOPIC_MIN_CLUSTER_SIZE", 4))


def load_positives(path, source_name):
    with open(path) as f:
        rows = [r for r in csv.DictReader(f) if r["llm_label"] == "POSITIVE"]
    return [{"source": source_name, "prompt": r["prompt"]} for r in rows]


if __name__ == "__main__":
    records = (
        load_positives(os.path.join(OUTPUT_DIR, "lmsys_llm_classified.csv"), "LMSYS")
        + load_positives(os.path.join(OUTPUT_DIR, "arena_llm_classified.csv"), "ARENA")
    )
    prompts = [r["prompt"] for r in records]
    print(f"Embedding {len(prompts)} confirmed positives...")

    model = SentenceTransformer(MODEL_NAME)
    emb = model.encode(prompts, normalize_embeddings=True, show_progress_bar=False)

    clusterer = hdbscan.HDBSCAN(min_cluster_size=MIN_CLUSTER_SIZE, metric="euclidean")
    labels = clusterer.fit_predict(emb)

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = int((labels == -1).sum())
    print(f"\n{n_clusters} distinct topic clusters found (min_cluster_size={MIN_CLUSTER_SIZE}), "
          f"{n_noise} noise/singleton points out of {len(prompts)} total\n")

    # Representative prompts per cluster: the 3 closest to the cluster centroid
    for cluster_id in sorted(set(labels)):
        if cluster_id == -1:
            continue
        idx = np.where(labels == cluster_id)[0]
        centroid = emb[idx].mean(axis=0)
        centroid /= np.linalg.norm(centroid)
        sims = emb[idx] @ centroid
        top = idx[np.argsort(-sims)[:3]]
        print(f"--- Cluster {cluster_id} (n={len(idx)}) ---")
        for i in top:
            print(f"  [{records[i]['source']}] {records[i]['prompt'][:120]}")
        print()

    print(f"--- Noise / one-off topics (n={n_noise}) ---")
    noise_idx = np.where(labels == -1)[0]
    sample_n = min(30, len(noise_idx))
    rng = np.random.default_rng(7)
    sample = rng.choice(noise_idx, size=sample_n, replace=False)
    for i in sample:
        print(f"  [{records[i]['source']}] {records[i]['prompt']}")
        print()
