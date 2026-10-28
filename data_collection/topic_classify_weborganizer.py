"""
Runs WebOrganizer/TopicClassifier-NoURL (Wettig et al. 2025, "Organize the Web")
over the 87-record working set, as an independent, externally-trained topic tag
on top of this project's own Academic/Business/Knowledge-field domain labels.

140M-param gte-base-en-v1.5 sequence classifier, 24 web-content topic categories,
plain-text input (just the prompt text, no special formatting).

Usage:
    python topic_classify_weborganizer.py
"""

import csv

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_NAME = "WebOrganizer/TopicClassifier-NoURL"

if __name__ == "__main__":
    with open("outputs/working_set_151sample.csv") as f:
        working = list(csv.DictReader(f))
    print(f"Working set: {len(working)} rows")

    print(f"Loading {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, trust_remote_code=True, use_memory_efficient_attention=False
    )
    model.eval()

    id2label = model.config.id2label

    results = []
    with torch.no_grad():
        for i, r in enumerate(working):
            text = r["prompt"]
            inputs = tokenizer([text], return_tensors="pt", truncation=True, max_length=512)
            outputs = model(**inputs)
            probs = outputs.logits.softmax(dim=-1)[0]
            top3_idx = torch.argsort(probs, descending=True)[:3]

            top1_label = id2label[int(top3_idx[0])]
            top1_prob = float(probs[top3_idx[0]])
            top2_label = id2label[int(top3_idx[1])]
            top2_prob = float(probs[top3_idx[1]])
            top3_label = id2label[int(top3_idx[2])]
            top3_prob = float(probs[top3_idx[2]])

            results.append({
                "case_id": r["case_id"],
                "prompt": r["prompt"],
                "our_level2_domain": r["stratum"].split(" x ")[0] if " x " in r["stratum"] else r["stratum"],
                "weborganizer_top1": top1_label,
                "weborganizer_top1_prob": round(top1_prob, 4),
                "weborganizer_top2": top2_label,
                "weborganizer_top2_prob": round(top2_prob, 4),
                "weborganizer_top3": top3_label,
                "weborganizer_top3_prob": round(top3_prob, 4),
            })

            if (i + 1) % 20 == 0 or (i + 1) == len(working):
                print(f"  {i+1}/{len(working)} classified")

    with open("outputs/weborganizer_topic_results.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    print(f"\nWrote outputs/weborganizer_topic_results.csv ({len(results)} rows)")

    from collections import Counter
    dist = Counter(r["weborganizer_top1"] for r in results)
    print("\nTop-1 topic distribution:")
    for label, n in dist.most_common():
        print(f"  {label}: {n} ({100*n/len(results):.1f}%)")
