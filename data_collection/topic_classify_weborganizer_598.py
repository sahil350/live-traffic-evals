"""
Same as topic_classify_weborganizer.py, but against the larger 598-row
final_positives.csv (pre-audit algorithmic set) instead of the 87-row
hand-verified working set -- checking whether topics absent from the 87
(History, Politics, Religion) are a real gap in the traffic or an artifact
of the much smaller n.
"""

import csv

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_NAME = "WebOrganizer/TopicClassifier-NoURL"

if __name__ == "__main__":
    with open("outputs/final_positives.csv") as f:
        rows = list(csv.DictReader(f))
    print(f"Input set: {len(rows)} rows")

    print(f"Loading {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, trust_remote_code=True, use_memory_efficient_attention=False
    )
    model.eval()

    id2label = model.config.id2label

    results = []
    with torch.no_grad():
        for i, r in enumerate(rows):
            text = r["prompt"]
            inputs = tokenizer([text], return_tensors="pt", truncation=True, max_length=512)
            outputs = model(**inputs)
            probs = outputs.logits.softmax(dim=-1)[0]
            top3_idx = torch.argsort(probs, descending=True)[:3]

            results.append({
                "source": r["source"],
                "prompt": r["prompt"],
                "our_level1": r["level1"],
                "our_level2": r["level2"],
                "weborganizer_top1": id2label[int(top3_idx[0])],
                "weborganizer_top1_prob": round(float(probs[top3_idx[0]]), 4),
                "weborganizer_top2": id2label[int(top3_idx[1])],
                "weborganizer_top2_prob": round(float(probs[top3_idx[1]]), 4),
                "weborganizer_top3": id2label[int(top3_idx[2])],
                "weborganizer_top3_prob": round(float(probs[top3_idx[2]]), 4),
            })

            if (i + 1) % 50 == 0 or (i + 1) == len(rows):
                print(f"  {i+1}/{len(rows)} classified")

    with open("outputs/weborganizer_topic_results_598.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    print(f"\nWrote outputs/weborganizer_topic_results_598.csv ({len(results)} rows)")

    from collections import Counter
    dist = Counter(r["weborganizer_top1"] for r in results)
    print("\nTop-1 topic distribution (598 set):")
    for label, n in dist.most_common():
        print(f"  {label}: {n} ({100*n/len(results):.1f}%)")

    all_labels = set(id2label.values())
    seen_labels = set(dist.keys())
    missing = all_labels - seen_labels
    print(f"\nCategories with ZERO occurrences even in the 598 set: {sorted(missing)}")
