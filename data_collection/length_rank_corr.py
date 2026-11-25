"""Spearman rank correlation between answer length and checklist score (motivation for the core/extra split).

Usage: python3 -I length_rank_corr.py PAIRS.csv
PAIRS.csv has a header and two columns, answer_length_chars and checklist_score_pct, one row per hand-graded case with a
gradable score (nine in the paper). Scores are full-checklist scores from before the core/extra tiering. The pairs are
per-case values and are not released. Recovered from the analysis session log, where it first ran inline.
"""
import csv
import sys

from scipy import stats

with open(sys.argv[1], newline="") as f:
    rows = [(float(r["answer_length_chars"]), float(r["checklist_score_pct"])) for r in csv.DictReader(f)]
rho, p = stats.spearmanr([a for a, _ in rows], [b for _, b in rows])
loo = [stats.spearmanr([a for j, (a, _) in enumerate(rows) if j != i], [b for j, (_, b) in enumerate(rows) if j != i])[0] for i in range(len(rows))]
print(f"n = {len(rows)}; Spearman rho = {rho:.3f}; p = {p:.4f}; leave-one-out range {min(loo):.2f} to {max(loo):.2f}")
