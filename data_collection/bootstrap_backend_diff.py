"""Paired difference between the two search backends, with a percentile bootstrap interval.

Usage: python3 -I bootstrap_backend_diff.py
Reads outputs/judge_runs/judge_B_s*, judge_B_serper_s* and key/ (free-backend and commercial-API answers, same judge and prompt).
Per case: core score from the majority of three runs, exact weights. Cases without a core item are dropped (25 remain).
Difference = commercial-API score minus free-stack score. Interval: random.seed(1), 20,000 resamples of the paired
differences, 2.5th and 97.5th percentiles (sorted positions 500 and 19500). Also prints the SD and a t-based interval.
Recovered from the analysis session log, where it first ran inline on unrounded pairs; the rounded percentages in
report_serper_compare.md give a slightly different interval (-4.4 to +16.1), so run this on the raw judge output.
"""
import json
import random
import statistics as st
from fractions import Fraction
from pathlib import Path

RUNS = Path(__file__).resolve().parents[1] / "outputs" / "judge_runs"
load = lambda prefix, q: [{i["item_id"]: i["judge_verdict"] for i in json.loads((RUNS / f"{prefix}{s}" / f"{q}.json").read_text())["items"]} for s in (0, 1, 2)]
maj = lambda vs, i: "pass" if sum(v[i] == "pass" for v in vs) >= 2 else "fail"


def score(key, vs):
    core = [k for k in key if k["tier"] == "core"]
    tot = sum(Fraction(str(k["weight"])) for k in core)
    return None if tot == 0 else float(100 * sum(Fraction(str(k["weight"])) for k in core if maj(vs, k["item_id"]) == "pass") / tot)


d = []
for p in sorted((RUNS / "judge_B_serper_s0").glob("*.json")):
    q = p.stem
    key = json.loads((RUNS / "key" / f"{q}.json").read_text())["items"]
    free, com = score(key, load("judge_B_s", q)), score(key, load("judge_B_serper_s", q))
    if free is not None and com is not None:
        d.append(com - free)
random.seed(1)
means = sorted(st.mean(random.choices(d, k=len(d))) for _ in range(20000))
n, m, sd = len(d), st.mean(d), st.stdev(d)
half = 2.0639 * sd / n ** 0.5  # t critical value, 24 degrees of freedom
print(f"cases {n}; mean difference {m:+.1f}; SD {sd:.1f}; mean absolute {st.mean(abs(x) for x in d):.1f}")
print(f"bootstrap 95% interval {means[500]:+.1f} to {means[19500]:+.1f}; t-based {m - half:+.1f} to {m + half:+.1f}")
