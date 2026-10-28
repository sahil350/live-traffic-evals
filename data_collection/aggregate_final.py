"""Aggregate the final judge: Muse Glimmer 30B (8-bit), prompt v2, three runs (seeds 0, 1, 2), majority vote.

Usage: python3 -I aggregate_final.py
Reads outputs/judge_runs/judge_B_s{0,1,2}/<case>.json, key/<case>.json and (for the 10 reviewed cases) reference_ratings.json.
Writes outputs/judge_runs/final/per_item.csv, per_case.csv and summary.md.
Item verdict = majority of the three runs. Core score = weighted points passed / weighted points on core items.
Extras are reported separately and never change the score. Accuracy is the most common status across the runs
(a three-way split is reported as 'minor'). A case missing a run or failing to parse is listed under problems.
"""
import csv
import json
from collections import Counter
from pathlib import Path

RUNS = Path(__file__).resolve().parents[1] / "outputs" / "judge_runs"
SEEDS = (0, 1, 2)
OUT = RUNS / "final"
OUT.mkdir(exist_ok=True)
ref = json.loads((RUNS / "reference_ratings.json").read_text())["cases"]

cases = sorted(p.stem for p in (RUNS / "key").glob("*.json"))
item_rows, case_rows, problems = [], [], []
for q in cases:
    key = json.loads((RUNS / "key" / f"{q}.json").read_text())["items"]
    ids = [k["item_id"] for k in key]
    runs = {}
    for s in SEEDS:
        p = RUNS / f"judge_B_s{s}" / f"{q}.json"
        if not p.exists():
            continue
        o = json.loads(p.read_text())
        if [i.get("item_id") for i in o.get("items", [])] != ids:
            problems.append(f"{q} seed {s}: no valid output ({o.get('error', 'item ids differ')})")
            continue
        runs[s] = o
    if len(runs) < 3:
        problems.append(f"{q}: only {len(runs)} of 3 runs usable; skipped")
        continue
    verdict = {}
    for k in key:
        iid = k["item_id"]
        votes = [next(i for i in runs[s]["items"] if i["item_id"] == iid) for s in SEEDS]
        vs = [v["judge_verdict"] for v in votes]
        maj = "pass" if vs.count("pass") >= 2 else "fail"
        verdict[iid] = maj
        item_rows.append(dict(case=q, item=iid, tier=k["tier"], weight=k["weight"], type=k["item_type"],
                              votes="".join("P" if v == "pass" else "F" for v in vs), majority=maj,
                              unanimous=len(set(vs)) == 1, borderline_runs=sum(bool(v.get("borderline")) for v in votes),
                              reference=ref.get(q, {}).get("items", {}).get(iid, {}).get("verdict", "")))

    def pts(tier):
        tot = sum(k["weight"] for k in key if k["tier"] == tier)
        got = sum(k["weight"] for k in key if k["tier"] == tier and verdict[k["item_id"]] == "pass")
        return got, tot
    cg, ct = pts("core")
    eg, et = pts("extra")
    st = [runs[s]["accuracy"]["status"] for s in SEEDS]
    top, n = Counter(st).most_common(1)[0]
    case_rows.append(dict(case=q, core_points=cg, core_total=ct, core_pct=round(100 * cg / ct) if ct else "",
                          extra_points=eg, extra_total=et, accuracy=top if n >= 2 else "minor", accuracy_runs="/".join(st),
                          core_items_unanimous=sum(r["unanimous"] for r in item_rows if r["case"] == q and r["tier"] == "core"),
                          core_items=sum(1 for k in key if k["tier"] == "core")))

for name, rows in (("per_item.csv", item_rows), ("per_case.csv", case_rows)):
    with open(OUT / name, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

scored = [r for r in case_rows if r["core_total"]]
pct = sorted(r["core_pct"] for r in scored)
core_items = [r for r in item_rows if r["tier"] == "core"]
L = ["# Final judge results: Muse Glimmer 30B 8-bit, prompt v2, 3-run majority", "",
     f"Cases judged: {len(case_rows)}; cases with core items (core score defined): {len(scored)}; items: {len(item_rows)} ({len(core_items)} core).", ""]
if problems:
    L += ["## Problems", ""] + [f"- {p}" for p in problems] + [""]
L += ["## Core score across cases", "",
      f"- mean {sum(pct) / len(pct):.0f}%, median {pct[len(pct) // 2]}%, min {pct[0]}%, max {pct[-1]}%",
      f"- cases at 100%: {sum(p == 100 for p in pct)}; cases below 50%: {sum(p < 50 for p in pct)}",
      f"- core items where all three runs agreed: {sum(r['unanimous'] for r in core_items)} of {len(core_items)} "
      f"({100 * sum(r['unanimous'] for r in core_items) / len(core_items):.0f}%)",
      f"- accuracy note (majority across runs): " + ", ".join(f"{k} {v}" for k, v in sorted(Counter(r['accuracy'] for r in case_rows).items())),
      "", "## Core score by case", "", "| Case | Core score | Core items | Extras (not scored) | Accuracy |", "|---|---:|---:|---:|---|"]
for r in sorted(case_rows, key=lambda r: (r["core_pct"] == "", r["core_pct"] if r["core_pct"] != "" else 0)):
    L.append(f"| {r['case']} | {str(r['core_pct']) + '%' if r['core_pct'] != '' else 'n/a'} | {r['core_items']} | {r['extra_points']}/{r['extra_total']} | {r['accuracy']} |")
(OUT / "summary.md").write_text("\n".join(L) + "\n")
print("\n".join(L[:14]))
