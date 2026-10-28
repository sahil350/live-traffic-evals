"""Report on the all-split judge run (every multi-part core item split into atomic items).

Usage: python3 -I analyze_splitall.py
Reads outputs/judge_runs/judge_B_s* (old, 80-case full run, key/) and judge_B_splitall_s* (new, key_splitall/).
Writes outputs/judge_runs/report_splitall.md and splitall_per_case.csv.
"""
import csv
import json
from fractions import Fraction
from itertools import combinations
from pathlib import Path

RUNS = Path(__file__).resolve().parents[1] / "outputs" / "judge_runs"
SEEDS = (0, 1, 2)
load = lambda prefix, q: [json.loads((RUNS / f"{prefix}{s}" / f"{q}.json").read_text()) for s in SEEDS]
verd = lambda runs: [{i["item_id"]: i["judge_verdict"] for i in r["items"]} for r in runs]
maj = lambda vs, i: "pass" if sum(v[i] == "pass" for v in vs) >= 2 else "fail"


def score(key, pick):
    tot = sum(Fraction(str(k["weight"])) for k in key if k["tier"] == "core")
    got = sum(Fraction(str(k["weight"])) for k in key if k["tier"] == "core" and pick(k["item_id"]) == "pass")
    return None if tot == 0 else float(100 * got / tot)


cases = sorted(p.stem for p in (RUNS / "judge_B_splitall_s0").glob("*.json"))
rows, unsplit_same, unsplit_n, agree3, n_core_new, parents = [], 0, 0, 0, 0, {}
sub_pass = sub_n = 0
pair_old, pair_new = [], []
for q in cases:
    ko = json.loads((RUNS / "key" / f"{q}.json").read_text())["items"]
    kn = json.loads((RUNS / "key_splitall" / f"{q}.json").read_text())["items"]
    vo, vn = verd(load("judge_B_s", q)), verd(load("judge_B_splitall_s", q))
    old, new = score(ko, lambda i: maj(vo, i)), score(kn, lambda i: maj(vn, i))
    per_run_new = [score(kn, lambda i, v=v: v[i]) for v in vn]
    per_run_old = [score(ko, lambda i, v=v: v[i]) for v in vo]
    rows.append(dict(case=q, old=old, new=new, runs_old=per_run_old, runs_new=per_run_new,
                     n_core_old=sum(k["tier"] == "core" for k in ko), n_core_new=sum(k["tier"] == "core" for k in kn)))
    for k in kn:
        if k["tier"] != "core":
            continue
        i = k["item_id"]; n_core_new += 1
        agree3 += len({v[i] for v in vn}) == 1
        pair_new += [vn[a][i] == vn[b][i] for a, b in combinations(range(3), 2)]
        if "parent" in k:
            parents.setdefault((q, k["parent"]), []).append(maj(vn, i))
            sub_n += 1; sub_pass += maj(vn, i) == "pass"
        else:
            unsplit_n += 1; unsplit_same += maj(vo, i) == maj(vn, i)
    for k in ko:
        if k["tier"] == "core":
            pair_old += [vo[a][k["item_id"]] == vo[b][k["item_id"]] for a, b in combinations(range(3), 2)]
sc = [r for r in rows if r["old"] is not None]
d = [r["new"] - r["old"] for r in sc]
mean = lambda x: sum(x) / len(x)
med = lambda x: sorted(x)[len(x) // 2] if len(x) % 2 else sum(sorted(x)[len(x) // 2 - 1:len(x) // 2 + 1]) / 2
shape = {"all pass": 0, "some pass": 0, "none pass": 0}
for v in parents.values():
    shape["all pass" if all(x == "pass" for x in v) else "none pass" if not any(x == "pass" for x in v) else "some pass"] += 1
# fully-pilot cases vs the earlier pilot run
pilot = {}
for q in sorted(f.stem for f in (RUNS / "judge_B_split_s0").glob("*.json")):
    if q in cases and (RUNS / "judge_B_split_s0" / f"{q}.json").exists():
        a = verd(load("judge_B_split_s", q)); b = verd(load("judge_B_splitall_s", q))
        kn = json.loads((RUNS / "key_splitall" / f"{q}.json").read_text())["items"]
        subs = [k["item_id"] for k in kn if k["tier"] == "core" and "parent" in k and k["item_id"] in a[0]]
        pilot[q] = (sum(maj(a, i) == maj(b, i) for i in subs), len(subs))
L = ["# All-split run (every multi-part core item split into atomic items)", "",
     "Same judge (Muse Glimmer 30B 8-bit, prompt v2, 3 runs, majority) and same free-backend answers as the full run. Old = the 80-case full run. New = 72 cases with split packets. "
     "The 8 cases with nothing to split are not re-run. The split wording is a draft that has not been reviewed item by item.", "",
     f"- Cases re-judged: {len(rows)} ({len(sc)} with core items). Core items: {sum(r['n_core_old'] for r in rows)} before, {n_core_new} after (parents {len(parents)} became {sub_n} atomic items).",
     f"- Mean core score over the {len(sc)} cases: old {mean([r['old'] for r in sc]):.1f}%, new {mean([r['new'] for r in sc]):.1f}% ({mean(d):+.1f} pts). Median: old {med([r['old'] for r in sc]):.0f}%, new {med([r['new'] for r in sc]):.0f}%.",
     f"- Cases up / equal / down: {sum(x > 1e-9 for x in d)} / {sum(abs(x) <= 1e-9 for x in d)} / {sum(x < -1e-9 for x in d)}.",
     f"- Atomic items passed by majority: {sub_pass} of {sub_n} ({100 * sub_pass / sub_n:.0f}%). Parents by outcome: {shape['all pass']} all sub-items pass, {shape['some pass']} some, {shape['none pass']} none.",
     f"- Unsplit core items keeping the same majority verdict: {unsplit_same} of {unsplit_n}.",
     f"- Judge agreement on core items: all three runs agree on {agree3} of {n_core_new} ({100 * agree3 / n_core_new:.0f}%); pairwise run agreement {100 * sum(pair_new) / len(pair_new):.0f}% (old {100 * sum(pair_old) / len(pair_old):.0f}%).",
     f"- Pilot sub-items, verdict in the earlier pilot run vs this run (same majority): " + ", ".join(f"{q} {a}/{b}" for q, (a, b) in pilot.items()), "",
     "| Case | Old core % | New core % | Diff | Core items old→new | New per run |", "|---|---:|---:|---:|---|---|"]
dd = lambda r: 'n/a' if r['old'] is None else '%+.0f' % (r['new'] - r['old'])
f0 = lambda x: "n/a" if x is None else f"{x:.0f}"
for r in sorted(rows, key=lambda r: -(r["new"] - r["old"]) if r["old"] is not None else 1e9):
    L.append(f"| {r['case']} | {f0(r['old'])} | {f0(r['new'])} | {dd(r)} | {r['n_core_old']}→{r['n_core_new']} | {' / '.join(map(f0, r['runs_new']))} |")
(RUNS / "report_splitall.md").write_text("\n".join(L) + "\n")
with open(RUNS / "splitall_per_case.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["case", "old_core_pct", "new_core_pct", "n_core_old", "n_core_new"])
    for r in rows: w.writerow([r["case"], r["old"], r["new"], r["n_core_old"], r["n_core_new"]])
print("\n".join(L[:14]))
