"""Score judge outputs and measure agreement.

Usage: python3 -I score_judge_runs.py <tag> <judgeA_dir> <judgeB_dir> case1 case2 ...
Example: python3 -I score_judge_runs.py calibration judge_A judge_B <case_id> <case_id> ...

Reads outputs/judge_runs/<judge_dir>/<case>.json (judge output), key/<case>.json (tiers, weights) and
reference_ratings.json. Writes outputs/judge_runs/report_<tag>.md.
Core score = weighted points passed on core items / weighted points on core items. Extras are reported
separately and never change the core score.
"""
import json
import sys
from itertools import combinations
from pathlib import Path

RUNS = Path(__file__).resolve().parents[1] / "outputs" / "judge_runs"


def kappa(pairs):
    """Cohen's kappa for binary labels given as a list of (a, b) with values 'pass'/'fail'."""
    n = len(pairs)
    if n == 0:
        return None
    po = sum(a == b for a, b in pairs) / n
    pa = sum(a == "pass" for a, _ in pairs) / n
    pb = sum(b == "pass" for _, b in pairs) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    return None if pe == 1 else (po - pe) / (1 - pe)


def fmt(x):
    return "n/a" if x is None else f"{x:.2f}"


def load(judge, q):
    p = RUNS / judge / f"{q}.json"
    return json.loads(p.read_text()) if p.exists() else None


def main(tag, judges, cases):
    ref = json.loads((RUNS / "reference_ratings.json").read_text())["cases"]
    names = {"A": judges[0], "B": judges[1], "R": "reference"}
    rows = []  # one per item
    problems = []
    case_scores = {}
    acc = {}
    for q in cases:
        key = {i["item_id"]: i for i in json.loads((RUNS / "key" / f"{q}.json").read_text())["items"]}
        out = {"A": load(judges[0], q), "B": load(judges[1], q)}
        verdict = {"A": {}, "B": {}, "R": {k: v["verdict"] for k, v in ref[q]["items"].items()}}
        for j in ("A", "B"):
            o = out[j]
            if o is None:
                problems.append(f"{q}: no output from {names[j]}")
                continue
            got = [i.get("item_id") for i in o.get("items", [])]
            if got != list(key):
                problems.append(f"{q}: {names[j]} item ids/order differ from the packet ({got} vs {list(key)})")
            for i in o.get("items", []):
                if i.get("judge_verdict") not in ("pass", "fail"):
                    problems.append(f"{q} {i.get('item_id')}: {names[j]} verdict is {i.get('judge_verdict')!r}")
                else:
                    verdict[j][i["item_id"]] = i["judge_verdict"]
            acc[(q, j)] = (o.get("accuracy") or {}).get("status")
        for iid, k in key.items():
            rows.append({"case": q, "item": iid, "tier": k["tier"], "weight": k["weight"],
                         "A": verdict["A"].get(iid), "B": verdict["B"].get(iid), "R": verdict["R"].get(iid)})
        for j in ("A", "B", "R"):
            def pts(tier):
                e = p = 0
                for iid, k in key.items():
                    v = verdict[j].get(iid)
                    if v is None or k["tier"] != tier:
                        continue
                    p += k["weight"]
                    e += k["weight"] if v == "pass" else 0
                return e, p
            case_scores[(q, j)] = (pts("core"), pts("extra"))

    L = [f"# Judge run report: {tag}", "",
         f"Judge A = {names['A']}, judge B = {names['B']}, R = reference ratings (LLM-drafted, user-reviewed; not independent of a Claude judge).",
         f"Cases: {', '.join(cases)}", ""]
    if problems:
        L += ["## Problems (fix before trusting any number)", ""] + [f"- {p}" for p in problems] + [""]

    L += ["## Core score by case", "", "| Case | A | B | Reference | A-B gap (pts of 100) |", "|---|---:|---:|---:|---:|"]

    def pct(t):
        e, p = t
        return None if p == 0 else 100 * e / p
    for q in cases:
        s = {j: pct(case_scores[(q, j)][0]) for j in ("A", "B", "R")}
        gap = None if s["A"] is None or s["B"] is None else abs(s["A"] - s["B"])
        L.append(f"| {q} | {fmt(s['A'])} | {fmt(s['B'])} | {fmt(s['R'])} | {fmt(gap)} |")
    L += ["", "## Extras (not scored), points passed of points available", "", "| Case | A | B | Reference |", "|---|---:|---:|---:|"]
    for q in cases:
        c = [f"{case_scores[(q, j)][1][0]}/{case_scores[(q, j)][1][1]}" for j in ("A", "B", "R")]
        L.append(f"| {q} | {c[0]} | {c[1]} | {c[2]} |")

    L += ["", "## Item-level agreement (pass/fail)", "", "| Pair | Items | Agreement | Kappa |", "|---|---:|---:|---:|"]
    for scope in ("all", "core", "extra"):
        for x, y in (("A", "B"), ("A", "R"), ("B", "R")):
            pairs = [(r[x], r[y]) for r in rows if r[x] and r[y] and (scope == "all" or r["tier"] == scope)]
            agree = sum(a == b for a, b in pairs) / len(pairs) if pairs else None
            L.append(f"| {scope}: {names[x]} vs {names[y]} | {len(pairs)} | {fmt(agree)} | {fmt(kappa(pairs))} |")
    L += ["", "With a few dozen items per scope these numbers have wide intervals. Read them as a smoke test, not an estimate.", ""]

    L += ["## Every disagreement", "", "| Case | Item | Tier | A | B | Reference |", "|---|---|---|---|---|---|"]
    n = 0
    for r in rows:
        v = {r["A"], r["B"], r["R"]} - {None}
        if len(v) > 1:
            n += 1
            L.append(f"| {r['case']} | {r['item']} | {r['tier']} | {r['A']} | {r['B']} | {r['R']} |")
    if n == 0:
        L.append("| none | | | | | |")

    L += ["", "## Accuracy note (reported separately)", "", "| Case | A | B | Reference |", "|---|---|---|---|"]
    for q in cases:
        L.append(f"| {q} | {acc.get((q, 'A'))} | {acc.get((q, 'B'))} | {ref[q]['accuracy']} |")
    (RUNS / f"report_{tag}.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    tag, ja, jb, *cases = sys.argv[1:]
    main(tag, (ja, jb), cases)
