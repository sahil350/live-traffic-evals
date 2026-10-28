"""Two reports from the judge runs: (1) Serper-backed answers vs free-backend answers (same judge, same prompt),
(2) the rubric split pilot (multi-part core items split into atomic items).

Usage: python3 -I analyze_serper_and_split.py
Reads outputs/judge_runs/judge_B_s*, judge_B_serper_s*, judge_B_split_s*, key/, key_split/, packets cases.
Writes outputs/judge_runs/report_serper_compare.md and report_split_pilot.md.
"""
import json
from fractions import Fraction
from itertools import combinations
from pathlib import Path

RUNS = Path(__file__).resolve().parents[1] / "outputs" / "judge_runs"
SEEDS = (0, 1, 2)


def load(prefix, q):
    return [json.loads((RUNS / f"{prefix}{s}" / f"{q}.json").read_text()) for s in SEEDS]


def verdicts(runs):
    return [{i["item_id"]: i["judge_verdict"] for i in r["items"]} for r in runs]


def maj(vs, iid):
    return "pass" if sum(v[iid] == "pass" for v in vs) >= 2 else "fail"


def score(key, pick, tier="core"):
    tot = sum(Fraction(str(k["weight"])) for k in key if k["tier"] == tier)
    got = sum(Fraction(str(k["weight"])) for k in key if k["tier"] == tier and pick(k["item_id"]) == "pass")
    return None if tot == 0 else float(100 * got / tot)


def f0(x):
    return "n/a" if x is None else f"{x:.0f}"


# ---------- 1. Serper vs free ----------
cases = sorted(p.stem for p in (RUNS / "judge_B_serper_s0").glob("*.json"))
rows, within, across, within_s, within_f = [], [], [], [], []
for q in cases:
    key = json.loads((RUNS / "key" / f"{q}.json").read_text())["items"]
    core = [k["item_id"] for k in key if k["tier"] == "core"]
    free, ser = verdicts(load("judge_B_s", q)), verdicts(load("judge_B_serper_s", q))
    ff = [score(key, lambda i, v=v: v[i]) for v in free]
    ss = [score(key, lambda i, v=v: v[i]) for v in ser]
    mf = score(key, lambda i: maj(free, i))
    ms = score(key, lambda i: maj(ser, i))
    same = [i for i in core if maj(free, i) == maj(ser, i)]
    rows.append((q, mf, ms, len(same), len(core), ff, ss, [i for i in core if maj(free, i) != maj(ser, i)]))
    for i in core:
        for a, b in combinations(range(3), 2):
            within_f.append(free[a][i] == free[b][i]); within_s.append(ser[a][i] == ser[b][i])
        across += [free[s][i] == ser[s][i] for s in range(3)]
within = within_f + within_s
rate = lambda x: 100 * sum(x) / len(x)
sc = [r for r in rows if r[1] is not None and r[2] is not None]
d = [r[2] - r[1] for r in sc]
L = ["# Serper-backed answers vs free-backend answers", "",
     "Same judge (Muse Glimmer 30B 8-bit, prompt v2, 3 runs, majority). Cases with no checklist, nothing scorable or an empty final answer were not judged. "
     "The two answer sets differ in search backend and in ordinary sampling, so a gap cannot be pinned on the backend.", "",
     f"- Cases judged: {len(rows)} ({len(sc)} with core items).",
     f"- Mean core score: free {sum(r[1] for r in sc) / len(sc):.1f}%, Serper {sum(r[2] for r in sc) / len(sc):.1f}%; mean per-case difference (Serper - free) {sum(d) / len(d):+.1f} pts; mean absolute difference {sum(abs(x) for x in d) / len(d):.1f} pts.",
     f"- Cases where Serper scored higher / equal / lower: {sum(x > 0 for x in d)} / {sum(x == 0 for x in d)} / {sum(x < 0 for x in d)}.",
     f"- Core-item verdict agreement (majority vs majority, same item, different answer): {sum(r[3] for r in rows)} of {sum(r[4] for r in rows)} ({100 * sum(r[3] for r in rows) / sum(r[4] for r in rows):.0f}%).",
     f"- For scale, pairwise agreement between two runs of the judge on the same answer: free {rate(within_f):.0f}%, Serper {rate(within_s):.0f}%; same seed across the two answers: {rate(across):.0f}%.", "",
     "| Case | Free core % | Serper core % | Diff | Core items same verdict | Free per run | Serper per run | Items that differ |", "|---|---:|---:|---:|---:|---|---|---|"]
for q, mf, ms, n, t, ff, ss, dif in sorted(rows, key=lambda r: (r[1] is None or r[2] is None, -(abs(r[2] - r[1]) if r[1] is not None and r[2] is not None else 0))):
    L.append(f"| {q} | {f0(mf)} | {f0(ms)} | {'n/a' if mf is None or ms is None else f'{ms - mf:+.0f}'} | {n}/{t} | {' / '.join(map(f0, ff))} | {' / '.join(map(f0, ss))} | {', '.join(dif)} |")
(RUNS / "report_serper_compare.md").write_text("\n".join(L) + "\n")
print("\n".join(L[:12]))

# ---------- 2. Split pilot ----------
spec = json.loads((RUNS.parents[1] / "data_collection" / "rubric_split_pilot.json").read_text())["splits"]
P = ["# Rubric split pilot (option 1)", "",
     "Ten multi-part core items split into 23 atomic items; parent weight divided equally. Same judge and answers as the full run. Old = the parent item in the full run (majority of 3). New = the sub-items in this run (majority of 3). "
     "Provisional ref = our rating derived from the reviewed rationale (to be confirmed by the user); only for the reference cases.", ""]
by_case = {}
for s in spec:
    by_case.setdefault(s["case"], []).append(s)
tot_old = tot_new = 0
case_lines, stab = [], []
for q, splits in by_case.items():
    okey = json.loads((RUNS / "key" / f"{q}.json").read_text())["items"]
    nkey = json.loads((RUNS / "key_split" / f"{q}.json").read_text())["items"]
    old, new = verdicts(load("judge_B_s", q)), verdicts(load("judge_B_split_s", q))
    parents = {s["parent"] for s in splits}
    so, sn = score(okey, lambda i: maj(old, i)), score(nkey, lambda i: maj(new, i))
    case_lines.append(f"| {q} | {f0(so)} | {f0(sn)} | {sn - so:+.0f} |")
    for k in okey:
        if k["item_id"] not in parents:
            stab.append(maj(old, k["item_id"]) == maj(new, k["item_id"]))
    for s in splits:
        o = maj(old, s["parent"])
        votes = ", ".join(f"{x['id']} {''.join('P' if v[x['id']] == 'pass' else 'F' for v in new)}=>{maj(new, x['id'])}"
                          + (f" (prov. ref {x['provisional_ref']})" if x.get("provisional_ref") else "") for x in s["subs"])
        P.append(f"- **{q}.{s['parent']}** old judge: {''.join('P' if v[s['parent']] == 'pass' else 'F' for v in old)}=>{o}"
                 + (f"; ours (parent): {s['parent_ref']}" if s["parent_ref"] else "") + f". New: {votes}")
P += ["", "## Case core score, old vs new (majority)", "", "| Case | Old core % | New core % | Change |", "|---|---:|---:|---:|"] + case_lines
P += ["", f"Unchanged items (not split) with the same majority verdict as in the full run: {sum(stab)} of {len(stab)}."]
(RUNS / "report_split_pilot.md").write_text("\n".join(P) + "\n")
print("\n".join(P))
