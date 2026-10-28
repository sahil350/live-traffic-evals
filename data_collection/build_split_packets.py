"""Build blind packets for the rubric split pilot (option 1: multi-part core items split into atomic items).

Usage: python3 -I build_split_packets.py [spec.json tag]   (default: rubric_split_pilot.json, tag "split")
Reads the spec (data_collection/<spec>). Builds normal packets for the affected cases into
outputs/judge_runs/packets_<tag>/, then replaces each parent item block with its sub-item blocks.
Writes outputs/judge_runs/key_<tag>/<case>.json (sub-item weights, tier core). The production key/ is not changed.
"""
import json
import re
import sys
from fractions import Fraction
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import build_judge_packets as B

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "outputs" / "judge_runs"
SPEC = sys.argv[1] if len(sys.argv) > 1 else "rubric_split_pilot.json"
TAG = sys.argv[2] if len(sys.argv) > 2 else "split"
spec = json.loads((ROOT / "data_collection" / SPEC).read_text())["splits"]
by_case = {}
for s in spec:
    by_case.setdefault(s["case"], []).append(s)

B.main(list(by_case), B.PROMPT, TAG)
(RUNS / f"key_{TAG}").mkdir(exist_ok=True)
for q, splits in by_case.items():
    pf = RUNS / f"packets_{TAG}" / f"{q}.md"
    head, body = pf.read_text().split("## CHECKLIST ITEMS (grade every item listed)\n", 1)
    blocks = re.split(r"\n(?=### )", body.strip("\n"))
    key = json.loads((RUNS / "key" / f"{q}.json").read_text())
    kitems = {i["item_id"]: i for i in key["items"]}
    sp = {s["parent"]: s for s in splits}
    out_blocks, out_keys = [], []
    for b in blocks:
        iid = re.match(r"### (\S+) \[(\w+)\]", b)
        pid, typ = iid.group(1), iid.group(2)
        if pid not in sp:
            out_blocks.append(b)
            out_keys.append(kitems[pid])
            continue
        parent = kitems[pid]
        assert parent["tier"] == "core" and Fraction(sp[pid]["parent_weight"]) == parent["weight"], (q, pid)
        assert sum(Fraction(x["weight"]) for x in sp[pid]["subs"]) == parent["weight"], (q, pid)
        for x in sp[pid]["subs"]:
            out_blocks.append(f"### {x['id']} [{typ}]\nRequirement: {x['requirement']}")
            out_keys.append({**parent, "item_id": x["id"], "weight": float(Fraction(x["weight"])), "parent": pid})
    pf.write_text(head + "## CHECKLIST ITEMS (grade every item listed)\n\n" + "\n\n".join(out_blocks) + "\n")
    (RUNS / f"key_{TAG}" / f"{q}.json").write_text(json.dumps({"case_id": q, "items": out_keys}, indent=1))
    print(q, "items now:", [k["item_id"] for k in out_keys])
