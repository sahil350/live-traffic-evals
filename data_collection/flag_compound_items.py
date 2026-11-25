"""Flag checklist items that bundle several requirements (the first-round heuristic behind the atomicity revision).

Usage: python3 -I flag_compound_items.py [MIN_PARTS]
Reads the Checklist sheet of the reference-checklist workbook (not released). Splits each criterion on commas and " and ",
keeps fragments of four words or fewer, and flags items with at least MIN_PARTS fragments (default 5, the first round).
The second round lowered the threshold to 4 for WIDTH items only; that code was not kept, but this script gives the same
rule with MIN_PARTS=4 and a WIDTH filter. Writes outputs/flagged_compound_items.json.
Recovered from the analysis session log, where it first ran inline; this file is a faithful copy of that logic.
"""
import json
import re
import sys
from pathlib import Path

import openpyxl

ROOT = Path(__file__).resolve().parents[1]
WORKBOOK = ROOT / "outputs" / "openresearcher_reference_checklists_batch1_equivalence_v1.xlsx"
MIN_PARTS = int(sys.argv[1]) if len(sys.argv) > 1 else 5

rows = list(openpyxl.load_workbook(WORKBOOK)["Checklist"].iter_rows(values_only=True))
idx = {h: i for i, h in enumerate(rows[0])}
flagged = []
for r in rows[1:]:
    parts = re.split(r",\s*|\s+and\s+", r[idx["criterion"]] or "")
    parts = [p for p in parts if len(p.split()) <= 4 and p.strip()]
    if len(parts) >= MIN_PARTS:
        flagged.append(dict(case_id=r[idx["case_id"]], item_id=r[idx["item_id"]], item_type=r[idx["item_type"]], n_parts=len(parts)))
print(f"total flagged (>= {MIN_PARTS} fragments): {len(flagged)}")
print("by item type:", {t: sum(f["item_type"] == t for f in flagged) for t in sorted({f["item_type"] for f in flagged})})
(ROOT / "outputs" / "flagged_compound_items.json").write_text(json.dumps(flagged, indent=2))
