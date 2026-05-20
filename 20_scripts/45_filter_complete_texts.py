"""
45_filter_complete_texts.py

Filter a stage-40 text generation JSONL to personas with a complete set of texts
(all coworker_id × target_words combinations present).

Personas with a complete set go to a *_included_* file.
Personas missing any texts go to a *_excluded_* file.

Usage:
    python 20_scripts/45_filter_complete_texts.py <input_file>
    python 20_scripts/45_filter_complete_texts.py <input_file> --dry-run
"""

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path


def load_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, records: list[dict]) -> None:
    if path.exists():
        raise RuntimeError(f"Output file already exists: {path}")
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Filter stage-40 JSONL to personas with complete text sets."
    )
    parser.add_argument("input_file", help="Path to stage-40 output JSONL")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print summary without writing output files")
    args = parser.parse_args()

    input_path = Path(args.input_file)
    if not input_path.exists():
        print(f"Error: file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    rows = load_jsonl(input_path)
    for r in rows:
        r["word_count"] = len((r.get("text") or "").split())

    # Determine expected combinations from the data itself
    target_words_vals = sorted(set(r["target_words"] for r in rows))
    coworker_id_vals = sorted(set(r["coworker_id"] for r in rows))
    expected_count = len(target_words_vals) * len(coworker_id_vals)

    print(f"Input:          {input_path.name}")
    print(f"Total rows:     {len(rows)}")
    print(f"target_words:   {target_words_vals}")
    print(f"coworker_ids:   {coworker_id_vals}")
    print(f"Expected / persona: {expected_count}")
    print()

    # Group rows by persona_uuid, preserving order
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        groups[r["persona_uuid"]].append(r)

    included_uuids = []
    excluded_uuids = []
    excluded_reasons = {}

    for uuid, persona_rows in groups.items():
        if len(persona_rows) == expected_count:
            included_uuids.append(uuid)
        else:
            excluded_uuids.append(uuid)
            present = {(r["coworker_id"], r["target_words"]) for r in persona_rows}
            expected_set = {
                (cid, tw) for cid in coworker_id_vals for tw in target_words_vals
            }
            missing = sorted(expected_set - present)
            excluded_reasons[uuid] = missing

    included_rows = [r for r in rows if r["persona_uuid"] in set(included_uuids)]

    print(f"Personas included (complete): {len(included_uuids)}")
    print(f"Personas excluded (missing texts): {len(excluded_uuids)}")
    for uuid, missing in excluded_reasons.items():
        print(f"  …{uuid[-8:]}  missing: {missing}")
    print()

    if args.dry_run:
        print("Dry run — no files written.")
        return

    # Strip the original timestamp and _n{count} suffix so the output name
    # is e.g. "40_pilot_textgeneration_v2_20260324-0902_included_49" instead of
    # "40_pilot_textgeneration_v2_20260323-1503_n50_20260324-0902_included_49".
    stage_stem = "45_filter_complete_texts"
    timestamp = datetime.now().strftime("%Y%m%d-%H%M")
    data_dir = Path("10_data")

    out_included = data_dir / f"{stage_stem}_{timestamp}_included_{len(included_uuids)}.jsonl"
    out_excluded = data_dir / f"{stage_stem}_{timestamp}_excluded_{len(excluded_uuids)}.jsonl"

    write_jsonl(out_included, included_rows)
    write_jsonl(out_excluded, [r for r in rows if r["persona_uuid"] in set(excluded_uuids)])

    print(f"Included -> {out_included}  ({len(included_rows)} rows)")
    print(f"Excluded -> {out_excluded}  ({len(rows) - len(included_rows)} rows)")


if __name__ == "__main__":
    main()
