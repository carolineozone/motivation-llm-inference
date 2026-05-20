"""
35_extract_personas.py

Extract persona records from a source JSONL based on a coworker-filter output file.
Personas with has_coworkers=true go to a *_included_* file;
personas with has_coworkers=false go to a *_excluded_* file.

Usage:
    python 20_scripts/35_extract_personas.py <filter_file> <personas_file>
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


def choose_variant(variants: list[str]) -> str:
    print("\nMultiple variant_ids found in filter file:")
    for i, v in enumerate(sorted(variants), 1):
        print(f"  {i}. {v}")
    while True:
        choice = input("Choose variant_id to use for filtering (enter number or name): ").strip()
        if choice.isdigit():
            idx = int(choice) - 1
            if 0 <= idx < len(variants):
                return sorted(variants)[idx]
        elif choice in variants:
            return choice
        print(f"  Invalid choice. Enter a number 1-{len(variants)} or the variant name.")


def main():
    parser = argparse.ArgumentParser(description="Extract personas based on coworker filter results.")
    parser.add_argument("filter_file", help="Path to coworkerfilter output JSONL")
    parser.add_argument("personas_file", help="Path to source personas JSONL")
    parser.add_argument("--variant", default=None,
                        help="variant_id to use for filtering (skips interactive prompt; "
                             "required when multiple variants exist in the filter file)")
    args = parser.parse_args()

    filter_path = Path(args.filter_file)
    personas_path = Path(args.personas_file)

    for p in (filter_path, personas_path):
        if not p.exists():
            print(f"Error: file not found: {p}", file=sys.stderr)
            sys.exit(1)

    # --- Load and deduplicate filter file ---
    filter_records = load_jsonl(filter_path)

    # Group by uuid → set of variant_ids
    uuid_variants: dict[str, set[str]] = defaultdict(set)
    for row in filter_records:
        uuid_variants[row["persona_uuid"]].add(row.get("variant_id", ""))

    all_variants: set[str] = set()
    for vs in uuid_variants.values():
        all_variants |= vs

    selected_variant: str | None = None
    if len(all_variants) > 1:
        if args.variant:
            if args.variant not in all_variants:
                print(f"Error: --variant {args.variant!r} not found in filter file. "
                      f"Available: {sorted(all_variants)}", file=sys.stderr)
                sys.exit(1)
            selected_variant = args.variant
            print(f"Using variant_id: {selected_variant!r}  (from --variant)")
        else:
            selected_variant = choose_variant(list(all_variants))
            print(f"Using variant_id: {selected_variant!r}")
        filter_records = [r for r in filter_records if r.get("variant_id") == selected_variant]
    else:
        # Single variant or no variant field — deduplicate by keeping first occurrence per UUID
        seen: set[str] = set()
        deduped = []
        for r in filter_records:
            if r["persona_uuid"] not in seen:
                deduped.append(r)
                seen.add(r["persona_uuid"])
        filter_records = deduped

    # Build included/excluded UUID sets
    included_uuids: set[str] = set()
    excluded_uuids: set[str] = set()
    for row in filter_records:
        uuid = row["persona_uuid"]
        if row.get("has_coworkers"):
            included_uuids.add(uuid)
        else:
            excluded_uuids.add(uuid)

    filter_uuids = included_uuids | excluded_uuids

    # --- Load personas ---
    personas = load_jsonl(personas_path)
    persona_by_uuid = {p["uuid"]: p for p in personas}
    personas_uuids = set(persona_by_uuid.keys())

    # --- Warnings ---
    missing_in_personas = filter_uuids - personas_uuids
    if missing_in_personas:
        print(f"\nWarning: {len(missing_in_personas)} UUID(s) from filter file not found in personas file:")
        for u in sorted(missing_in_personas):
            print(f"  {u}")

    unprocessed = personas_uuids - filter_uuids
    if unprocessed:
        print(f"\nWarning: {len(unprocessed)} persona UUID(s) not present in filter file (unprocessed):")
        for u in sorted(unprocessed):
            print(f"  {u}")

    # --- Build output records ---
    included_personas = [persona_by_uuid[u] for u in included_uuids if u in persona_by_uuid]
    excluded_personas = [persona_by_uuid[u] for u in excluded_uuids if u in persona_by_uuid]

    # --- Output paths ---
    source_stem = personas_path.stem[3:]  # strip leading "00_"
    timestamp = datetime.now().strftime("%Y%m%d-%H%M")
    data_dir = Path("10_data")

    out_included = data_dir / f"35_{source_stem}_{timestamp}_included_{len(included_personas)}.jsonl"
    out_excluded = data_dir / f"35_{source_stem}_{timestamp}_excluded_{len(excluded_personas)}.jsonl"

    write_jsonl(out_included, included_personas)
    write_jsonl(out_excluded, excluded_personas)

    print(f"\nDone.")
    print(f"  Included (has_coworkers=true):  {len(included_personas):>4}  ->  {out_included}")
    print(f"  Excluded (has_coworkers=false): {len(excluded_personas):>4}  ->  {out_excluded}")
    if missing_in_personas:
        print(f"  Warnings: {len(missing_in_personas)} UUID(s) from filter not found in personas file")
    if unprocessed:
        print(f"  Warnings: {len(unprocessed)} persona(s) not in filter file")


if __name__ == "__main__":
    main()
