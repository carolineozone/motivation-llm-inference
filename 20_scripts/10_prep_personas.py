"""
10_prep_personas.py
Split 00_personas.jsonl into a pilot set and a production set.

Usage:
    python 20_scripts/10_prep_personas.py [--n-pilot 200] [--n-prod 800]
"""

import argparse
import json
import os
import sys

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "10_data")
SOURCE_FILE = os.path.join(DATA_DIR, "00_personas.jsonl")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Split 00_personas.jsonl into pilot and prod sets.")
    p.add_argument("--n-pilot", type=int, default=200, help="Records for pilot file (default: 200)")
    p.add_argument("--n-prod", type=int, default=800, help="Records for prod file (default: 800)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    n_pilot = args.n_pilot
    n_prod = args.n_prod

    source_path = os.path.normpath(SOURCE_FILE)
    if not os.path.exists(source_path):
        sys.exit(f"Error: source file not found: {source_path}")

    # --- Scan source file ---
    with open(source_path, encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]

    total = len(records)
    size_kb = os.path.getsize(source_path) / 1024

    print(f"Source: {source_path}")
    print(f"  Records : {total}")
    print(f"  Size    : {size_kb:.1f} KB")
    print()

    if n_pilot + n_prod > total:
        sys.exit(
            f"Error: n_pilot ({n_pilot}) + n_prod ({n_prod}) = {n_pilot + n_prod} "
            f"exceeds total records ({total})."
        )

    pilot_path = os.path.normpath(os.path.join(DATA_DIR, f"00_personas_pilot_{n_pilot}.jsonl"))
    prod_path = os.path.normpath(os.path.join(DATA_DIR, f"00_personas_prod_{n_prod}.jsonl"))

    print(f"Will write:")
    print(f"  Pilot ({n_pilot} records) → {pilot_path}")
    print(f"  Prod  ({n_prod} records)  → {prod_path}")
    print()

    answer = input("Proceed? [y/N] ").strip().lower()
    if answer != "y":
        sys.exit("Aborted.")

    pilot_records = records[:n_pilot]
    prod_records = records[n_pilot: n_pilot + n_prod]

    with open(pilot_path, "w", encoding="utf-8") as f:
        for row in pilot_records:
            f.write(json.dumps(row) + "\n")

    with open(prod_path, "w", encoding="utf-8") as f:
        for row in prod_records:
            f.write(json.dumps(row) + "\n")

    print(f"Done.")
    print(f"  Written {len(pilot_records)} records → {pilot_path}")
    print(f"  Written {len(prod_records)} records → {prod_path}")


if __name__ == "__main__":
    main()
