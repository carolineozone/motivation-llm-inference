"""
70_merge_outputs.py — Merge BPNS completions and scoring results into analysis CSV.

Joins stage 50 (BPNS completions) and stage 60 (LLM scores) on persona_uuid.
Optionally joins demographics from the personas JSONL (--personas-file).
Output: one row per persona × target_words × n_texts × repeat_id (13,500 rows for full production run).
Texts are not included — they are intermediate pipeline data, not needed for analysis.

Output columns:
    uuid,
    age, gender, education, occupation, country, race, nationality, income_category,
                                                        (demographics, constant per persona)
    target_words, n_texts, repeat_id,
    q_aut_1..4, q_com_1..4, q_rel_1..4,               (BPNS self-report, constant per persona)
    word_count,                                         (sum of word counts for scored texts)
    llm_aut, llm_com, llm_rel                          (LLM scores, vary per condition)

Usage (standalone):
    python 20_scripts/70_merge_outputs.py \\
        --bpns-file     10_data/50_pilot_complete_bpns_...jsonl \\
        --scores-file   10_data/60_pilot_scoring_...jsonl \\
        --personas-file 10_data/00_personas_pilot_200.jsonl
"""

import argparse
import csv
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR   = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_DIR     = PROJECT_ROOT / "10_data"

# ── Demographic column mapping (raw survey key → CSV column name) ──────────────
DEMO_COLUMN_MAP = {
    "Select Your Age":                                               "age",
    "Select Your Gender":                                            "gender",
    "Select Your Highest Level of Education":                        "education",
    "Provide Your Occupation. (_NA if not applicable_)":             "occupation",
    "Provide Your Country of Residence.":                            "country",
    "Select Your Race":                                              "race",
    "Provide Your Nationality.":                                     "nationality",
    "Select Your Income Category":                                   "income_category",
}

# ── Output columns (fixed — matches RStudio expectations) ─────────────────────
FIELDNAMES = [
    "uuid",
    "age", "gender", "education", "occupation", "country", "race", "nationality", "income_category",
    "target_words", "n_texts", "repeat_id",
    "q_aut_1", "q_aut_2", "q_aut_3", "q_aut_4",
    "q_com_1", "q_com_2", "q_com_3", "q_com_4",
    "q_rel_1", "q_rel_2", "q_rel_3", "q_rel_4",
    "word_count",
    "llm_aut", "llm_com", "llm_rel",
]

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ── CLI ───────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Merge BPNS and scoring outputs into analysis CSV.")
    p.add_argument("--bpns-file",   required=True, help="Stage 50 output JSONL (BPNS completions)")
    p.add_argument("--scores-file", required=True, help="Stage 60 output JSONL (LLM scores)")
    p.add_argument("--output-file", default=None,
                   help="Output CSV path (default: auto-named in 10_data/)")
    p.add_argument("--personas-file", default=None,
                   help="Personas JSONL (pilot or prod) for demographic join. "
                        "If omitted, demographic columns are written as empty strings.")
    return p.parse_args()


# ── Loaders ───────────────────────────────────────────────────────────────────
def load_bpns(path: Path) -> dict[str, dict]:
    """Load BPNS JSONL and index by persona_uuid for O(1) lookup.

    Aborts if the same persona_uuid appears more than once — this indicates
    the file mixes outputs from multiple pilot runs and must be deduplicated
    before merging.
    """
    index: dict[str, dict] = {}
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            row = json.loads(line)
            uid = row.get("persona_uuid") or row.get("uuid")
            if not uid:
                continue
            if uid in index:
                log.error(
                    "Duplicate persona_uuid in BPNS file (line %d): %s — "
                    "file likely mixes multiple pilot runs. Deduplicate before merging.",
                    lineno, uid,
                )
                sys.exit(1)
            index[uid] = row
    return index


def load_personas(path: Path | None) -> dict[str, dict]:
    """Load personas JSONL and index by uuid for demographic join.

    Returns an empty dict if path is None (demographics will be empty strings).
    """
    if path is None:
        return {}
    index: dict[str, dict] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            uid = row.get("uuid")
            if not uid:
                continue
            index[uid] = row.get("demographic_information", {})
    log.info("Loaded demographics for %d personas from %s", len(index), path)
    return index


def load_scores(path: Path) -> list[dict]:
    """Load scores JSONL, aborting if the same (uuid, target_words, n_texts, repeat_id, variant_id)
    appears more than once — this indicates unintended duplicate scoring runs."""
    rows: list[dict] = []
    seen: set[tuple] = set()
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            uid          = row.get("persona_uuid") or row.get("uuid")
            target_words = row.get("target_words")
            n_texts      = row.get("n_texts")
            repeat       = row.get("repeat_id")
            variant      = row.get("variant_id")
            key = (uid, target_words, n_texts, repeat, variant)
            if key in seen:
                log.error(
                    "Duplicate score row (line %d): uuid=%s target_words=%s n_texts=%s repeat_id=%s variant_id=%s — "
                    "file likely mixes multiple scoring runs. Deduplicate before merging.",
                    lineno, uid, target_words, n_texts, repeat, variant,
                )
                sys.exit(1)
            seen.add(key)
            rows.append(row)
    return rows


def select_variant(rows: list[dict]) -> list[dict]:
    """If scores contain multiple variant_ids, prompt the user to pick one."""
    variants = sorted({r.get("variant_id") for r in rows if r.get("variant_id")})
    if len(variants) <= 1:
        return rows
    print("\nScores file contains multiple variants:")
    for i, v in enumerate(variants, 1):
        count = sum(1 for r in rows if r.get("variant_id") == v)
        print(f"  {i}. {v}  ({count} rows)")
    while True:
        choice = input("Select variant to merge (number or name): ").strip()
        if choice.isdigit() and 1 <= int(choice) <= len(variants):
            chosen = variants[int(choice) - 1]
            break
        if choice in variants:
            chosen = choice
            break
        print(f"  Invalid choice — enter a number 1–{len(variants)} or the variant name.")
    filtered = [r for r in rows if r.get("variant_id") == chosen]
    log.info("Filtering to variant '%s' (%d rows)", chosen, len(filtered))
    return filtered


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    args = parse_args()

    bpns_path   = Path(args.bpns_file)
    scores_path = Path(args.scores_file)

    if not bpns_path.is_absolute():
        bpns_path = PROJECT_ROOT / bpns_path
    if not scores_path.is_absolute():
        scores_path = PROJECT_ROOT / scores_path

    if not bpns_path.exists():
        log.error("BPNS file not found: %s", bpns_path)
        sys.exit(1)
    if not scores_path.exists():
        log.error("Scores file not found: %s", scores_path)
        sys.exit(1)

    # Auto-name output
    if args.output_file:
        output_path = Path(args.output_file)
        if not output_path.is_absolute():
            output_path = PROJECT_ROOT / output_path
    else:
        ts = datetime.now().strftime("%Y%m%d-%H%M")
        output_path = DATA_DIR / f"70_merge_outputs_{ts}.csv"

    personas_path = Path(args.personas_file) if args.personas_file else None
    if personas_path and not personas_path.is_absolute():
        personas_path = PROJECT_ROOT / personas_path

    log.info("Loading BPNS from %s", bpns_path)
    bpns_index = load_bpns(bpns_path)
    log.info("Loaded %d BPNS records", len(bpns_index))

    log.info("Loading scores from %s", scores_path)
    scores = load_scores(scores_path)
    log.info("Loaded %d score rows", len(scores))
    scores = select_variant(scores)

    personas_index = load_personas(personas_path)
    if not personas_index:
        log.warning("No personas file supplied — demographic columns will be empty.")

    skipped = 0
    written = 0
    demo_missing: set[str] = set()

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()

        for score_row in scores:
            uuid = score_row.get("persona_uuid") or score_row.get("uuid")

            bpns_row = bpns_index.get(uuid)
            if bpns_row is None:
                log.warning("No BPNS record for uuid=%s — skipping", uuid)
                skipped += 1
                continue

            demo = personas_index.get(uuid)
            if demo is None and personas_index:
                demo_missing.add(uuid)
            demo = demo or {}

            output_row = {
                "uuid":             uuid,
                "age":              demo.get("Select Your Age"),
                "gender":           demo.get("Select Your Gender"),
                "education":        demo.get("Select Your Highest Level of Education"),
                "occupation":       demo.get("Provide Your Occupation. (_NA if not applicable_)"),
                "country":          demo.get("Provide Your Country of Residence."),
                "race":             demo.get("Select Your Race"),
                "nationality":      demo.get("Provide Your Nationality."),
                "income_category":  demo.get("Select Your Income Category"),
                "target_words":     score_row.get("target_words"),
                "n_texts":          score_row.get("n_texts"),
                "repeat_id":        score_row.get("repeat_id"),
                "q_aut_1":          bpns_row.get("q_aut_1"),
                "q_aut_2":          bpns_row.get("q_aut_2"),
                "q_aut_3":          bpns_row.get("q_aut_3"),
                "q_aut_4":          bpns_row.get("q_aut_4"),
                "q_com_1":          bpns_row.get("q_com_1"),
                "q_com_2":          bpns_row.get("q_com_2"),
                "q_com_3":          bpns_row.get("q_com_3"),
                "q_com_4":          bpns_row.get("q_com_4"),
                "q_rel_1":          bpns_row.get("q_rel_1"),
                "q_rel_2":          bpns_row.get("q_rel_2"),
                "q_rel_3":          bpns_row.get("q_rel_3"),
                "q_rel_4":          bpns_row.get("q_rel_4"),
                "word_count":       score_row.get("word_count"),
                "llm_aut":          score_row.get("llm_aut"),
                "llm_com":          score_row.get("llm_com"),
                "llm_rel":          score_row.get("llm_rel"),
            }
            writer.writerow(output_row)
            written += 1

    if demo_missing:
        log.warning("Demographics missing for %d uuids: %s", len(demo_missing),
                    ", ".join(sorted(demo_missing)))

    if skipped:
        log.warning("Skipped %d rows due to missing BPNS record", skipped)
    log.info("Done. %d rows written to %s", written, output_path)
    print(f"Output file: {output_path}")


if __name__ == "__main__":
    main()
