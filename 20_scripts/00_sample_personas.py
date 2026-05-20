# %% [markdown]
# # 00_sample_personas.py
# Draws exactly 1000 personas from SCOPE-Persona that pass inclusion filters.
# Saves uuid, demographic_information, and professional_identity_and_career to JSONL.
# Saves excluded personas with exclusion reason to a separate JSONL.
# Prints and saves a full exclusion report for the thesis Method section.
#
# Note: Target is 1000 to create a reserve pool.
# The first 500 passing the coworker filter (in 02_filter_coworker.py)
# will be used for analysis.

# %% Imports
import json
import pandas as pd
from datasets import load_dataset

DATASET_ID = "Salesforce/SCOPE-Persona"
CONFIG = "scope_qa"
SEED = 42
BUFFER = 5000
TARGET_N = 1000
OUTPUT_FILE = "../10_data/00_personas.jsonl"
EXCLUDED_FILE = "../10_data/00_personas_excluded.jsonl"
REPORT_FILE = f"../10_data/00_personas_report_{pd.Timestamp.now().strftime('%Y%m%d')}.txt"


# %% Define filter
def passes_filters(record: dict) -> tuple[bool, str]:
    """
    Returns (True, 'included') or (False, reason_for_exclusion).

    Inclusion criteria:
      - Age: not in ("Less than 20", "65 or older", "")
      - Occupation answer value: not in ("NA", "N/A", "")
      - ALL professional_identity_and_career answers: not in ("NA", "N/A", "")
        (except 'Do you enjoy your work?' which is a numeric rating)
    """
    demo = record.get("demographic_information") or {}

    # Age
    age_raw = str(demo.get("Select Your Age") or "").strip()
    if age_raw in ("Less than 20", "65 or older", ""):
        return False, "age"

    # Occupation
    occupation = str(demo.get("Provide Your Occupation. (_NA if not applicable_)") or "").strip()
    if occupation.upper() in ("NA", "N/A", ""):
        return False, "occupation_NA"

    # Professional identity — ALL answers must be non-NA
    prof = record.get("professional_identity_and_career") or {}
    for q, a in prof.items():
        if "enjoy your work" in q.lower():
            continue
        if str(a).strip().upper() in ("NA", "N/A", ""):
            return False, "prof_incomplete"

    return True, "included"


print("✓ Filter defined")
print("  - Exclude: age in ('Less than 20', '65 or older', '')")
print("  - Exclude: occupation answer value in ('NA', 'N/A', '')")
print("  - Exclude: any professional_identity answer in ('NA', 'N/A', '')")

# %% Stream, filter, write incrementally
print(f"\nStreaming dataset to collect {TARGET_N} qualifying personas …\n")

ds = load_dataset(DATASET_ID, CONFIG, split="train", streaming=True)
ds = ds.shuffle(buffer_size=BUFFER, seed=SEED)

collected = 0
scanned = 0
exclusion_counts = {
    "age": 0,
    "occupation_NA": 0,
    "prof_incomplete": 0,
}

with (
    open(OUTPUT_FILE, "w", encoding="utf-8") as f_out,
    open(EXCLUDED_FILE, "w", encoding="utf-8") as f_excl,
):
    for record in ds:
        scanned += 1
        passed, reason = passes_filters(record)

        if passed:
            row = {
                "uuid": record["uuid"],
                "demographic_information": record.get("demographic_information", {}),
                "professional_identity_and_career": record.get("professional_identity_and_career", {}),
            }
            f_out.write(json.dumps(row) + "\n")
            collected += 1
            if collected % 100 == 0:
                print(f"  … {collected} collected (scanned {scanned})", flush=True)
        else:
            excl_row = {
                "uuid": record["uuid"],
                "exclusion_reason": reason,
                "demographic_information": record.get("demographic_information", {}),
                "professional_identity_and_career": record.get("professional_identity_and_career", {}),
            }
            f_excl.write(json.dumps(excl_row) + "\n")
            exclusion_counts[reason] += 1

        if collected >= TARGET_N:
            break

# %% Exclusion report — print
excluded_total = scanned - collected

print(f"\n{'=' * 60}")
print("  SAMPLING REPORT")
print(f"{'=' * 60}")
print(f"  Total scanned              : {scanned:,}")
print(f"  Collected (included)       : {collected:,}")
print(f"  Excluded (total)           : {excluded_total:,}  ({100 * excluded_total / scanned:.1f}%)")
print("")
print("  Exclusion breakdown:")
print(f"    Age out of range         : {exclusion_counts['age']:,}  ({100 * exclusion_counts['age'] / scanned:.1f}%)") # noqa: E501
print(
    f"    Occupation = NA          : {exclusion_counts['occupation_NA']:,}  ({100 * exclusion_counts['occupation_NA'] / scanned:.1f}%)" # noqa: E501 
)
print(
    f"    Prof. identity incomplete: {exclusion_counts['prof_incomplete']:,}  ({100 * exclusion_counts['prof_incomplete'] / scanned:.1f}%)" # noqa: E501
)
print(f"{'=' * 60}")

# %% Save sampling report for thesis Method section
with open(REPORT_FILE, "w", encoding="utf-8") as f:
    f.write("============================================================\n")
    f.write("SAMPLING REPORT: SCOPE-Persona Dataset\n")
    f.write(f"Date/Time: {pd.Timestamp.now()}\n")
    f.write("============================================================\n\n")

    f.write(f"Target N (Included)      : {collected}\n")
    f.write(f"Total Records Scanned    : {scanned}\n")
    f.write(f"Exclusion Rate           : {100 * excluded_total / scanned:.2f}%\n\n")

    f.write("INCLUSION CRITERIA:\n")
    f.write("  - Age: answer value not in ('Less than 20', '65 or older', '')\n")
    f.write("  - Occupation: answer value not in ('NA', 'N/A', '')\n")
    f.write("  - Prof. Identity: all professional_identity_and_career answer values\n")
    f.write("    not in ('NA', 'N/A', ''), except 'Do you enjoy your work?' (numeric)\n\n")

    f.write("EXCLUSION BREAKDOWN:\n")
    f.write(f"  - Age out of range      : {exclusion_counts['age']} ({100 * exclusion_counts['age'] / scanned:.2f}%)\n")
    f.write(
        f"  - Occupation = NA       : {exclusion_counts['occupation_NA']} ({100 * exclusion_counts['occupation_NA'] / scanned:.2f}%)\n" # noqa: E501
    )
    f.write(
        f"  - Incomplete Prof. ID   : {exclusion_counts['prof_incomplete']} ({100 * exclusion_counts['prof_incomplete'] / scanned:.2f}%)\n\n" # noqa: E501
    )

    f.write("SAMPLING PARAMETERS:\n")
    f.write(f"  - Dataset ID            : {DATASET_ID}\n")
    f.write(f"  - Config                : {CONFIG}\n")
    f.write(f"  - Seed                  : {SEED}\n")
    f.write(f"  - Shuffle Buffer        : {BUFFER}\n")
    f.write(f"  - Output file           : {OUTPUT_FILE}\n")
    f.write(f"  - Excluded file         : {EXCLUDED_FILE}\n")
    f.write("============================================================\n")

print(f"✓ Method section report saved to: {REPORT_FILE}")

# %% Quick verification
print(f"\n✓ Included personas saved to: {OUTPUT_FILE}")
print(f"✓ Excluded personas saved to: {EXCLUDED_FILE}")
print("\nFirst 3 included uuids:")

count = 0
with open(OUTPUT_FILE, encoding="utf-8") as f:
    for line in f:
        if count < 3:
            obj = json.loads(line)
            print(f"  {obj['uuid']}")
        count += 1
print(f"\n  Total included lines : {count}")

excl_count = 0
with open(EXCLUDED_FILE, encoding="utf-8") as f:
    for line in f:
        excl_count += 1
print(f"  Total excluded lines : {excl_count}")
