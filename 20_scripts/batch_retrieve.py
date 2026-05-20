"""
batch_retrieve.py — Re-attach to a submitted OpenAI batch and retrieve results.

Usage:
    python 20_scripts/batch_retrieve.py <batch_id>
    python 20_scripts/batch_retrieve.py <batch_id> --poll-interval 60

Looks up the batch in 10_data/batch_registry.jsonl, polls until complete,
writes the output JSONL, and prints the inspect table.
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import yaml
from dotenv import load_dotenv
from openai import OpenAI

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_DIR = PROJECT_ROOT / "10_data"
PROMPTS_DIR = PROJECT_ROOT / "30_prompts"
PRICING_FILE = PROJECT_ROOT / "pricing.yaml"

sys.path.insert(0, str(SCRIPT_DIR))
from batch_ops import BATCHES_SUBDIR, load_batch_meta, poll_batch, read_batch_registry, retrieve_batch_output  # noqa: E402
from inspect_output import inspect_file  # noqa: E402
from utils import load_schema  # noqa: E402

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Retrieve results for a submitted OpenAI batch.")
    p.add_argument("batch_id", help="OpenAI batch ID (e.g. batch_abc123)")
    p.add_argument("--poll-interval", type=int, default=30,
                   help="Seconds between status checks (default: 30)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    batch_id = args.batch_id

    # Look up registry
    entry = read_batch_registry(DATA_DIR, batch_id)
    if not entry:
        raise SystemExit(f"batch_id {batch_id!r} not found in {DATA_DIR / BATCHES_SUBDIR / 'batch_registry.jsonl'}")

    log.info("Found registry entry: stage=%s schema=%s output=%s",
             entry["stage"], entry["schema"], entry["output_file"])

    # Load dependencies
    pricing = yaml.safe_load(PRICING_FILE.read_text(encoding="utf-8"))
    schema_class = load_schema(entry["schema"], PROMPTS_DIR / "schemas.py")
    request_meta = load_batch_meta(DATA_DIR, batch_id)
    output_file = DATA_DIR / entry["output_file"]

    if output_file.exists():
        raise SystemExit(
            f"Output file already exists: {output_file}\n"
            "Retrieval already completed. Delete or rename the file if you want to re-retrieve."
        )

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    # Poll + retrieve
    batch = poll_batch(client, batch_id, args.poll_interval)
    written, errors = retrieve_batch_output(
        client=client,
        batch=batch,
        request_meta=request_meta,
        schema_class=schema_class,
        pricing=pricing,
        output_file=output_file,
        batch_id=batch_id,
        data_dir=DATA_DIR,
        stage=entry["stage"],
    )

    if written > 0:
        print(f"\n--- OUTPUT INSPECTION: {output_file.name} ---\n")
        inspect_file(output_file)
    else:
        log.warning("No rows written (written=%d, errors=%d).", written, errors)


if __name__ == "__main__":
    main()
