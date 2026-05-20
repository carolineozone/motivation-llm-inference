"""
claude_batch_retrieve.py — Retrieve results for a submitted Anthropic Message Batch.

Re-attaches to a submitted batch by ID, polls until complete, writes output JSONL,
and prints the inspect table. Reads registry from 10_data/50_batches/claude_batch_registry.jsonl.

Usage:
    python 20_scripts/claude_batch_retrieve.py <batch_id>
    python 20_scripts/claude_batch_retrieve.py <batch_id> --poll-interval 60
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import yaml
from anthropic import Anthropic
from dotenv import load_dotenv

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_DIR = PROJECT_ROOT / "10_data"
PROMPTS_DIR = PROJECT_ROOT / "30_prompts"
PRICING_FILE = PROJECT_ROOT / "pricing.yaml"

sys.path.insert(0, str(SCRIPT_DIR))
from claude_batch_ops import (  # noqa: E402
    BATCHES_SUBDIR, CLAUDE_BATCH_REGISTRY_FILE,
    load_claude_batch_meta, poll_claude_batch,
    read_claude_batch_registry, retrieve_claude_batch_output,
)
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
    p = argparse.ArgumentParser(
        description="Retrieve results for a submitted Anthropic Message Batch."
    )
    p.add_argument("batch_id", help="Anthropic batch ID (e.g. msgbatch_abc123)")
    p.add_argument("--poll-interval", type=int, default=30,
                   help="Seconds between status checks (default: 30)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    batch_id = args.batch_id

    entry = read_claude_batch_registry(DATA_DIR, batch_id)
    if not entry:
        raise SystemExit(
            f"batch_id {batch_id!r} not found in "
            f"{DATA_DIR / BATCHES_SUBDIR / CLAUDE_BATCH_REGISTRY_FILE}"
        )

    log.info("Found registry entry: stage=%s schema=%s output=%s",
             entry["stage"], entry["schema"], entry["output_file"])

    pricing = yaml.safe_load(PRICING_FILE.read_text(encoding="utf-8"))
    schema_class = load_schema(entry["schema"], PROMPTS_DIR / "schemas.py")
    request_meta = load_claude_batch_meta(DATA_DIR, batch_id)
    output_file = DATA_DIR / entry["output_file"]

    if output_file.exists():
        raise SystemExit(
            f"Output file already exists: {output_file}\n"
            "Retrieval already completed. Delete or rename the file if you want to re-retrieve."
        )

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise SystemExit("ERROR: ANTHROPIC_API_KEY environment variable not set.")
    client = Anthropic(api_key=api_key)

    batch = poll_claude_batch(client, batch_id, args.poll_interval)
    written, errors = retrieve_claude_batch_output(
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
