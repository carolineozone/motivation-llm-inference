"""
claude_batch_stage.py — Batch (async) scoring runner for Anthropic/Claude models.

Submits a scoring stage to the Anthropic Message Batches API. Results are retrieved
separately using claude_batch_retrieve.py. For synchronous execution use claude_run_stage.py.
For OpenAI batch use batch_stage.py.

Groups input by (persona_uuid, target_words) and builds one request per
(group, n_texts, variant, repeat) — implementing the locked 3×3 factorial design.

Usage:
    python claude_batch_stage.py --stage 60_pilot_scoring_claude --input-file <name> -n 20 --dry-run
    python claude_batch_stage.py --stage 60_pilot_scoring_claude --input-file <name> -n 20
"""

import argparse
import json
import logging
import os
import sys
from collections import defaultdict
from pathlib import Path

import yaml
from anthropic import Anthropic
from dotenv import load_dotenv

# --- Paths ---
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_DIR = PROJECT_ROOT / "10_data"
PROMPTS_DIR = PROJECT_ROOT / "30_prompts"
CONFIG_FILE = PROJECT_ROOT / "config.yaml"
PRICING_FILE = PROJECT_ROOT / "pricing.yaml"

sys.path.insert(0, str(SCRIPT_DIR))
from claude_batch_ops import N_TEXTS_LEVELS, run_claude_scoring_batch  # noqa: E402
from utils import load_prompt, load_schema, unique_output_path  # noqa: E402

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# --- CLI ---
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Submit a scoring stage to the Anthropic Message Batches API."
    )
    p.add_argument("--stage", required=True, help="Stage key in config.yaml > stages")
    p.add_argument("input_file", nargs="?", default=None, metavar="INPUT_FILE",
                   help="JSONL filename in 10_data/ (positional shorthand for --input-file)")
    p.add_argument("--input-file", dest="input_file_flag", default=None,
                   help="JSONL filename in 10_data/ (overrides input_data from config)")
    p.add_argument("-n", "--num-personas", type=int, default=None,
                   help="Max number of personas to process (default: all)")
    p.add_argument("--dry-run", action="store_true",
                   help="Print batch cost estimate + prompt preview without submitting.")
    p.add_argument("--poll-interval", type=int, default=30,
                   help="Seconds between status checks (default: 30)")
    return p.parse_args()


# --- Config helpers ---
def load_config() -> dict:
    return yaml.safe_load(CONFIG_FILE.read_text(encoding="utf-8"))


def load_pricing() -> dict:
    return yaml.safe_load(PRICING_FILE.read_text(encoding="utf-8"))


def _get_snapshot(pricing: dict, model_key: str) -> str:
    """Read pinned model string; handles both 'snapshot' and legacy 'version' key."""
    return pricing[model_key].get("snapshot") or pricing[model_key].get("version")


# --- Main ---
def main() -> None:
    args = parse_args()

    config = load_config()
    pricing = load_pricing()

    if args.stage not in config["stages"]:
        raise SystemExit(f"Stage {args.stage!r} not found in config.yaml > stages.")

    stage_cfg = config["stages"][args.stage]
    if "input_data" not in stage_cfg:
        raise SystemExit(
            f"Stage {args.stage!r} has no 'input_data' field. "
            "claude_batch_stage.py only handles scoring stages."
        )

    defaults = stage_cfg.get("defaults", {})
    variants = stage_cfg["variants"]
    log.info("Stage %r — %d variant(s): %s", args.stage, len(variants), list(variants.keys()))

    schema_class = load_schema(stage_cfg["schema"], PROMPTS_DIR / "schemas.py")

    # Load and group input data by (persona_uuid, target_words) — same as claude_run_stage.py
    if args.input_file_flag is None and args.input_file is None:
        raise SystemExit(
            "ERROR: --input-file is required for scoring stages. "
            "The config default does not include the target_words field. "
            "Pass a post-2026-03-23 generation file, e.g. --input-file 40_pilot_textgeneration_v2_..._n50.jsonl"
        )
    input_path = DATA_DIR / (args.input_file_flag or args.input_file or stage_cfg["input_data"])
    source_file = input_path.name
    grouped: dict[tuple, list] = defaultdict(list)
    with open(input_path, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            grouped[(row["persona_uuid"], row["target_words"])].append(row)

    if args.num_personas is not None:
        seen_uuids: list[str] = []
        kept: dict[tuple, list] = {}
        for key, rows in grouped.items():
            uuid = key[0]
            if uuid not in seen_uuids:
                if len(seen_uuids) >= args.num_personas:
                    continue
                seen_uuids.append(uuid)
            kept[key] = rows
        grouped = kept

    n_unique = len({k[0] for k in grouped})
    log.info("Loaded %d group(s) for %d persona(s) from %s", len(grouped), n_unique, source_file)

    min_pool = max(N_TEXTS_LEVELS)
    n_repeats = defaults.get("n_repeats", 1)
    n_valid = sum(1 for pool in grouped.values() if len(pool) >= min_pool)

    if args.dry_run:
        total_calls_per_variant = n_valid * len(N_TEXTS_LEVELS) * n_repeats

        first_valid_pool = next(
            (pool[:min_pool] for pool in grouped.values() if len(pool) >= min_pool), []
        )
        sample_texts_xml = "\n".join(
            f'<text id="{i+1}">{r["text"]}</text>' for i, r in enumerate(first_valid_pool)
        )
        cost_lines = [
            f"\n--- DRY-RUN COST ESTIMATE (stage={args.stage}, mode=batch) ---",
            f"Valid groups (pool >= {min_pool}): {n_valid} / {len(grouped)}",
        ]
        for v_id, v_overrides in variants.items():
            params = {**defaults, **(v_overrides or {})}
            model_key = params["model_key"]
            model_snapshot = _get_snapshot(pricing, model_key)
            raw_usr = load_prompt(params["prompt_user"], PROMPTS_DIR)
            sample_prompt = raw_usr.replace("{APPRECIATION_TEXTS}", sample_texts_xml)
            est_in_tok = len(sample_prompt) // 4
            est_out_tok = params["max_completion_tokens"]
            est_calls = total_calls_per_variant
            price = pricing.get(model_key, {})
            in_price = price.get("batchinput", 0)
            out_price = price.get("batchoutput", 0)
            est_usd = est_calls * (est_in_tok * in_price + est_out_tok * out_price) / 1_000_000
            cost_lines.append(
                f"  {v_id:12s} | {model_snapshot:40s} | calls={est_calls:4d} | "
                f"~{est_in_tok:,} in_tok | ~{est_out_tok:,} out_tok/call | ~${est_usd:.4f}"
            )

        # Prompt preview for first valid group
        first_uuid, first_tw = next(
            (k for k, pool in grouped.items() if len(pool) >= min_pool),
            next(iter(grouped)),
        )
        first_pool = grouped[(first_uuid, first_tw)][:min_pool]
        for v_id, v_overrides in variants.items():
            params = {**defaults, **(v_overrides or {})}
            prompt_dev_name = params.get("prompt_developer") or None
            raw_usr = load_prompt(params["prompt_user"], PROMPTS_DIR)
            texts_xml = "\n".join(
                f'<text id="{i+1}">{r["text"]}</text>' for i, r in enumerate(first_pool)
            )
            usr_p = raw_usr.replace("{APPRECIATION_TEXTS}", texts_xml)
            sys_p = load_prompt(prompt_dev_name, PROMPTS_DIR) if prompt_dev_name else None
            print(f"\n--- PROMPT PREVIEW (variant={v_id}, uuid={first_uuid[:8]}, "
                  f"tw={first_tw}, n={min_pool}) ---")
            if sys_p:
                print(f"[SYSTEM]\n{sys_p}")
            print(f"[USER]\n{usr_p}")

        # Sample output row
        v_id, v_overrides = next(iter(variants.items()))
        params = {**defaults, **(v_overrides or {})}
        sample_row = {
            "timestamp":        "<from API>",
            "stage":            args.stage,
            "variant_id":       v_id,
            "model_key":        params["model_key"],
            "model_snapshot":   _get_snapshot(pricing, params["model_key"]),
            "prompt_developer": params.get("prompt_developer"),
            "prompt_user":      params["prompt_user"],
            "persona_uuid":     first_uuid,
            "target_words":     first_tw,
            "n_texts":          max(N_TEXTS_LEVELS),
            "word_count":       "<computed>",
            "source_file":      source_file,
            "repeat_id":        1,
            "stop_reason":      "<from API>",
            "input_tokens":     "<from API>",
            "visible_output":   "<from API>",
            "reasoning_output": "<from API>",
            "thinking_text":    "<from API (None if thinking disabled)>",
            "cost_usd":         "<from API>",
            **{field: "<from API>" for field in schema_class.model_fields},
        }
        expected_output = unique_output_path(DATA_DIR, args.stage,n_unique)
        print(f"\n--- OUTPUT ROW PREVIEW (schema={stage_cfg['schema']}, output={expected_output}) ---")
        print(json.dumps(sample_row, indent=2))
        print("\n".join(cost_lines))
        print("\n" + "=" * 60 + "\nDry run complete. Remove --dry-run to submit.")
        return

    output_file = unique_output_path(DATA_DIR, args.stage,n_unique)
    log.info("Output → %s (reserved for claude_batch_retrieve)", output_file)
    print(f"\nOutput file (reserved): {output_file}\n")

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise SystemExit("ERROR: ANTHROPIC_API_KEY environment variable not set.")
    client = Anthropic(api_key=api_key)
    run_claude_scoring_batch(
        client=client,
        grouped=grouped,
        source_file=source_file,
        stage=args.stage,
        variants=variants,
        defaults=defaults,
        pricing=pricing,
        schema_class=schema_class,
        output_file=output_file,
        poll_interval=args.poll_interval,
        no_wait=True,
        data_dir=DATA_DIR,
        prompts_dir=PROMPTS_DIR,
    )


if __name__ == "__main__":
    main()
