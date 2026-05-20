"""
claude_run_stage.py — Standard (synchronous) scoring runner for Anthropic/Claude models.

Mirrors the scoring branch of run_stage.py but uses the Anthropic Messages API
with messages.parse() for structured output; groups by (persona_uuid, target_words),
scores 3 subsets (n=1/3/5 texts). Only handles scoring stages (input_data).
For OpenAI scoring use run_stage.py. For async batch use claude_batch_stage.py.

Usage:
    python claude_run_stage.py --stage 60_pilot_scoring_claude --input-file <name> -n 20 --dry-run
    python claude_run_stage.py --stage 60_pilot_scoring_claude --input-file <name> -n 20
"""

import argparse
import json
import logging
import os
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import anthropic
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
from claude_batch_ops import _add_no_additional_properties  # noqa: E402
from inspect_output import inspect_file  # noqa: E402
from utils import (  # noqa: E402
    append_cost_log, calculate_cost_anthropic, load_prompt, load_schema,
    unique_output_path,
)

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# Locked study design: 3 text-count levels (must match 3×3 factorial)
N_TEXTS_LEVELS: list[int] = [1, 3, 5]
MAX_RETRIES: int = 3  # Max retry attempts for transient API errors (rate limit, connection, 5xx)


# --- CLI ---
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run a scoring stage (Anthropic standard mode).")
    p.add_argument("--stage", required=True, help="Stage key in config.yaml > stages")
    p.add_argument("input_file", nargs="?", default=None, metavar="INPUT_FILE",
                   help="JSONL filename in 10_data/ (positional shorthand for --input-file)")
    p.add_argument("--input-file", dest="input_file_flag", default=None,
                   help="JSONL filename in 10_data/ (overrides input_data from config)")
    p.add_argument("-n", "--num-personas", type=int, default=None,
                   help="Max number of personas to process (default: all)")
    p.add_argument("--dry-run", action="store_true",
                   help="Print cost estimate + prompt preview without making API calls.")
    return p.parse_args()


# --- Config helpers ---
def load_config() -> dict:
    return yaml.safe_load(CONFIG_FILE.read_text(encoding="utf-8"))


def load_pricing() -> dict:
    return yaml.safe_load(PRICING_FILE.read_text(encoding="utf-8"))


# --- Anthropic helpers ---
def _get_snapshot(pricing: dict, model_key: str) -> str:
    """Read pinned model string; handles both 'snapshot' and legacy 'version' key."""
    return pricing[model_key].get("snapshot") or pricing[model_key].get("version")


# --- Scoring standard mode (Anthropic) ---
def run_scoring_standard(
    client: Anthropic,
    grouped: dict,
    source_file: str,
    stage: str,
    variants: dict,
    defaults: dict,
    pricing: dict,
    schema_class: type,
    output_file: Path,
) -> None:
    n_repeats = defaults.get("n_repeats", 1)
    min_pool = max(N_TEXTS_LEVELS)

    with open(output_file, "a", encoding="utf-8") as f_out:
        for (uuid, target_words), pool in grouped.items():
            if len(pool) < min_pool:
                log.warning("SKIP: %s | target_words=%s | pool=%d < %d",
                            uuid[:8], target_words, len(pool), min_pool)
                continue
            pool = pool[:min_pool]

            for n_texts in N_TEXTS_LEVELS:
                texts_subset = pool[:n_texts]
                word_count = sum(r.get("word_count", 0) for r in texts_subset)
                texts_xml = "\n".join(
                    f'<text id="{i+1}">{r["text"]}</text>'
                    for i, r in enumerate(texts_subset)
                )

                for v_id, overrides in variants.items():
                    params = {**defaults, **(overrides or {})}
                    model_key = params["model_key"]
                    model_snapshot = _get_snapshot(pricing, model_key)
                    prompt_usr_name = params["prompt_user"]
                    prompt_dev_name = params.get("prompt_developer") or None

                    raw_usr = load_prompt(prompt_usr_name, PROMPTS_DIR)
                    usr_prompt = raw_usr.replace("{APPRECIATION_TEXTS}", texts_xml)
                    sys_prompt = load_prompt(prompt_dev_name, PROMPTS_DIR) if prompt_dev_name else None

                    output_config: dict = {
                        "format": {
                            "type": "json_schema",
                            "schema": _add_no_additional_properties(schema_class.model_json_schema()),
                        }
                    }
                    if (effort := params.get("effort")) is not None:
                        output_config["effort"] = effort

                    parse_kwargs: dict = {
                        "model":         model_snapshot,
                        "max_tokens":    params["max_completion_tokens"],
                        "messages":      [{"role": "user", "content": usr_prompt}],
                        "output_format": schema_class,
                        "output_config": output_config,
                    }
                    if sys_prompt:
                        parse_kwargs["system"] = sys_prompt
                    if params.get("thinking") is not None and params.get("temperature") is not None:
                        raise ValueError(
                            f"Config error for stage={stage!r} variant={v_id!r}: "
                            "'thinking' and 'temperature' are mutually exclusive in the Anthropic API. "
                            "Set exactly one to null in config.yaml."
                        )
                    if (thinking_type := params.get("thinking")) is not None:
                        parse_kwargs["thinking"] = {"type": thinking_type}
                    for key in ("temperature", "top_p", "top_k", "stop_sequences"):
                        if params.get(key) is not None:
                            parse_kwargs[key] = params[key]

                    for repeat_id in range(1, n_repeats + 1):
                        for attempt in range(MAX_RETRIES + 1):
                            try:
                                timestamp = datetime.now(timezone.utc).isoformat()
                                response = client.messages.parse(**parse_kwargs)

                                if response.stop_reason == "refusal":
                                    log.warning("REFUSAL: %s | tw=%s | n=%d | %s | r=%d",
                                                uuid[:8], target_words, n_texts, v_id, repeat_id)
                                    break

                                result = response.parsed_output
                                if result is None:
                                    log.error("PARSE FAILED: %s | tw=%s | n=%d | %s | r=%d | stop=%s",
                                              uuid[:8], target_words, n_texts, v_id, repeat_id,
                                              response.stop_reason)
                                    break

                                thinking_text = next(
                                    (b.thinking for b in response.content if b.type == "thinking"), None
                                )
                                cost = calculate_cost_anthropic(
                                    response.usage, pricing, model_key, mode="standard"
                                )
                                append_cost_log(
                                    data_dir=output_file.parent,
                                    log_timestamp=timestamp,
                                    stage=stage,
                                    variant_id=v_id,
                                    model_key=model_key,
                                    model_snapshot=model_snapshot,
                                    pricing_mode="standard",
                                    persona_uuid=uuid,
                                    cost=cost,
                                    output_file=output_file,
                                    request_id=response.id,
                                )
                                row = {
                                    "timestamp":        timestamp,
                                    "stage":            stage,
                                    "variant_id":       v_id,
                                    "model_key":        model_key,
                                    "model_snapshot":   model_snapshot,
                                    "prompt_developer": prompt_dev_name,
                                    "prompt_user":      prompt_usr_name,
                                    "persona_uuid":     uuid,
                                    "target_words":     target_words,
                                    "n_texts":          n_texts,
                                    "word_count":       word_count,
                                    "source_file":      source_file,
                                    "repeat_id":        repeat_id,
                                    "stop_reason":      response.stop_reason,
                                    "input_tokens":     cost["input_tokens"],
                                    "visible_output":   cost["output_tokens"] - cost["thinking_tokens"],
                                    "reasoning_output": cost["thinking_tokens"],
                                    "thinking_text":    thinking_text,
                                    "cost_usd":         cost["cost_usd"],
                                    **result.model_dump(),
                                }
                                f_out.write(json.dumps(row, ensure_ascii=False) + "\n")
                                f_out.flush()
                                log.info("DONE: %s | tw=%s | n=%d | %s | r=%d | $%.6f",
                                         uuid[:8], target_words, n_texts, v_id, repeat_id,
                                         cost["cost_usd"])
                                break  # success

                            except anthropic.AuthenticationError:
                                raise
                            except anthropic.BadRequestError as e:
                                log.error("BAD REQUEST: %s | tw=%s | n=%d | %s | r=%d | %s",
                                          uuid[:8], target_words, n_texts, v_id, repeat_id, e.message)
                                break  # non-retriable
                            except anthropic.RateLimitError as e:
                                if attempt < MAX_RETRIES:
                                    delay = 60 * (2 ** attempt)
                                    log.warning("RATE LIMIT: %s | tw=%s | n=%d | %s | r=%d | "
                                                "retrying in %ds (%d/%d)",
                                                uuid[:8], target_words, n_texts, v_id, repeat_id,
                                                delay, attempt + 1, MAX_RETRIES)
                                    time.sleep(delay)
                                else:
                                    log.error("RATE LIMIT: %s | tw=%s | n=%d | %s | r=%d | "
                                              "giving up after %d attempts",
                                              uuid[:8], target_words, n_texts, v_id, repeat_id,
                                              MAX_RETRIES)
                            except anthropic.APIStatusError as e:
                                if e.status_code >= 500 and attempt < MAX_RETRIES:
                                    delay = 30 * (2 ** attempt)
                                    log.warning("API ERROR %d: %s | tw=%s | n=%d | %s | r=%d | "
                                                "retrying in %ds (%d/%d)",
                                                e.status_code, uuid[:8], target_words, n_texts,
                                                v_id, repeat_id, delay, attempt + 1, MAX_RETRIES)
                                    time.sleep(delay)
                                else:
                                    log.error("API ERROR %d: %s | tw=%s | n=%d | %s | r=%d | %s",
                                              e.status_code, uuid[:8], target_words, n_texts,
                                              v_id, repeat_id, e.message)
                                    break  # non-retriable 4xx
                            except anthropic.APIConnectionError as e:
                                if attempt < MAX_RETRIES:
                                    delay = 30 * (2 ** attempt)
                                    log.warning("CONNECTION ERROR: %s | tw=%s | n=%d | %s | r=%d | "
                                                "retrying in %ds (%d/%d)",
                                                uuid[:8], target_words, n_texts, v_id, repeat_id,
                                                delay, attempt + 1, MAX_RETRIES)
                                    time.sleep(delay)
                                else:
                                    log.error("CONNECTION ERROR: %s | tw=%s | n=%d | %s | r=%d | "
                                              "giving up after %d attempts",
                                              uuid[:8], target_words, n_texts, v_id, repeat_id,
                                              MAX_RETRIES)

    log.info("Scoring standard mode complete. Output → %s", output_file)
    print(f"\n--- OUTPUT INSPECTION: {output_file.name} ---\n")
    inspect_file(output_file)


# --- Config validation ---
def _validate_stage_config(stage: str, variants: dict, defaults: dict) -> None:
    """Fail fast with a clear message if config.yaml variants are malformed."""
    for v_id, overrides in variants.items():
        if not isinstance(overrides, (dict, type(None))):
            raise SystemExit(
                f"Variant {v_id!r} in stage {stage!r} has an invalid value: {overrides!r}\n"
                "Variant values must be a dict or null. "
                "Check for indentation errors in config.yaml — "
                "defaults keys may have leaked into variants."
            )
        params = {**defaults, **(overrides or {})}
        for key in ("model_key", "prompt_user"):
            if not params.get(key):
                raise SystemExit(
                    f"Variant {v_id!r} in stage {stage!r} is missing required key: {key!r}\n"
                    "Check defaults and variant overrides in config.yaml."
                )


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
            "claude_run_stage.py only handles scoring stages."
        )

    defaults = stage_cfg.get("defaults", {})
    variants = stage_cfg["variants"]
    _validate_stage_config(args.stage, variants, defaults)
    log.info("Stage %r — %d variant(s): %s", args.stage, len(variants), list(variants.keys()))

    schema_class = load_schema(stage_cfg["schema"], PROMPTS_DIR / "schemas.py")

    # Load and group input data by (persona_uuid, target_words)
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
        # Cost estimate
        print(f"\n--- DRY-RUN COST ESTIMATE (stage={args.stage}) ---")
        print(f"Valid groups (pool ≥ {min_pool}): {n_valid} / {len(grouped)}")
        total_calls_per_variant = n_valid * len(N_TEXTS_LEVELS) * n_repeats
        # Use the largest subset of the first valid group for token estimation
        first_valid_pool = next(
            (pool[:min_pool] for pool in grouped.values() if len(pool) >= min_pool), []
        )
        sample_texts_xml = "\n".join(
            f'<text id="{i+1}">{r["text"]}</text>' for i, r in enumerate(first_valid_pool)
        )
        for v_id, v_overrides in variants.items():
            params = {**defaults, **(v_overrides or {})}
            model_key = params["model_key"]
            model_snapshot = _get_snapshot(pricing, model_key)
            raw_usr = load_prompt(params["prompt_user"], PROMPTS_DIR)
            sample_prompt = raw_usr.replace("{APPRECIATION_TEXTS}", sample_texts_xml)
            # Rough token estimate: 1 token ≈ 4 chars
            est_in_tok = len(sample_prompt) // 4
            est_out_tok = params["max_completion_tokens"]
            est_calls = total_calls_per_variant
            price = pricing.get(model_key, {})
            in_price = price.get("input", 0)
            out_price = price.get("output", 0)
            est_usd = est_calls * (est_in_tok * in_price + est_out_tok * out_price) / 1_000_000
            print(f"  {v_id:12s} | {model_snapshot:40s} | calls={est_calls:4d} | "
                  f"~{est_in_tok:,} in_tok | ~{est_out_tok:,} out_tok/call | ~${est_usd:.4f}")

        # Prompt preview for first valid group
        first_uuid, first_tw = next(
            (k for k, pool in grouped.items() if len(pool) >= min_pool),
            (next(iter(grouped)), None)
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
        print("\n" + "=" * 60 + "\nDry run complete. Remove --dry-run to execute.")
        return

    output_file = unique_output_path(DATA_DIR, args.stage,n_unique)
    log.info("Output → %s", output_file)
    print(f"\nOutput file: {output_file}\n")
    client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    run_scoring_standard(
        client=client,
        grouped=grouped,
        source_file=source_file,
        stage=args.stage,
        variants=variants,
        defaults=defaults,
        pricing=pricing,
        schema_class=schema_class,
        output_file=output_file,
    )


if __name__ == "__main__":
    main()
