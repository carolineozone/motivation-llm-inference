"""
batch_stage.py — Batch (async) pipeline stage runner.

Submits a stage to the OpenAI Batch API. Results are retrieved separately
using batch_retrieve.py. For synchronous execution use run_stage.py.

Usage:
    python batch_stage.py --stage 40_pilot_textgeneration -n 20 --dry-run
    python batch_stage.py --stage 40_pilot_textgeneration -n 20
    python batch_stage.py --stage 40_pilot_textgeneration --personas-file custom.jsonl -n 20
"""

import argparse
import json
import logging
import os
import sys
from collections import defaultdict
from pathlib import Path

import yaml
from dotenv import load_dotenv
from openai import OpenAI

# --- Paths ---
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_DIR = PROJECT_ROOT / "10_data"
PROMPTS_DIR = PROJECT_ROOT / "30_prompts"
CONFIG_FILE = PROJECT_ROOT / "config.yaml"
PRICING_FILE = PROJECT_ROOT / "pricing.yaml"

sys.path.insert(0, str(SCRIPT_DIR))
from batch_ops import run_batch, run_scoring_batch  # noqa: E402
from utils import (  # noqa: E402
    build_text_only, check_reasoning_token_budget, fill_prompt, load_prompt, load_schema,
    print_cost_estimate, print_scoring_cost_estimate,
    unique_output_path,
)

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# --- CLI ---
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Submit a pipeline stage to the OpenAI Batch API.")
    p.add_argument("--stage", required=True, help="Stage key in config.yaml > stages")
    p.add_argument("--personas-file", default=None,
                   help="JSONL filename in 10_data/ (overrides stage config persona_data)")
    p.add_argument("--input-file", default=None,
                   help="JSONL filename in 10_data/ (overrides input_data from config; scoring stages only)")
    p.add_argument("-n", "--num-personas", type=int, default=None,
                   help="Max number of personas to process (default: all)")
    p.add_argument("--dry-run", action="store_true",
                   help="Print batch cost estimate + prompt preview without submitting.")
    p.add_argument("--poll-interval", type=int, default=30,
                   help="Seconds between status checks if waiting inline (default: 30)")
    return p.parse_args()


# --- Config helpers ---
def load_config() -> dict:
    return yaml.safe_load(CONFIG_FILE.read_text(encoding="utf-8"))


def load_pricing() -> dict:
    return yaml.safe_load(PRICING_FILE.read_text(encoding="utf-8"))


# --- Main ---
def main() -> None:
    args = parse_args()

    config = load_config()
    pricing = load_pricing()

    if args.stage not in config["stages"]:
        raise SystemExit(f"Stage {args.stage!r} not found in config.yaml > stages.")

    stage_cfg = config["stages"][args.stage]
    defaults = stage_cfg.get("defaults", {})
    variants = stage_cfg["variants"]
    log.info("Stage %r — %d variant(s): %s", args.stage, len(variants), list(variants.keys()))

    schema_class = load_schema(stage_cfg["schema"], PROMPTS_DIR / "schemas.py")

    # ── Scoring branch (input_data) ──────────────────────────────────────────
    if "input_data" in stage_cfg:
        input_path = DATA_DIR / (args.input_file or stage_cfg["input_data"])
        source_file = input_path.name
        grouped: dict[tuple, list] = defaultdict(list)
        with open(input_path, encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                grouped[(row["persona_uuid"], row["variant_id"])].append(row)
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

        if args.dry_run:
            print_scoring_cost_estimate(grouped, variants, defaults, pricing, mode="batch")
            first_uuid = next(iter(grouped))[0]
            for (uuid, source_variant_id), persona_rows in grouped.items():
                if uuid != first_uuid:
                    continue
                texts_xml = "\n".join(
                    f'<text id="{r["coworker_id"]}">{r["text"]}</text>' for r in persona_rows
                )
                for v_id, v_overrides in variants.items():
                    params = {**defaults, **(v_overrides or {})}
                    prompt_dev_name = params.get("prompt_developer") or None
                    raw_usr = load_prompt(params["prompt_user"], PROMPTS_DIR)
                    usr_p = raw_usr.replace("{APPRECIATION_TEXTS}", texts_xml)
                    sys_p = load_prompt(prompt_dev_name, PROMPTS_DIR) if prompt_dev_name else None
                    msgs = build_text_only(None, sys_p, usr_p)
                    print(f"--- PROMPT PREVIEW (stage={args.stage}, variant={v_id}, "
                          f"uuid={uuid[:8]}, source_variant={source_variant_id}) ---")
                    for m in msgs:
                        print(f"[{m['role'].upper()}]\n{m['content']}")
            v_id, v_overrides = next(iter(variants.items()))
            params = {**defaults, **(v_overrides or {})}
            prompt_dev_name = params.get("prompt_developer") or None
            sample_rows = next(iter(grouped.values()))
            sample_source_variant = next(iter(grouped))[1]
            model_key = params["model_key"]
            sample_row = {
                "timestamp":         "<from API>",
                "stage":             args.stage,
                "variant_id":        v_id,
                "model_key":         model_key,
                "model_snapshot":    pricing[model_key]["snapshot"],
                "prompt_developer":  prompt_dev_name,
                "prompt_user":       params["prompt_user"],
                "persona_uuid":      first_uuid,
                "source_variant_id": sample_source_variant,
                "source_file":       source_file,
                "n_texts_scored":    len(sample_rows),
                "repeat_id":         1,
                **{field: "<from API>" for field in schema_class.model_fields},
                "cost_usd":          "<from API>",
            }
            expected_output = unique_output_path(DATA_DIR, args.stage,n_unique)
            print(f"\n--- OUTPUT ROW PREVIEW (schema={stage_cfg['schema']}, output={expected_output}) ---")
            print(json.dumps(sample_row, indent=2))
            print("\n" + "="*60 + "\nDry run complete. Remove --dry-run to submit.")
            return

        output_file = unique_output_path(DATA_DIR, args.stage,n_unique)
        log.info("Output → %s (reserved for batch_retrieve)", output_file)
        client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        run_scoring_batch(
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
            load_prompt=load_prompt,
        )
        return

    # ── Persona branch (persona_data) ────────────────────────────────────────
    personas_file = args.personas_file or stage_cfg["persona_data"]
    personas_path = DATA_DIR / personas_file
    personas = []
    with open(personas_path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if args.num_personas is not None and i >= args.num_personas:
                break
            personas.append(json.loads(line))
    log.info("Loaded %d personas from %s", len(personas), personas_path.name)
    log.info("  %d top-level fields: %s", len(personas[0]), list(personas[0].keys()))
    log.info("  Showing first%s:", " and last" if len(personas) > 1 else "")
    p0 = personas[0]
    demo = p0.get("demographic_information", {})
    log.info(
        "  [0] uuid=%-36s | age=%-5s | gender=%-10s | occupation=%s",
        p0["uuid"],
        demo.get("Select Your Age", "?"),
        demo.get("Select Your Gender", "?"),
        demo.get("Provide Your Occupation. (_NA if not applicable_)", "?"),
    )
    if len(personas) > 1:
        p1 = personas[-1]
        demo1 = p1.get("demographic_information", {})
        log.info(
            "  [%d] uuid=%-36s | age=%-5s | gender=%-10s | occupation=%s",
            len(personas) - 1,
            p1["uuid"],
            demo1.get("Select Your Age", "?"),
            demo1.get("Select Your Gender", "?"),
            demo1.get("Provide Your Occupation. (_NA if not applicable_)", "?"),
        )

    for v_id, overrides in variants.items():
        check_reasoning_token_budget({**defaults, **(overrides or {})}, variant_label=v_id)

    if args.dry_run:
        print_cost_estimate(personas, variants, defaults, pricing, stage_cfg, mode="batch")
        from utils import BUILDERS
        _build = BUILDERS[stage_cfg.get("conversation_format", "json_dump")]
        v_id, v_overrides = next(iter(variants.items()))
        params = {**defaults, **(v_overrides or {})}
        raw_sys = load_prompt(params["prompt_developer"], PROMPTS_DIR)
        raw_usr = load_prompt(params["prompt_user"], PROMPTS_DIR)
        sys_p = fill_prompt(raw_sys, personas[0], params)
        usr_p = fill_prompt(raw_usr, personas[0], params)
        msgs = _build(personas[0], sys_p, usr_p)
        print(f"--- PROMPT PREVIEW (stage={args.stage}, variant={v_id}, uuid={personas[0]['uuid'][:8]}) ---")
        for m in msgs:
            preview = m['content'].replace('\n', ' ')[:50]
            print(f"[{m['role'].upper()}] \"{preview}...\"")
        model_key = params["model_key"]
        sample_row = {
            "timestamp":        "<from API>",
            "stage":            args.stage,
            "variant_id":       v_id,
            "model_key":        model_key,
            "model_snapshot":   pricing[model_key]["snapshot"],
            "prompt_developer": params["prompt_developer"],
            "prompt_user":      params["prompt_user"],
            "persona_uuid":     personas[0]["uuid"],
            "target_words":     params.get("target_words"),
            **{field: "<from API>" for field in schema_class.model_fields},
            "cost_usd":         "<from API>",
        }
        expected_output = unique_output_path(DATA_DIR, args.stage,len(personas))
        print(f"\n--- OUTPUT ROW PREVIEW (schema={stage_cfg['schema']}, output={expected_output}) ---")
        print(json.dumps(sample_row, indent=2))
        print("\n" + "="*60 + "\nDry run complete. Remove --dry-run to submit.")
        return

    output_file = unique_output_path(DATA_DIR, args.stage,len(personas))
    log.info("Output → %s (reserved for batch_retrieve)", output_file)
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    run_batch(
        client=client,
        personas=personas,
        stage=args.stage,
        stage_cfg=stage_cfg,
        variants=variants,
        defaults=defaults,
        pricing=pricing,
        schema_class=schema_class,
        output_file=output_file,
        poll_interval=args.poll_interval,
        no_wait=True,
        data_dir=DATA_DIR,
        prompts_dir=PROMPTS_DIR,
        fill_prompt=fill_prompt,
        load_prompt=load_prompt,
    )


if __name__ == "__main__":
    main()
