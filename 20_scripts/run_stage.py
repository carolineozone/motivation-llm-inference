"""
run_stage.py — Standard (synchronous) pipeline stage runner.

Drives any stage defined in config.yaml against a personas JSONL file
using live API calls. For async batch execution use batch_stage.py.

Usage:
    python run_stage.py --stage 40_pilot_textgeneration -n 20 --dry-run
    python run_stage.py --stage 40_pilot_textgeneration -n 20
    python run_stage.py --stage 40_pilot_textgeneration --personas-file custom.jsonl -n 20
"""

import argparse
import json
import logging
import os
import sys
from collections import defaultdict
from datetime import datetime, timezone
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
from inspect_output import inspect_file  # noqa: E402
from utils import (  # noqa: E402
    BUILDERS, append_cost_log, build_api_params, build_text_only,
    calculate_cost, check_reasoning_token_budget, fill_prompt, load_prompt, load_schema,
    print_cost_estimate, print_scoring_cost_estimate,
    unique_output_path, write_output_rows,
)

load_dotenv()

logging.basicConfig(
    level=logging.INFO,  # set to DEBUG for more verbose output
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# --- CLI ---
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run a pipeline stage (standard mode).")
    p.add_argument("--stage", required=True, help="Stage key in config.yaml > stages")
    p.add_argument("--personas-file", default=None,
                   help="JSONL filename in 10_data/ (overrides stage config persona_data)")
    p.add_argument("--input-file", default=None,
                   help="JSONL filename in 10_data/ (overrides input_data from config; scoring stages only)")
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


# --- Scoring standard mode ---
def run_scoring_standard(
    client: OpenAI,
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

    with open(output_file, "a", encoding="utf-8") as f_out:
        for (uuid, source_variant_id), persona_rows in grouped.items():
            texts_xml = "\n".join(
                f'<text id="{row["coworker_id"]}">{row["text"]}</text>'
                for row in persona_rows
            )
            n_texts_scored = len(persona_rows)

            for v_id, overrides in variants.items():
                params = {**defaults, **(overrides or {})}
                model_key = params["model_key"]
                model_snapshot = pricing[model_key]["snapshot"]
                prompt_usr_name = params["prompt_user"]
                prompt_dev_name = params.get("prompt_developer") or None

                raw_usr = load_prompt(prompt_usr_name, PROMPTS_DIR)
                usr_prompt = raw_usr.replace("{APPRECIATION_TEXTS}", texts_xml)
                sys_prompt = load_prompt(prompt_dev_name, PROMPTS_DIR) if prompt_dev_name else None

                messages = (
                    [{"role": "system", "content": sys_prompt}] if sys_prompt else []
                ) + [{"role": "user", "content": usr_prompt}]

                api_params = build_api_params(model_snapshot, messages, schema_class, params)

                for repeat_id in range(1, n_repeats + 1):
                    try:
                        timestamp = datetime.now(timezone.utc).isoformat()
                        response = client.beta.chat.completions.parse(**api_params)
                        cost = calculate_cost(response.usage, pricing, model_key, mode="standard")
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
                        result = response.choices[0].message.parsed
                        row = {
                            "timestamp":         timestamp,
                            "stage":             stage,
                            "variant_id":        v_id,
                            "model_key":         model_key,
                            "model_snapshot":    model_snapshot,
                            "prompt_developer":  prompt_dev_name,
                            "prompt_user":       prompt_usr_name,
                            "persona_uuid":      uuid,
                            "source_variant_id": source_variant_id,
                            "source_file":       source_file,
                            "n_texts_scored":    n_texts_scored,
                            "repeat_id":         repeat_id,
                            **cost,
                            **result.model_dump(),
                        }
                        f_out.write(json.dumps(row, ensure_ascii=False) + "\n")
                        f_out.flush()
                        log.info("DONE: %s | %s | repeat=%d | $%.6f",
                                 uuid[:8], v_id, repeat_id, cost["cost_usd"])
                    except Exception as e:
                        log.error("FAIL: %s | %s | repeat=%d | %s", uuid[:8], v_id, repeat_id, e)

    log.info("Scoring standard mode complete. Output → %s", output_file)
    print(f"\n--- OUTPUT INSPECTION: {output_file.name} ---\n")
    inspect_file(output_file)


# --- Standard mode ---
def run_standard(
    client: OpenAI,
    personas: list,
    stage: str,
    stage_cfg: dict,
    variants: dict,
    defaults: dict,
    pricing: dict,
    schema_class: type,
    output_file: Path,
) -> None:
    _build = BUILDERS[stage_cfg.get("conversation_format", "json_dump")]

    with open(output_file, "a", encoding="utf-8") as f_out:
        for persona in personas:
            uuid = persona["uuid"]
            for v_id, overrides in variants.items():
                params = {**defaults, **(overrides or {})}
                model_key = params["model_key"]
                model_snapshot = pricing[model_key]["snapshot"]
                prompt_dev_name = params["prompt_developer"]
                prompt_usr_name = params["prompt_user"]

                raw_sys = load_prompt(prompt_dev_name, PROMPTS_DIR)
                raw_usr = load_prompt(prompt_usr_name, PROMPTS_DIR)
                sys_prompt = fill_prompt(raw_sys, persona, params)
                usr_prompt = fill_prompt(raw_usr, persona, params)
                messages = _build(persona, sys_prompt, usr_prompt)

                api_params = build_api_params(model_snapshot, messages, schema_class, params)

                if log.isEnabledFor(logging.DEBUG):
                    skip = {"messages", "response_format"}
                    debug_params = {k: v for k, v in api_params.items() if k not in skip}
                    debug_params["response_format"] = api_params["response_format"].__name__
                    log.debug("messages uuid=%s variant=%s\n%s",
                              uuid[:8], v_id,
                              "\n".join(f"[{m['role'].upper()}]\n{m['content']}" for m in messages))
                    log.debug("api_params %s", json.dumps(debug_params, indent=2))

                try:
                    timestamp = datetime.now(timezone.utc).isoformat()
                    response = client.beta.chat.completions.parse(**api_params)
                    finish_reason = response.choices[0].finish_reason
                    if finish_reason == "length":
                        details = getattr(response.usage, "completion_tokens_details", None)
                        reasoning_tok = getattr(details, "reasoning_tokens", 0) if details else 0
                        visible_tok = response.usage.completion_tokens - reasoning_tok
                        log.warning(
                            "TRUNCATED (finish_reason=length) | persona=%s variant=%s | "
                            "reasoning=%d visible=%d total=%d / limit=%d — increase max_completion_tokens",
                            uuid[:8], v_id,
                            reasoning_tok, visible_tok,
                            response.usage.completion_tokens,
                            params["max_completion_tokens"],
                        )
                    cost = calculate_cost(response.usage, pricing, model_key, mode="standard")
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
                    result = response.choices[0].message.parsed
                    base = {
                        "timestamp":        timestamp,
                        "stage":            stage,
                        "variant_id":       v_id,
                        "model_key":        model_key,
                        "model_snapshot":   model_snapshot,
                        "prompt_developer": prompt_dev_name,
                        "prompt_user":      prompt_usr_name,
                        "persona_uuid":     uuid,
                        "target_words":     params.get("target_words"),
                        **cost,
                    }
                    n = write_output_rows(f_out, base, result, cost)
                    f_out.flush()
                    log.info("DONE: %s | %s | %d row(s) | $%.6f", uuid[:8], v_id, n, cost["cost_usd"])

                except Exception as e:
                    log.error("FAIL: %s | %s | %s", uuid[:8], v_id, e)

    log.info("Standard mode complete. Output → %s", output_file)
    print(f"\n--- OUTPUT INSPECTION: {output_file.name} ---\n")
    inspect_file(output_file)


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
            print_scoring_cost_estimate(grouped, variants, defaults, pricing, mode="standard")
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
            print("\n" + "="*60 + "\nDry run complete. Remove --dry-run to execute.")
            return

        output_file = unique_output_path(DATA_DIR, args.stage,n_unique)
        log.info("Output → %s", output_file)
        print(f"\nOutput file: {output_file}\n")
        client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
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
        print_cost_estimate(personas, variants, defaults, pricing, stage_cfg, mode="standard")
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
        print("\n" + "="*60 + "\nDry run complete. Remove --dry-run to execute.")
        return

    output_file = unique_output_path(DATA_DIR, args.stage,len(personas))
    log.info("Output → %s", output_file)
    print(f"\nOutput file: {output_file}\n")
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    run_standard(
        client=client,
        personas=personas,
        stage=args.stage,
        stage_cfg=stage_cfg,
        variants=variants,
        defaults=defaults,
        pricing=pricing,
        schema_class=schema_class,
        output_file=output_file,
    )


if __name__ == "__main__":
    main()
