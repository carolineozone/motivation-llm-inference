"""
claude_batch_ops.py — All Anthropic Message Batches API logic for the pipeline.

Provides:
  - _add_no_additional_properties() / _make_claude_batch_request() — batch request builders
  - Registry I/O: write_claude_batch_registry, read_claude_batch_registry, load_claude_batch_meta
  - submit_claude_batch()         — submit batch, returns batch_id
  - poll_claude_batch()           — indefinite poll until processing_status == "ended"
  - retrieve_claude_batch_output() — iterate results, write rows, log costs
  - run_claude_scoring_batch()    — full scoring-branch batch flow (N_TEXTS_LEVELS × variants × repeats)
"""

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

import anthropic
from anthropic import Anthropic
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.messages.batch_create_params import Request

from utils import append_cost_log, calculate_cost_anthropic, load_prompt

log = logging.getLogger(__name__)

CLAUDE_BATCH_REGISTRY_FILE = "claude_batch_registry.jsonl"
BATCHES_SUBDIR = "50_batches"

# Locked study design: 3 text-count levels (must match 3×3 factorial)
N_TEXTS_LEVELS: list[int] = [1, 3, 5]


def _batches_dir(data_dir: Path) -> Path:
    """Return (and create if needed) the 50_batches subdirectory."""
    d = data_dir / BATCHES_SUBDIR
    d.mkdir(exist_ok=True)
    return d


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _add_no_additional_properties(schema: dict) -> dict:
    """Recursively add additionalProperties: false to every object in a JSON schema."""
    if not isinstance(schema, dict):
        return schema
    if schema.get("type") == "object":
        schema.setdefault("additionalProperties", False)
    for value in schema.values():
        if isinstance(value, dict):
            _add_no_additional_properties(value)
        elif isinstance(value, list):
            for item in value:
                _add_no_additional_properties(item)
    return schema


def _make_claude_batch_request(
    custom_id: str,
    model_snapshot: str,
    messages: list,
    params: dict,
    schema_class: type,
    system: str | None = None,
) -> Request:
    """Build one Anthropic batch request with structured output schema."""
    schema = _add_no_additional_properties(schema_class.model_json_schema())
    output_config: dict = {
        "format": {
            "type": "json_schema",
            "schema": schema,
        }
    }
    if (effort := params.get("effort")) is not None:
        output_config["effort"] = effort

    kw: dict = {
        "model": model_snapshot,
        "max_tokens": params["max_completion_tokens"],
        "messages": messages,
        "output_config": output_config,
    }
    if system:
        kw["system"] = system
    if params.get("thinking") is not None and params.get("temperature") is not None:
        raise ValueError(
            f"Config error for custom_id={custom_id!r}: "
            "'thinking' and 'temperature' are mutually exclusive in the Anthropic API. "
            "Set exactly one to null in config.yaml."
        )
    if (thinking_type := params.get("thinking")) is not None:
        kw["thinking"] = {"type": thinking_type}
    if (temperature := params.get("temperature")) is not None:
        kw["temperature"] = temperature

    return Request(
        custom_id=custom_id,
        params=MessageCreateParamsNonStreaming(**kw),
    )



# ---------------------------------------------------------------------------
# Registry I/O
# ---------------------------------------------------------------------------

def write_claude_batch_registry(
    data_dir: Path,
    batch_id: str,
    stage: str,
    schema: str,
    output_file: Path,
    submitted_at: str,
    request_meta: dict,
) -> None:
    """Append one entry to claude_batch_registry.jsonl and write the request_meta sidecar."""
    entry = {
        "batch_id":     batch_id,
        "stage":        stage,
        "schema":       schema,
        "output_file":  output_file.name,
        "submitted_at": submitted_at,
    }
    batches = _batches_dir(data_dir)
    with open(batches / CLAUDE_BATCH_REGISTRY_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")
    meta_path = batches / f"batch_{batch_id}_meta.json"
    meta_path.write_text(json.dumps(request_meta, indent=2), encoding="utf-8")
    log.info("Claude registry updated: %s | meta sidecar → %s", batch_id, meta_path.name)


def read_claude_batch_registry(data_dir: Path, batch_id: str) -> dict | None:
    """Return the registry entry for batch_id, or None if not found."""
    registry_path = _batches_dir(data_dir) / CLAUDE_BATCH_REGISTRY_FILE
    if not registry_path.exists():
        return None
    with open(registry_path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                entry = json.loads(line)
                if entry.get("batch_id") == batch_id:
                    return entry
    return None


def load_claude_batch_meta(data_dir: Path, batch_id: str) -> dict:
    """Load the request_meta sidecar for batch_id. Raises FileNotFoundError if missing."""
    path = _batches_dir(data_dir) / f"batch_{batch_id}_meta.json"
    return json.loads(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Core batch operations
# ---------------------------------------------------------------------------

def submit_claude_batch(client: Anthropic, requests: list) -> str:
    """Submit a list of batch requests. Returns batch_id."""
    try:
        batch = client.messages.batches.create(requests=requests)
    except anthropic.AuthenticationError:
        raise SystemExit("ERROR: Anthropic authentication failed — check ANTHROPIC_API_KEY.")
    except anthropic.APIConnectionError as e:
        raise SystemExit(f"ERROR: Could not connect to Anthropic API: {e}")
    except anthropic.APIStatusError as e:
        raise SystemExit(f"ERROR: Anthropic API returned {e.status_code}: {e.message}")
    batch_id = batch.id
    print(f"\n{'='*60}")
    print(f"  CLAUDE BATCH SUBMITTED")
    print(f"  batch_id = {batch_id}")
    print(f"  Save this ID — retrieve results with:")
    print(f"    python 20_scripts/claude_batch_retrieve.py {batch_id}")
    print(f"{'='*60}\n")
    log.info("Claude batch submitted: batch_id=%s", batch_id)
    return batch_id


def poll_claude_batch(client: Anthropic, batch_id: str, poll_interval: int = 30):
    """Poll until processing_status == 'ended'. Returns the final batch object.

    Transient errors (rate limits, connection errors, server errors) are logged
    and retried. Auth errors are re-raised immediately.
    """
    start = time.monotonic()
    while True:
        try:
            batch = client.messages.batches.retrieve(batch_id)
        except anthropic.AuthenticationError:
            raise SystemExit("ERROR: Anthropic authentication failed — check ANTHROPIC_API_KEY.")
        except anthropic.RateLimitError as e:
            log.warning("Rate limited during poll: %s — retrying in %ds", e, poll_interval * 2)
            time.sleep(poll_interval * 2)
            continue
        except anthropic.APIConnectionError as e:
            log.warning("Connection error during poll: %s — retrying in %ds", e, poll_interval)
            time.sleep(poll_interval)
            continue
        except anthropic.APIStatusError as e:
            log.warning("API error %d during poll: %s — retrying in %ds",
                        e.status_code, e.message, poll_interval)
            time.sleep(poll_interval)
            continue

        status = batch.processing_status
        elapsed = time.monotonic() - start
        counts = batch.request_counts
        log.info(
            "Claude batch status: %s (elapsed %.0fs) — processing=%s succeeded=%s errored=%s",
            status, elapsed,
            getattr(counts, "processing", "?"),
            getattr(counts, "succeeded", "?"),
            getattr(counts, "errored", "?"),
        )
        if status == "ended":
            return batch
        time.sleep(poll_interval)


def retrieve_claude_batch_output(
    client: Anthropic,
    batch,
    request_meta: dict,
    schema_class: type,
    pricing: dict,
    output_file: Path,
    batch_id: str,
    data_dir: Path,
    stage: str,
) -> tuple[int, int]:
    """Iterate batch results, parse, write output rows, log costs.
    Returns (written, errors)."""
    batches_dir = _batches_dir(data_dir)
    raw_path = batches_dir / f"batch_{batch_id}_raw_output.jsonl"
    errors = 0
    written = 0

    with open(output_file, "a", encoding="utf-8") as f_out, \
         open(raw_path, "w", encoding="utf-8") as f_raw:
        try:
            results_iter = client.messages.batches.results(batch_id)
        except anthropic.AuthenticationError:
            raise SystemExit("ERROR: Anthropic authentication failed — check ANTHROPIC_API_KEY.")
        except anthropic.APIConnectionError as e:
            raise SystemExit(f"ERROR: Could not connect to Anthropic API to fetch results: {e}")
        except anthropic.APIStatusError as e:
            raise SystemExit(f"ERROR: Anthropic API returned {e.status_code} fetching results: {e.message}")

        for result in results_iter:
            # Save raw result for debugging
            f_raw.write(result.model_dump_json() + "\n")

            custom_id = result.custom_id
            meta = request_meta.get(custom_id, {})

            if result.result.type != "succeeded":
                log.error(
                    "Non-success result for custom_id=%s: type=%s",
                    custom_id, result.result.type,
                )
                errors += 1
                continue

            try:
                message = result.result.message
                text = next(
                    (b.text for b in message.content if b.type == "text"), None
                )
                if text is None:
                    log.error(
                        "No text block for custom_id=%s stop_reason=%s — writing row with null scores",
                        custom_id, message.stop_reason,
                    )
                    usage = message.usage
                    cost = calculate_cost_anthropic(usage, pricing, meta["model_key"], mode="batch")
                    timestamp = datetime.now(timezone.utc).isoformat()
                    append_cost_log(
                        data_dir=data_dir,
                        log_timestamp=timestamp,
                        stage=stage,
                        variant_id=meta["variant_id"],
                        model_key=meta["model_key"],
                        model_snapshot=meta["model_snapshot"],
                        pricing_mode="batch",
                        persona_uuid=meta["persona_uuid"],
                        cost=cost,
                        output_file=output_file,
                        request_id=message.id,
                        batch_id=batch_id,
                    )
                    row = {
                        "timestamp":        timestamp,
                        "stage":            stage,
                        "variant_id":       meta["variant_id"],
                        "model_key":        meta["model_key"],
                        "model_snapshot":   meta["model_snapshot"],
                        "prompt_developer": meta["prompt_developer"],
                        "prompt_user":      meta["prompt_user"],
                        "persona_uuid":     meta["persona_uuid"],
                        "target_words":     meta["target_words"],
                        "n_texts":          meta["n_texts"],
                        "word_count":       meta.get("word_count"),
                        "source_file":      meta["source_file"],
                        "repeat_id":        meta["repeat_id"],
                        "stop_reason":      message.stop_reason,
                        "input_tokens":     cost["input_tokens"],
                        "visible_output":   cost["output_tokens"] - cost["thinking_tokens"],
                        "reasoning_output": cost["thinking_tokens"],
                        "thinking_text":    None,
                        "cost_usd":         cost["cost_usd"],
                        **{field: None for field in schema_class.model_fields},
                    }
                    f_out.write(json.dumps(row, ensure_ascii=False) + "\n")
                    f_out.flush()
                    written += 1
                    errors += 1
                    continue

                # thinking block is present only when thinking is enabled (e.g. sonnet+adaptive)
                thinking_text = next(
                    (b.thinking for b in message.content if b.type == "thinking"), None
                )

                parsed = schema_class.model_validate_json(text)
                usage = message.usage
                cost = calculate_cost_anthropic(usage, pricing, meta["model_key"], mode="batch")

                timestamp = datetime.now(timezone.utc).isoformat()
                append_cost_log(
                    data_dir=data_dir,
                    log_timestamp=timestamp,
                    stage=stage,
                    variant_id=meta["variant_id"],
                    model_key=meta["model_key"],
                    model_snapshot=meta["model_snapshot"],
                    pricing_mode="batch",
                    persona_uuid=meta["persona_uuid"],
                    cost=cost,
                    output_file=output_file,
                    request_id=message.id,
                    batch_id=batch_id,
                )

                row = {
                    "timestamp":        timestamp,
                    "stage":            stage,
                    "variant_id":       meta["variant_id"],
                    "model_key":        meta["model_key"],
                    "model_snapshot":   meta["model_snapshot"],
                    "prompt_developer": meta["prompt_developer"],
                    "prompt_user":      meta["prompt_user"],
                    "persona_uuid":     meta["persona_uuid"],
                    "target_words":     meta["target_words"],
                    "n_texts":          meta["n_texts"],
                    "word_count":       meta.get("word_count"),
                    "source_file":      meta["source_file"],
                    "repeat_id":        meta["repeat_id"],
                    "stop_reason":      message.stop_reason,
                    "input_tokens":     cost["input_tokens"],
                    "visible_output":   cost["output_tokens"] - cost["thinking_tokens"],
                    "reasoning_output": cost["thinking_tokens"],  # thinking token count (0 for haiku)
                    "thinking_text":    thinking_text,            # thinking block text (None for haiku); distinct from schema "reasoning" field
                    "cost_usd":         cost["cost_usd"],
                    **parsed.model_dump(),  # includes "reasoning": str from schema
                }
                f_out.write(json.dumps(row, ensure_ascii=False) + "\n")
                f_out.flush()
                written += 1

            except Exception as e:
                log.error("Parse error for custom_id=%s: %s", custom_id, e)
                errors += 1

    log.info(
        "Claude batch output written: %d rows, %d errors → %s",
        written, errors, output_file,
    )
    log.info("Raw batch output saved → %s", raw_path.name)
    return written, errors


# ---------------------------------------------------------------------------
# Top-level runner
# ---------------------------------------------------------------------------

def run_claude_scoring_batch(
    client: Anthropic,
    grouped: dict,
    source_file: str,
    stage: str,
    variants: dict,
    defaults: dict,
    pricing: dict,
    schema_class: type,
    output_file: Path,
    poll_interval: int,
    no_wait: bool,
    data_dir: Path,
    prompts_dir: Path,
) -> None:
    """Build, submit (and optionally poll+retrieve) a Claude batch for the scoring branch.

    grouped is keyed by (persona_uuid, target_words); each pool is capped at max(N_TEXTS_LEVELS).
    Builds requests for every (pool, n_texts, variant, repeat) combination.
    """
    n_repeats = defaults.get("n_repeats", 1)
    min_pool = max(N_TEXTS_LEVELS)
    requests: list[Request] = []
    request_meta: dict[str, dict] = {}

    for (uuid, target_words), pool in grouped.items():
        if len(pool) < min_pool:
            log.warning(
                "SKIP: %s | target_words=%s | pool=%d < %d",
                uuid[:8], target_words, len(pool), min_pool,
            )
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
                model_snapshot = (
                    pricing[model_key].get("snapshot") or pricing[model_key].get("version")
                )
                prompt_usr_name = params["prompt_user"]
                prompt_dev_name = params.get("prompt_developer") or None

                raw_usr = load_prompt(prompt_usr_name, prompts_dir)
                usr_prompt = raw_usr.replace("{APPRECIATION_TEXTS}", texts_xml)
                sys_prompt = load_prompt(prompt_dev_name, prompts_dir) if prompt_dev_name else None

                messages = [{"role": "user", "content": usr_prompt}]

                for repeat_id in range(1, n_repeats + 1):
                    custom_id = f"scoring-{uuid[:8]}-tw{target_words}-n{n_texts}-{v_id}-r{repeat_id}"
                    assert len(custom_id) <= 64, f"custom_id too long ({len(custom_id)}): {custom_id}"
                    req = _make_claude_batch_request(
                        custom_id, model_snapshot, messages, params, schema_class,
                        system=sys_prompt,
                    )
                    requests.append(req)
                    request_meta[custom_id] = {
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
                    }

    log.info("Built %d Claude batch requests.", len(requests))
    submitted_at = datetime.now(timezone.utc).isoformat()
    batch_id = submit_claude_batch(client, requests)
    write_claude_batch_registry(
        data_dir, batch_id, stage, schema_class.__name__,
        output_file, submitted_at, request_meta,
    )

    if no_wait:
        log.info("Exiting after submission. batch_id=%s", batch_id)
        return

    batch = poll_claude_batch(client, batch_id, poll_interval)
    retrieve_claude_batch_output(
        client, batch, request_meta, schema_class,
        pricing, output_file, batch_id, data_dir, stage,
    )
