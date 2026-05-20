"""
batch_ops.py — All OpenAI Batch API logic for the pipeline.

Provides:
  - _parse_usage()         — convert raw batch response usage dict to object
  - _add_no_additional_properties() / _make_batch_request() — batch request builders
  - Registry I/O           — write_batch_registry, read_batch_registry, load_batch_meta
  - submit_batch()         — upload + submit, returns batch_id
  - poll_batch()           — indefinite poll until terminal state
  - retrieve_batch_output() — download results, write rows, log costs
  - run_batch()            — full persona-branch batch flow
  - run_scoring_batch()    — full scoring-branch batch flow
"""

import io
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

from openai import OpenAI

from utils import append_cost_log, calculate_cost, write_output_rows

log = logging.getLogger(__name__)

BATCH_REGISTRY_FILE = "batch_registry.jsonl"
BATCHES_SUBDIR = "50_batches"


def _batches_dir(data_dir: Path) -> Path:
    """Return (and create if needed) the 50_batches subdirectory."""
    d = data_dir / BATCHES_SUBDIR
    d.mkdir(exist_ok=True)
    return d


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_usage(usage_raw: dict) -> SimpleNamespace:
    """Convert a raw batch-response usage dict to a usage-like object for calculate_cost."""
    details_raw = usage_raw.get("completion_tokens_details") or {}
    details = SimpleNamespace(reasoning_tokens=details_raw.get("reasoning_tokens", 0))
    return SimpleNamespace(
        prompt_tokens=usage_raw.get("prompt_tokens", 0),
        completion_tokens=usage_raw.get("completion_tokens", 0),
        completion_tokens_details=details if details_raw else None,
    )


def _add_no_additional_properties(schema: dict) -> dict:
    """Recursively add additionalProperties: false to every object in a JSON schema.
    Required for OpenAI batch API strict mode."""
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


def _make_batch_request(custom_id: str, model_snapshot: str, messages: list,
                        params: dict, schema_class: type) -> dict:
    schema = _add_no_additional_properties(schema_class.model_json_schema())
    body: dict = {
        "model": model_snapshot,
        "messages": messages,
        "max_completion_tokens": params["max_completion_tokens"],
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": schema_class.__name__,
                "schema": schema,
                "strict": True,
            },
        },
    }
    if params.get("reasoning_effort") is not None:
        body["reasoning_effort"] = params["reasoning_effort"]
    if params.get("temperature") is not None:
        body["temperature"] = params["temperature"]
    return {
        "custom_id": custom_id,
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": body,
    }


# ---------------------------------------------------------------------------
# Registry I/O
# ---------------------------------------------------------------------------

def write_batch_registry(
    data_dir: Path,
    batch_id: str,
    stage: str,
    schema: str,
    output_file: Path,
    submitted_at: str,
    request_meta: dict,
) -> None:
    """Append one entry to batch_registry.jsonl and write the request_meta sidecar."""
    entry = {
        "batch_id":     batch_id,
        "stage":        stage,
        "schema":       schema,
        "output_file":  output_file.name,
        "submitted_at": submitted_at,
    }
    batches = _batches_dir(data_dir)
    with open(batches / BATCH_REGISTRY_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")
    meta_path = batches / f"batch_{batch_id}_meta.json"
    meta_path.write_text(json.dumps(request_meta, indent=2), encoding="utf-8")
    log.info("Registry updated: %s | meta sidecar → %s", batch_id, meta_path.name)


def read_batch_registry(data_dir: Path, batch_id: str) -> dict | None:
    """Return the registry entry for batch_id, or None if not found."""
    registry_path = _batches_dir(data_dir) / BATCH_REGISTRY_FILE
    if not registry_path.exists():
        return None
    with open(registry_path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                entry = json.loads(line)
                if entry.get("batch_id") == batch_id:
                    return entry
    return None


def load_batch_meta(data_dir: Path, batch_id: str) -> dict:
    """Load the request_meta sidecar for batch_id. Raises FileNotFoundError if missing."""
    path = _batches_dir(data_dir) / f"batch_{batch_id}_meta.json"
    return json.loads(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Core batch operations
# ---------------------------------------------------------------------------

def submit_batch(client: OpenAI, requests: list) -> str:
    """Upload requests JSONL and submit a batch. Returns batch_id."""
    batch_jsonl = "\n".join(json.dumps(r) for r in requests).encode("utf-8")
    file_obj = client.files.create(
        file=("batch_input.jsonl", io.BytesIO(batch_jsonl), "application/jsonl"),
        purpose="batch",
    )
    log.info("Uploaded input file: %s", file_obj.id)

    batch = client.batches.create(
        input_file_id=file_obj.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
    )
    batch_id = batch.id
    print(f"\n{'='*60}")
    print(f"  BATCH SUBMITTED")
    print(f"  batch_id = {batch_id}")
    print(f"  Save this ID — retrieve results with:")
    print(f"    python 20_scripts/batch_retrieve.py {batch_id}")
    print(f"{'='*60}\n")
    log.info("Batch submitted: batch_id=%s", batch_id)
    return batch_id


def poll_batch(client: OpenAI, batch_id: str, poll_interval: int = 30):
    """Poll until the batch reaches a terminal state. No timeout — polls indefinitely.
    Returns the final batch object."""
    start = time.monotonic()
    while True:
        batch = client.batches.retrieve(batch_id)
        status = batch.status
        elapsed = time.monotonic() - start
        log.info("Batch status: %s (elapsed %.0fs)", status, elapsed)
        if status == "completed":
            return batch
        if status in ("failed", "expired", "cancelled"):
            log.error("Batch ended with status=%s. batch_id=%s", status, batch_id)
            if batch.errors and batch.errors.data:
                for err in batch.errors.data:
                    log.error("  line=%s code=%s message=%s", err.line, err.code, err.message)
            return batch
        time.sleep(poll_interval)


def retrieve_batch_output(
    client: OpenAI,
    batch,
    request_meta: dict,
    schema_class: type,
    pricing: dict,
    output_file: Path,
    batch_id: str,
    data_dir: Path,
    stage: str,
) -> tuple[int, int]:
    """Download batch output, parse results, write rows, log costs.
    Returns (written, errors)."""
    if batch.status != "completed":
        log.error("Cannot retrieve output: batch status=%s", batch.status)
        return 0, 0

    if not batch.output_file_id:
        log.error("Batch completed but output_file_id is None. batch_id=%s", batch_id)
        if batch.error_file_id:
            raw_errors = client.files.content(batch.error_file_id).text
            for line in raw_errors.splitlines():
                if line.strip():
                    err = json.loads(line)
                    log.error("  custom_id=%s  error=%s", err.get("custom_id"), err.get("error"))
        return 0, 0

    raw_output = client.files.content(batch.output_file_id).text
    raw_path = _batches_dir(data_dir) / f"batch_{batch_id}_raw_output.jsonl"
    raw_path.write_text(raw_output, encoding="utf-8")
    log.info("Raw batch output saved → %s", raw_path.name)
    errors = 0
    written = 0
    truncation_count = 0
    total_processed = 0

    with open(output_file, "a", encoding="utf-8") as f_out:
        for line in raw_output.splitlines():
            if not line.strip():
                continue
            resp = json.loads(line)
            total_processed += 1
            custom_id = resp["custom_id"]
            meta = request_meta.get(custom_id, {})

            # Check for individual request errors (failed requests have resp["error"] set)
            if resp.get("error"):
                log.error("API error for custom_id=%s: %s", custom_id, resp["error"])
                errors += 1
                continue

            finish_reason = (
                resp.get("response", {})
                    .get("body", {})
                    .get("choices", [{}])[0]
                    .get("finish_reason")
            )
            if finish_reason == "length":
                truncation_count += 1
                usage_raw = resp["response"]["body"].get("usage", {})
                details = usage_raw.get("completion_tokens_details") or {}
                reasoning_tok = details.get("reasoning_tokens", 0)
                visible_tok = usage_raw.get("completion_tokens", 0) - reasoning_tok
                log.warning(
                    "TRUNCATED (finish_reason=length) | custom_id=%s | "
                    "reasoning=%d visible=%d total=%d — increase max_completion_tokens",
                    custom_id, reasoning_tok, visible_tok,
                    usage_raw.get("completion_tokens", 0),
                )

            try:
                content = resp["response"]["body"]["choices"][0]["message"]["content"]
                result = schema_class.model_validate_json(content)
                usage = _parse_usage(resp["response"]["body"].get("usage", {}))
                cost = calculate_cost(usage, pricing, meta["model_key"], mode="batch")
                created_unix = resp.get("response", {}).get("body", {}).get("created")
                timestamp = (datetime.fromtimestamp(created_unix, tz=timezone.utc).isoformat()
                             if created_unix else "")
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
                    request_id=resp["response"]["body"].get("id"),
                    batch_id=batch_id,
                )
                base = {"timestamp": timestamp, **meta, **cost}
                n = write_output_rows(f_out, base, result, cost)
                written += n
            except Exception as e:
                log.error("Parse error for custom_id=%s: %s", custom_id, e)
                errors += 1

        f_out.flush()

    if truncation_count:
        log.warning(
            "BATCH TRUNCATION SUMMARY: %d / %d responses hit max_completion_tokens limit. "
            "Increase max_completion_tokens for the affected stage variant.",
            truncation_count, total_processed,
        )
    log.info("Batch output written: %d rows, %d errors → %s", written, errors, output_file)
    return written, errors


# ---------------------------------------------------------------------------
# Top-level runners
# ---------------------------------------------------------------------------

def run_batch(
    client: OpenAI,
    personas: list,
    stage: str,
    stage_cfg: dict,
    variants: dict,
    defaults: dict,
    pricing: dict,
    schema_class: type,
    output_file: Path,
    poll_interval: int,
    no_wait: bool,
    data_dir: Path,
    prompts_dir: Path,
    fill_prompt,
    load_prompt,
) -> None:
    """Build, submit (and optionally poll+retrieve) a batch for the persona branch."""
    from utils import BUILDERS

    _build = BUILDERS[stage_cfg.get("conversation_format", "json_dump")]
    requests = []
    request_meta: dict[str, dict] = {}

    for persona in personas:
        uuid = persona["uuid"]
        for v_id, overrides in variants.items():
            params = {**defaults, **(overrides or {})}
            model_key = params["model_key"]
            model_snapshot = pricing[model_key]["snapshot"]
            prompt_dev_name = params["prompt_developer"]
            prompt_usr_name = params["prompt_user"]

            raw_sys = load_prompt(prompt_dev_name, prompts_dir)
            raw_usr = load_prompt(prompt_usr_name, prompts_dir)
            sys_prompt = fill_prompt(raw_sys, persona, params)
            usr_prompt = fill_prompt(raw_usr, persona, params)
            messages = _build(persona, sys_prompt, usr_prompt)

            custom_id = f"persona-{uuid}-variant-{v_id}"
            req = _make_batch_request(custom_id, model_snapshot, messages, params, schema_class)
            requests.append(req)
            request_meta[custom_id] = {
                "stage":            stage,
                "variant_id":       v_id,
                "model_key":        model_key,
                "model_snapshot":   model_snapshot,
                "prompt_developer": prompt_dev_name,
                "prompt_user":      prompt_usr_name,
                "persona_uuid":     uuid,
                "target_words":     params.get("target_words"),
            }

    log.info("Built %d batch requests.", len(requests))
    submitted_at = datetime.now(timezone.utc).isoformat()
    batch_id = submit_batch(client, requests)
    write_batch_registry(data_dir, batch_id, stage, schema_class.__name__,
                         output_file, submitted_at, request_meta)

    if no_wait:
        log.info("--no-wait: exiting after submission. batch_id=%s", batch_id)
        return

    batch = poll_batch(client, batch_id, poll_interval)
    if batch.status == "completed":
        retrieve_batch_output(client, batch, request_meta, schema_class,
                              pricing, output_file, batch_id, data_dir, stage)


def run_scoring_batch(
    client: OpenAI,
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
    load_prompt,
) -> None:
    """Build, submit (and optionally poll+retrieve) a batch for the scoring branch."""
    n_repeats = defaults.get("n_repeats", 1)
    requests = []
    request_meta: dict[str, dict] = {}

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

            raw_usr = load_prompt(prompt_usr_name, prompts_dir)
            usr_prompt = raw_usr.replace("{APPRECIATION_TEXTS}", texts_xml)
            sys_prompt = None
            if prompt_dev_name:
                sys_prompt = load_prompt(prompt_dev_name, prompts_dir)

            messages = (
                [{"role": "system", "content": sys_prompt}] if sys_prompt else []
            ) + [{"role": "user", "content": usr_prompt}]

            for repeat_id in range(1, n_repeats + 1):
                custom_id = f"scoring-{uuid[:8]}-src-{source_variant_id}-variant-{v_id}-repeat-{repeat_id}"
                req = _make_batch_request(custom_id, model_snapshot, messages, params, schema_class)
                requests.append(req)
                request_meta[custom_id] = {
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
                }

    log.info("Built %d scoring batch requests.", len(requests))
    submitted_at = datetime.now(timezone.utc).isoformat()
    batch_id = submit_batch(client, requests)
    write_batch_registry(data_dir, batch_id, stage, schema_class.__name__,
                         output_file, submitted_at, request_meta)

    if no_wait:
        log.info("--no-wait: exiting after submission. batch_id=%s", batch_id)
        return

    batch = poll_batch(client, batch_id, poll_interval)
    if batch.status == "completed":
        retrieve_batch_output(client, batch, request_meta, schema_class,
                              pricing, output_file, batch_id, data_dir, stage)
