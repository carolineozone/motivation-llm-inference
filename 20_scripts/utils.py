# %% [markdown]
# # utils.py
# Shared utilities for all pipeline scripts.
# Import from here — do not redefine inline.

# %% Imports
import importlib.util
import json
import logging
from collections.abc import Callable
from pathlib import Path

log = logging.getLogger(__name__)

# %% Pricing tier map
# Maps mode name → (input_price_key, output_price_key) in pricing.yaml
_PRICE_KEYS: dict[str, tuple[str, str]] = {
    "standard": ("input",       "output"),
    "batch":    ("batchinput",  "batchoutput"),
    "cached":   ("cachedinput", "output"),
}


# %% Cost calculation
def calculate_cost(
    usage,
    pricing: dict,
    model_key: str,
    mode: str = "standard",
) -> dict:
    """
    Compute API call cost from a response usage object.

    Args:
        usage:     OpenAI response.usage object.
        pricing:   Full _pricing dict loaded from pricing.yaml (passed by caller).
        model_key: Lookup key in pricing.yaml (e.g. "gpt-5-mini"), NOT the snapshot string.
        mode:      Pricing tier — "standard" | "batch" | "cached". Default: "standard".

    Returns dict with input_tokens, visible_output, reasoning_output, cost_usd.
    """
    if mode not in _PRICE_KEYS:
        raise ValueError(f"Unknown pricing mode {mode!r}. Expected: {list(_PRICE_KEYS)}")
    input_key, output_key = _PRICE_KEYS[mode]
    prices = pricing[model_key]
    details = getattr(usage, "completion_tokens_details", None)
    reasoning_tokens = getattr(details, "reasoning_tokens", 0) if details else 0
    visible_tokens = usage.completion_tokens - reasoning_tokens
    cost = (usage.prompt_tokens       / 1_000_000 * prices[input_key]
            + usage.completion_tokens / 1_000_000 * prices[output_key])
    return {
        "input_tokens":    usage.prompt_tokens,
        "visible_output":  visible_tokens,
        "reasoning_output": reasoning_tokens,
        "cost_usd":        round(cost, 6),
    }


def calculate_cost_anthropic(
    usage,
    pricing: dict,
    model_key: str,
    mode: str = "standard",
) -> dict:
    """
    Compute API call cost from an Anthropic response usage object.

    Args:
        usage:     Anthropic response.usage object.
        pricing:   Full pricing dict loaded from pricing.yaml.
        model_key: Lookup key in pricing.yaml (e.g. "claude-haiku").
        mode:      Pricing tier — "standard" | "batch". Default: "standard".

    Returns dict with input_tokens, output_tokens, thinking_tokens, cost_usd.
    thinking_tokens is a subset of output_tokens (billed at output rate) — logged for tracking only.
    """
    if mode not in ("standard", "batch"):
        raise ValueError(f"Unknown pricing mode {mode!r}. Expected: standard | batch")
    prices = pricing[model_key]
    input_price  = prices["batchinput"]  if mode == "batch" else prices["input"]
    output_price = prices["batchoutput"] if mode == "batch" else prices["output"]

    input_tokens    = getattr(usage, "input_tokens",    0) or 0
    output_tokens   = getattr(usage, "output_tokens",   0) or 0
    thinking_tokens = getattr(usage, "thinking_tokens", 0) or 0

    cost = (
        input_tokens  / 1_000_000 * input_price
        + output_tokens / 1_000_000 * output_price
    )
    return {
        "input_tokens":    input_tokens,
        "output_tokens":   output_tokens,
        "thinking_tokens": thinking_tokens,
        "cost_usd":        round(cost, 6),
    }


# %% Token estimation
def estimate_tokens(messages: list[dict]) -> int:
    """
    Rough token estimate for a message list. Uses word count × 1.35 heuristic.
    Good enough for DRY_RUN cost previews — no tiktoken dependency needed.
    """
    total_words = sum(len(m["content"].split()) for m in messages)
    return int(total_words * 1.35)


# %% Prompt loading
def load_prompt(name: str, prompts_dir: Path) -> str:
    """
    Read a prompt file by name.
    Extension-agnostic: matches an extension-less file first, then any extension.
    Raises FileNotFoundError if no match is found.
    """
    exact = prompts_dir / name
    matches = sorted(prompts_dir.glob(f"{name}.*"))
    candidates = ([exact] if exact.is_file() else []) + matches
    if not candidates:
        raise FileNotFoundError(f"Prompt file not found: {prompts_dir / name} (no extension match)")
    path = candidates[0]
    return path.read_text(encoding="utf-8").strip()


# %% Schema loading
def load_schema(schema_name: str, schemas_path: Path) -> type:
    """
    Dynamically load a Pydantic class by name from schemas.py.
    Uses importlib to avoid polluting sys.path.
    """
    spec = importlib.util.spec_from_file_location("schemas", schemas_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return getattr(mod, schema_name)


# %% API params builder
def build_api_params(
    model: str,
    messages: list,
    schema_class: type,
    params: dict,
) -> dict:
    """
    Build the api_params dict for client.beta.chat.completions.parse().
    Conditionally adds temperature and reasoning_effort only when present in params.
    """
    api_params: dict = {
        "model":                 model,
        "messages":              messages,
        "max_completion_tokens": params["max_completion_tokens"],
        "response_format":       schema_class,
    }
    if params.get("temperature") is not None:
        api_params["temperature"] = params["temperature"]
    if params.get("reasoning_effort") is not None:
        api_params["reasoning_effort"] = params["reasoning_effort"]
    return api_params


# %% Reasoning token budget check
# Only meaningful for stage 40 (text generation), where target_words drives output size.
# Stages without target_words (e.g. 30, 60) skip the check — output is short and bounded.
# Fraction of max_completion_tokens the model allocates to internal reasoning,
# per reasoning_effort level. Source: OpenAI reasoning models documentation.
_REASONING_SHARE: dict[str, float] = {
    "minimal": 0.10,
    "low":     0.20,
    "medium":  0.50,
    "high":    0.80,
}

# TextsOutput schema always has 5 texts (text1-text5); prompt always requests exactly 5.
_N_TEXTS_PER_CALL = 5
_TOKENS_PER_WORD  = 1.3        # rough English average
_JSON_SCHEMA_OVERHEAD = 50     # field names + braces + quotes in structured output


def check_reasoning_token_budget(params: dict, variant_label: str = "") -> None:
    """
    Warn or raise if max_completion_tokens is likely insufficient for a reasoning model.

    Uses OpenAI's documented reasoning share per effort level to estimate tokens remaining
    for visible output. Raises ValueError if estimated visible tokens fall below what the
    output schema needs; logs a warning if headroom is less than 1.5× the estimated need.

    If target_words is present in params, estimates needed tokens from TextsOutput schema
    (5 texts × target_words × 1.3 tok/word + overhead). Otherwise uses a 200-token floor.

    Call once per variant before the main generation loop or batch submission.
    """
    effort = params.get("reasoning_effort")
    if effort is None:
        return  # not a reasoning model — nothing to check

    share = _REASONING_SHARE.get(effort)
    if share is None:
        log.warning(
            "Unknown reasoning_effort=%r; skipping token budget check. "
            "Add it to _REASONING_SHARE in utils.py.",
            effort,
        )
        return

    limit = params.get("max_completion_tokens", 0)
    estimated_reasoning = int(limit * share)
    estimated_visible   = limit - estimated_reasoning
    tag = f" [{variant_label}]" if variant_label else ""

    target_words = params.get("target_words")
    if target_words:
        needed = int(target_words * _TOKENS_PER_WORD * _N_TEXTS_PER_CALL + _JSON_SCHEMA_OVERHEAD)
    else:
        needed = 50  # generic floor for stages without target_words

    if estimated_visible < needed:
        raise ValueError(
            f"max_completion_tokens={limit} with reasoning_effort={effort!r} leaves "
            f"~{estimated_visible} tokens for visible output, but ~{needed} are needed"
            f"{tag}. Increase max_completion_tokens to at least "
            f"{int(needed / (1 - share)) + 200}."
        )

    if estimated_visible < needed * 1.5:
        log.warning(
            "Token budget tight%s: max_completion_tokens=%d, reasoning_effort=%r "
            "→ ~%d reasoning, ~%d visible (need ~%d). Headroom: %.1f×.",
            tag, limit, effort, estimated_reasoning, estimated_visible, needed,
            estimated_visible / needed,
        )


# %% Conversation builders
def build_persona_multiturn(persona: dict, sys_prompt: str, usr_prompt: str) -> list[dict]:
    """
    Build a multi-turn conversation from persona Q&A pairs.
    professional_identity_and_career answers become user/assistant turns.
    usr_prompt is appended as the final user message.
    """
    qa = {k: v for k, v in persona.get("professional_identity_and_career", {}).items()
          if k != "Do you enjoy your work?"}
    messages = [{"role": "system", "content": sys_prompt}]
    for question, answer in qa.items():
        messages.append({"role": "user", "content": question})
        messages.append({"role": "assistant", "content": str(answer)})
    messages.append({"role": "user", "content": usr_prompt})
    return messages


def build_persona_dump(persona: dict, sys_prompt: str, usr_prompt: str) -> list[dict]:
    """
    Build a single-turn conversation with a JSON persona block appended to the user prompt.
    Extracts demographics + professional_identity into a structured dict.
    """
    demo = persona.get("demographic_information", {})
    profile = {
        "demographics": {
            "age":        demo.get("Select Your Age"),
            "occupation": demo.get("Provide Your Occupation. (_NA if not applicable_)"),
            "gender":     demo.get("Select Your Gender"),
            "education":  demo.get("Select Your Highest Level of Education"),
        },
        "professional_identity": {k: v for k, v in persona.get("professional_identity_and_career", {}).items()
                                   if k != "Do you enjoy your work?"},
    }
    content = f"{usr_prompt}\n\n### PERSONA PROFILE ###\n{json.dumps(profile, indent=2)}"
    return [{"role": "system", "content": sys_prompt}, {"role": "user", "content": content}]


def build_text_only(_ignored: dict, sys_prompt: str | None, usr_prompt: str) -> list[dict]:
    """
    Blind text scoring: user message only. System message omitted if sys_prompt is None.
    The usr_prompt must already have {APPRECIATION_TEXTS} substituted before being passed in.
    """
    messages = []
    if sys_prompt:
        messages.append({"role": "system", "content": sys_prompt})
    messages.append({"role": "user", "content": usr_prompt})
    return messages


BUILDERS: dict[str, Callable] = {
    "multiturn":  build_persona_multiturn,
    "json_dump":  build_persona_dump,
    "text_only":  build_text_only,
}


# %% Stage execution helpers

def fill_prompt(raw: str, persona: dict, params: dict) -> str:
    """
    Apply format_map substitution to a raw prompt string.
    {persona} is injected from the persona JSONL; other placeholders come from params.
    """
    import json as _json
    demo = persona.get("demographic_information", {})
    persona_profile = {
        "demographics": {
            "age":        demo.get("Select Your Age"),
            "occupation": demo.get("Provide Your Occupation. (_NA if not applicable_)"),
            "gender":     demo.get("Select Your Gender"),
            "education":  demo.get("Select Your Highest Level of Education"),
        },
        "professional_identity": {
            k: v for k, v in persona.get("professional_identity_and_career", {}).items()
            if k != "Do you enjoy your work?"
        },
    }
    substitutions = {"persona": _json.dumps(persona_profile, indent=2), **params}
    try:
        return raw.format_map(substitutions)
    except KeyError as e:
        raise KeyError(f"Prompt placeholder {e} not found in params or persona. "
                       f"Available keys: {list(substitutions.keys())}") from e


def unique_output_path(data_dir: Path, stage: str, n: int) -> Path:
    """Return data_dir/{stage}_{YYYYMMDD-HHMM}_n{n}.jsonl."""
    from datetime import datetime as _dt
    ts = _dt.now().strftime("%Y%m%d-%H%M")
    path = data_dir / f"{stage}_{ts}_n{n}.jsonl"
    if path.exists():
        raise RuntimeError(f"Output file already exists: {path}")
    return path


def print_cost_estimate(
    personas: list,
    variants: dict,
    defaults: dict,
    pricing: dict,
    stage_cfg: dict,
    mode: str = "standard",
) -> None:
    """Print a cost estimate table for a persona-branch stage."""
    input_key, output_key = _PRICE_KEYS[mode]
    _build = BUILDERS[stage_cfg.get("conversation_format", "json_dump")]
    sample_msgs = _build(personas[0], "sys", "usr")
    est_input = estimate_tokens(sample_msgs)
    n = len(personas)
    print(f"\n=== COST ESTIMATE [{mode}] ({n} persona(s) × {len(variants)} variant(s)) ===")
    header = f"{'variant':<16} {'model_key':<14} {'est_in':>10} {'max_out':>10} {'est_total_usd':>14}"
    print(header)
    print("-" * len(header))
    total = 0.0
    for v_id, overrides in variants.items():
        params = {**defaults, **(overrides or {})}
        model_key = params["model_key"]
        prices = pricing[model_key]
        max_out = params["max_completion_tokens"]
        est_cost = ((est_input / 1e6 * prices[input_key]) + (max_out / 1e6 * prices[output_key])) * n
        total += est_cost
        print(f"{v_id:<16} {model_key:<14} {est_input:>10,} {max_out:>10,} ${est_cost:>13.6f}")
    print("-" * len(header))
    print(f"{'TOTAL':<42} ${total:>13.6f}\n")


def print_scoring_cost_estimate(
    grouped: dict,
    variants: dict,
    defaults: dict,
    pricing: dict,
    mode: str = "standard",
) -> None:
    """Print a cost estimate table for a scoring-branch stage."""
    input_key, output_key = _PRICE_KEYS[mode]
    n_repeats = defaults.get("n_repeats", 1)
    n_groups = len(grouped)
    n_calls = n_groups * len(variants) * n_repeats
    sample_rows = next(iter(grouped.values()))
    sample_text = " ".join(r.get("text", "") for r in sample_rows)
    est_input = int(len(sample_text.split()) * 1.35) + 50
    print(f"\n=== SCORING COST ESTIMATE [{mode}] "
          f"({n_groups} group(s) × {len(variants)} variant(s) × {n_repeats} repeat(s)"
          f" = {n_calls} call(s)) ===")
    header = f"{'variant':<16} {'model_key':<14} {'est_in':>10} {'max_out':>10} {'est_total_usd':>14}"
    print(header)
    print("-" * len(header))
    total = 0.0
    for v_id, overrides in variants.items():
        params = {**defaults, **(overrides or {})}
        model_key = params["model_key"]
        prices = pricing[model_key]
        max_out = params["max_completion_tokens"]
        est_cost = ((est_input / 1e6 * prices[input_key]) + (max_out / 1e6 * prices[output_key])) * n_calls
        total += est_cost
        print(f"{v_id:<16} {model_key:<14} {est_input:>10,} {max_out:>10,} ${est_cost:>13.6f}")
    print("-" * len(header))
    print(f"{'TOTAL':<42} ${total:>13.6f}\n")


# %% Output row writer
def write_output_rows(f_out, base_meta: dict, result, cost: dict) -> int:
    """
    Write one JSONL row per generated text item to f_out.

    For schemas with a .texts attribute (list of str or TextItem), expands to
    one row per item and zeros cost fields on rows 2+. For single-result schemas,
    writes one row. Returns the number of rows written.
    """
    import json
    if hasattr(result, "as_list"):
        items = result.as_list()
    elif hasattr(result, "texts"):
        items = result.texts
    else:
        items = [result]
    for i, item in enumerate(items):
        row_cost = cost if i == 0 else {k: (0 if k == "cost_usd" else v) for k, v in cost.items()}
        row = {"text": item} if isinstance(item, str) else item.model_dump()
        row["coworker_id"] = i + 1
        f_out.write(json.dumps({**base_meta, **row_cost, **row}, ensure_ascii=False) + "\n")
    return len(items)


# %% API cost log
COST_LOG_FILE = "api_cost_log.jsonl"


def append_cost_log(
    data_dir: Path,
    log_timestamp: str,
    stage: str,
    variant_id: str,
    model_key: str,
    model_snapshot: str,
    pricing_mode: str,
    persona_uuid: str,
    cost: dict,
    output_file: Path,
    request_id: str | None = None,
    batch_id: str | None = None,
) -> None:
    """Append one row per API call to the persistent cost log."""
    row = {
        "log_timestamp":    log_timestamp,
        "stage":            stage,
        "variant_id":       variant_id,
        "model_key":        model_key,
        "model_snapshot":   model_snapshot,
        "pricing_mode":     pricing_mode,
        "persona_uuid":     persona_uuid,
        **cost,
        "output_file":      output_file.name,
        "request_id":       request_id,
        "batch_id":         batch_id,
    }
    with open(data_dir / COST_LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")
