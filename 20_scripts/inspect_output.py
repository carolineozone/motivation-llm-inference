"""inspect_output.py — pretty-print a pipeline output JSONL file."""

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def truncate(text: str, max_len: int = 80) -> str:
    return text if len(text) <= max_len else text[:max_len - 1] + "…"


BPNS_COLS = ("q_aut_1", "q_aut_2", "q_aut_3", "q_aut_4",
             "q_com_1", "q_com_2", "q_com_3", "q_com_4",
             "q_rel_1", "q_rel_2", "q_rel_3", "q_rel_4")

SCORING_COLS = ("llm_aut", "llm_com", "llm_rel")

SENTENCE_ENDINGS = frozenset(".!?\"'\u201d\u2019")


def _mean_sd(values: list[float]) -> str:
    n = len(values)
    if n == 0:
        return "—"
    m = sum(values) / n
    if n < 2:
        return f"{m:.2f}"
    sd = math.sqrt(sum((x - m) ** 2 for x in values) / (n - 1))
    return f"{m:.2f}±{sd:.2f}"


def _group_by(rows: list[dict], *keys: str) -> dict:
    groups: dict = defaultdict(list)
    for r in rows:
        k = tuple(r.get(k, "") for k in keys)
        groups[k].append(r)
    return dict(groups)


# ---------------------------------------------------------------------------
# Truncation detection (new)
# ---------------------------------------------------------------------------

def _check_truncation(rows: list[dict]) -> None:
    """Warn about text rows that look truncated. Prints nothing if no issues."""
    text_rows = [r for r in rows if r.get("text") is not None and r.get("target_words") is not None]
    if not text_rows:
        return

    flagged = []
    for r in text_rows:
        text = r.get("text", "") or ""
        target = r.get("target_words") or 0
        word_count = len(text.split())
        ratio = word_count / target if target else 1.0
        stripped = text.rstrip()
        ends_ok = bool(stripped) and stripped[-1] in SENTENCE_ENDINGS

        reasons = []
        if ratio < 0.75:
            reasons.append(f"only {word_count}/{target} words ({ratio:.0%})")
        if not ends_ok:
            last_char = repr(stripped[-1]) if stripped else "empty"
            reasons.append(f"ends with {last_char} (not sentence-ending)")

        if reasons:
            flagged.append((r, reasons, stripped[-60:] if stripped else ""))

    if not flagged:
        return

    print(f"  *** TRUNCATION WARNING: {len(flagged)}/{len(text_rows)} texts look truncated ***")
    for r, reasons, tail in flagged[:5]:
        vid = r.get("variant_id", "?")
        pid = (r.get("persona_uuid") or "")[-8:]
        print(f"      [{vid} | …{pid}] {'; '.join(reasons)}")
        print(f"      tail: …{truncate(tail, 60)}")
    if len(flagged) > 5:
        print(f"      … and {len(flagged) - 5} more")
    print()


# ---------------------------------------------------------------------------
# Summary table (unchanged)
# ---------------------------------------------------------------------------

def print_table(rows: list[dict]) -> None:
    is_bpns    = any(r.get("q_aut_1") is not None for r in rows)
    is_scoring = any(r.get("llm_aut") is not None for r in rows)
    is_filter  = any(r.get("has_coworkers") is not None for r in rows)

    if is_scoring:
        # One summary row per variant × repeat
        groups = _group_by(rows, "variant_id", "repeat_id", "source_variant_id")
        headers = ("variant_id", "repeat_id", "source_variant_id") + SCORING_COLS
        data = []
        for (vid, rid, svid), grp in sorted(groups.items()):
            row = (vid, str(rid), svid) + tuple(
                _mean_sd([r[c] for r in grp if r.get(c) is not None])
                for c in SCORING_COLS
            )
            data.append(row)
        print(f"  (means ± SD across {len(next(iter(groups.values())))} personas per cell)")

    elif is_bpns:
        # One summary row per variant — 3 subscale means instead of 12 items
        SUBSCALES = {
            "aut": ("q_aut_1", "q_aut_2", "q_aut_3", "q_aut_4"),
            "com": ("q_com_1", "q_com_2", "q_com_3", "q_com_4"),
            "rel": ("q_rel_1", "q_rel_2", "q_rel_3", "q_rel_4"),
        }
        groups = _group_by(rows, "variant_id")
        headers = ("variant_id", "aut_mean", "com_mean", "rel_mean")
        data = []
        for (vid,), grp in sorted(groups.items()):
            row = (vid,) + tuple(
                _mean_sd([
                    sum(r.get(q, 0) for q in items) / len(items)
                    for r in grp
                    if all(r.get(q) is not None for q in items)
                ])
                for items in SUBSCALES.values()
            )
            data.append(row)
        n_per = len(next(iter(groups.values())))
        print(f"  (subscale means ± SD across {n_per} personas; items averaged within subscale)")

    elif is_filter:
        headers = ("variant_id", "persona_uuid", "has_coworkers")
        data = [
            (r.get("variant_id", ""), r.get("persona_uuid", "")[-8:], str(r.get("has_coworkers", "")))
            for r in rows
        ]
    else:
        # Text gen: one sample row per variant
        groups = _group_by(rows, "variant_id")
        headers = ("variant_id", "coworker_id", "text_sample")
        data = [
            (vid, str(grp[0].get("coworker_id", "")), truncate(grp[0].get("text", "")))
            for (vid,), grp in sorted(groups.items())
        ]
        print(f"  (1 sample row per variant — {len(rows)} rows total)")

    print()
    col_widths = [len(h) for h in headers]
    for row in data:
        for i, cell in enumerate(row):
            col_widths[i] = max(col_widths[i], len(cell))

    fmt = "  ".join(f"{{:<{w}}}" for w in col_widths)
    sep = "  ".join("-" * w for w in col_widths)

    print(fmt.format(*headers))
    print(sep)
    for row in data:
        print(fmt.format(*row))


# ---------------------------------------------------------------------------
# inspect_file (importable entry point) — truncation check added
# ---------------------------------------------------------------------------

def inspect_file(path: Path) -> None:
    """Load and pretty-print a pipeline output JSONL file. Importable by other scripts."""
    try:
        with open(path, encoding="utf-8") as f:
            rows = [json.loads(line) for line in f if line.strip()]
    except FileNotFoundError:
        print(f"Error: file not found: {path}", file=sys.stderr)
        return

    if not rows:
        print("File is empty.")
        return

    first = rows[0]
    stage = first.get("stage", "—")
    model = ", ".join(dict.fromkeys(r.get("model_snapshot", "—") for r in rows))
    n_personas = len({r.get("persona_uuid") for r in rows})
    n_variants = len({r.get("variant_id") for r in rows})
    is_scoring = any(r.get("llm_aut") is not None for r in rows)

    print(f"Stage:     {stage}")
    print(f"Model:     {model}")
    print(f"Personas:  {n_personas}")
    print(f"Variants:  {n_variants}")
    if is_scoring:
        n_repeats = len({r.get("repeat_id") for r in rows if r.get("persona_uuid") == first.get("persona_uuid")})
        source_file = first.get("source_file", "—")
        print(f"Repeats:   {n_repeats}")
        print(f"Source:    {source_file}")
        print(f"Rows:      {len(rows)}  (= {n_personas} personas × {n_variants} variants × {n_repeats} repeats)")
    else:
        n_api_calls = n_personas * n_variants
        print(f"API calls: {n_api_calls}")
        print(f"Rows:      {len(rows)}")
    print()

    _check_truncation(rows)
    print_table(rows)


# ---------------------------------------------------------------------------
# print_texts (unchanged)
# ---------------------------------------------------------------------------

def print_texts(rows: list[dict], n: int | None) -> None:
    """Print full text content, grouped by variant then coworker."""
    groups = _group_by(rows, "variant_id")
    shown = 0
    for (vid,), grp in sorted(groups.items()):
        grp_sorted = sorted(grp, key=lambda r: r.get("coworker_id", 0))
        for r in grp_sorted:
            if n is not None and shown >= n:
                return
            cid = r.get("coworker_id", "?")
            persona = r.get("persona_uuid", "")[-8:]
            text = r.get("text", "")
            print(f"── {vid} | coworker {cid} | persona …{persona} ──")
            print(text)
            print()
            shown += 1


# ---------------------------------------------------------------------------
# browse mode — paginated table for any JSONL/JSON file (new)
# ---------------------------------------------------------------------------

def _load_any(path: Path) -> list[dict]:
    """Load JSONL or JSON file into a list of dicts."""
    with open(path, encoding="utf-8") as f:
        raw = f.read()

    # Try JSONL: one JSON object per non-empty line
    lines = [l for l in raw.splitlines() if l.strip()]
    if lines:
        try:
            parsed = [json.loads(l) for l in lines]
            if all(isinstance(p, dict) for p in parsed):
                return parsed
        except json.JSONDecodeError:
            pass

    # Fall back to single JSON document
    doc = json.loads(raw)
    if isinstance(doc, list):
        return [r if isinstance(r, dict) else {"value": r} for r in doc]
    if isinstance(doc, dict):
        for v in doc.values():
            if isinstance(v, list) and v and isinstance(v[0], dict):
                return v
        return [doc]
    return [{"value": doc}]


def _auto_cols(rows: list[dict]) -> list[str]:
    """Pick default columns: all keys in order of first appearance, skipping long/noisy fields."""
    noisy = {"text", "reasoning", "prompt_developer", "prompt_user",
             "reasoning_output", "visible_output", "timestamp"}
    seen: set[str] = set()
    cols: list[str] = []
    for r in rows:
        for k in r:
            if k not in seen:
                seen.add(k)
                if k not in noisy:
                    cols.append(k)
    return cols if cols else list(rows[0].keys())


def _cell(value, max_len: int = 35) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.3f}"
    s = str(value)
    return s if len(s) <= max_len else s[:max_len - 1] + "…"


def browse(path: Path, cols: list[str] | None, page_size: int) -> None:
    """Interactive paginated table browser for any JSONL/JSON file."""
    try:
        rows = _load_any(path)
    except FileNotFoundError:
        print(f"Error: file not found: {path}", file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: could not parse JSON: {e}", file=sys.stderr)
        sys.exit(1)

    if not rows:
        print("File is empty.")
        return

    if cols is None:
        cols = _auto_cols(rows)

    # Compute column widths from header + sample of rows
    sample = rows[:200]
    col_widths = {c: len(c) for c in cols}
    for r in sample:
        for c in cols:
            col_widths[c] = max(col_widths[c], len(_cell(r.get(c))))

    fmt = "  ".join(f"{{:<{col_widths[c]}}}" for c in cols)
    sep = "  ".join("-" * col_widths[c] for c in cols)
    header_line = fmt.format(*[truncate(c, col_widths[c]) for c in cols])

    total_rows = len(rows)
    offset = 0

    while offset < total_rows:
        page = rows[offset: offset + page_size]
        end = offset + len(page)
        n_pages = math.ceil(total_rows / page_size)

        print()
        print(f"  {path.name}  —  rows {offset + 1}–{end} of {total_rows}"
              + (f"  (page {offset // page_size + 1} of {n_pages})" if n_pages > 1 else ""))
        print(f"  cols: {', '.join(cols)}")
        print()
        print(header_line)
        print(sep)
        for r in page:
            cells = [truncate(_cell(r.get(c)), col_widths[c]) for c in cols]
            print(fmt.format(*cells))

        offset = end
        if offset >= total_rows:
            print("\n  (end of file)")
            break

        try:
            ans = input("\n  [Enter] next page  [q] quit  [a] all remaining: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if ans == "q":
            break
        if ans == "a":
            page_size = total_rows


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect a pipeline output JSONL file.")
    parser.add_argument("file", help="Path to JSONL or JSON file")
    parser.add_argument(
        "--texts", nargs="?", const=0, type=int, metavar="N",
        help="Print full text content instead of summary table. Optionally limit to N texts.",
    )
    parser.add_argument(
        "--browse", action="store_true",
        help="Interactive paginated table browser (works with any JSONL/JSON file).",
    )
    parser.add_argument(
        "--cols", metavar="FIELDS",
        help="Comma-separated column names for --browse (default: auto-selected).",
    )
    parser.add_argument(
        "--page-size", type=int, default=20, metavar="N",
        help="Rows per page in --browse mode (default: 20).",
    )
    args = parser.parse_args()
    path = Path(args.file)

    if args.browse:
        cols = [c.strip() for c in args.cols.split(",")] if args.cols else None
        browse(path, cols, args.page_size)
        return

    if args.texts is None:
        inspect_file(path)
        return

    try:
        with open(path, encoding="utf-8") as f:
            rows = [json.loads(line) for line in f if line.strip()]
    except FileNotFoundError:
        print(f"Error: file not found: {path}", file=sys.stderr)
        sys.exit(1)

    if not rows:
        print("File is empty.")
        return

    limit = args.texts if args.texts > 0 else None
    print_texts(rows, limit)


if __name__ == "__main__":
    main()
