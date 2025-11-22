"""Send the wrong-task prompts back to Gemini 3 in batches."""

from __future__ import annotations

import argparse
import ast
import datetime as dt
import json
import os
import pathlib
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from generate_wrong_task_prompts import (
    BASE_DIR,
    EVAL_ROOT,
    MODEL_ROOT,
    build_prompt,
    gather_wrong_task_files,
    load_entries,
    pick_pair_index,
    pick_wrong_answer,
)

try:
    from google import genai
    from google.genai import types as genai_types
except Exception:  # pragma: no cover - optional dependency
    genai = None
    genai_types = None


DEFAULT_MODEL = "gemini-3-pro-preview"
DEFAULT_PROVIDER = "gemini"
# Add task IDs here to skip them without passing CLI flags.
IGNORE_TASKS: set[str] = set()  # set({"09c534e7", "0d87d2a6", "0934a4d8", "14754a24"})


@dataclass
class PromptJob:
    task_id: str
    entry_index: int
    pair_index: int
    prompt: str
    expected_output: Any


def chunked(seq: Sequence[PromptJob], size: int) -> Iterable[Sequence[PromptJob]]:
    for idx in range(0, len(seq), size):
        yield seq[idx : idx + size]


def parse_answer_text(text: str) -> Optional[Any]:
    """
    Try to recover a JSON-ish grid from a free-form LLM response.

    The model sometimes prefixes the answer with prose and a fenced code block
    that holds the grid. Earlier code naively grabbed the *first* "[" in the
    response, which breaks when bracketed snippets (like ``[target_r]``) appear
    before the final grid. We now try, in order:
      1) any fenced ```json```/```...``` blocks (last one first),
      2) the whole response,
      3) bracketed/dict substrings searched from the end of the message.
    """

    cleaned = text.strip()

    def try_parse(candidate: str) -> Optional[Any]:
        candidate = candidate.strip()
        try:
            return json.loads(candidate)
        except Exception:
            pass
        try:
            return ast.literal_eval(candidate)
        except Exception:
            return None

    # 1) Look for fenced code blocks and try the last one first.
    import re

    fenced = re.findall(r"```(?:json)?\s*([\s\S]*?)```", cleaned, flags=re.IGNORECASE)
    for block in reversed(fenced):
        parsed = try_parse(block)
        if parsed is not None:
            return parsed

    # 2) Maybe the whole response is already JSON-ish.
    parsed = try_parse(cleaned)
    if parsed is not None:
        return parsed

    # 3) Walk backwards through bracketed chunks to avoid early non-grid brackets.
    starts = [m.start() for m in re.finditer(r"\[", cleaned)]
    for start in reversed(starts):
        end = cleaned.rfind("]", start)
        if end != -1 and end > start:
            parsed = try_parse(cleaned[start : end + 1])
            if parsed is not None:
                return parsed

    # 4) Do the same for potential dict-style answers.
    brace_starts = [m.start() for m in re.finditer(r"\{", cleaned)]
    for start in reversed(brace_starts):
        end = cleaned.rfind("}", start)
        if end != -1 and end > start:
            parsed = try_parse(cleaned[start : end + 1])
            if parsed is not None:
                return parsed

    return None


def extract_response_text(response: Any) -> str:
    # Prefer the aggregated text field if present.
    for attr in ("output_text", "text"):
        if hasattr(response, attr):
            value = getattr(response, attr)
            if isinstance(value, str):
                return value
    # Otherwise join all text parts from the first candidate.
    if hasattr(response, "candidates"):
        candidates = getattr(response, "candidates") or []
        if candidates:
            parts = getattr(
                getattr(candidates[0], "content", candidates[0]), "parts", []
            )
            texts = [
                str(getattr(p, "text")) for p in parts or [] if getattr(p, "text", None)
            ]
            if texts:
                return "\n".join(texts)
    return ""


def build_jobs(
    model_dir: pathlib.Path,
    attempt: int,
    limit: Optional[int],
    ignore_tasks: Optional[Iterable[str]] = None,
    retry_targets: Optional[set[Tuple[str, int]]] = None,
) -> List[PromptJob]:
    ignore = set(ignore_tasks or IGNORE_TASKS)
    jobs: List[PromptJob] = []
    for wrong_path in gather_wrong_task_files(model_dir):
        task_id = wrong_path.stem
        if task_id in ignore:
            continue
        gold_path = EVAL_ROOT / f"{task_id}.json"
        if not gold_path.exists():
            print(f"Missing gold task file for {task_id}, skipping.", file=sys.stderr)
            continue

        gold = json.load(gold_path.open())
        train_examples = gold.get("train") or []
        test_examples = gold.get("test") or []
        if not test_examples:
            print(f"No test examples found for {task_id}, skipping.", file=sys.stderr)
            continue

        entries = load_entries(wrong_path)
        for entry_idx, entry in enumerate(entries):
            if retry_targets is not None and (task_id, entry_idx) not in retry_targets:
                continue
            pair_index = pick_pair_index(entry)
            if pair_index >= len(test_examples):
                print(
                    f"Pair index {pair_index} out of range for {task_id}, skipping.",
                    file=sys.stderr,
                )
                continue
            test_input = test_examples[pair_index].get("input")
            gold_output = test_examples[pair_index].get("output")
            try:
                wrong_output = pick_wrong_answer(entry, preferred=attempt)
            except ValueError as exc:
                print(f"Skipping {task_id} #{entry_idx}: {exc}", file=sys.stderr)
                continue

            prompt = build_prompt(train_examples, test_input, wrong_output)
            jobs.append(
                PromptJob(
                    task_id=task_id,
                    entry_index=entry_idx,
                    pair_index=pair_index,
                    prompt=prompt,
                    expected_output=gold_output,
                )
            )
            if limit is not None and len(jobs) >= limit:
                return jobs
    return jobs


def is_null_attempt(entry: Dict[str, Any], attempt_label: str) -> bool:
    attempt = entry.get(attempt_label)
    if attempt is None:
        return True
    if not isinstance(attempt, dict):
        return True
    return attempt.get("answer") is None


def find_failed_attempts(
    output_dir: pathlib.Path, attempt_label: str
) -> set[Tuple[str, int]]:
    failed: set[Tuple[str, int]] = set()
    if not output_dir.is_dir():
        return failed
    for path in sorted(output_dir.glob("*.json")):
        try:
            entries = load_entries(path)
        except Exception as exc:
            print(f"Skipping {path.name}: {exc}", file=sys.stderr)
            continue
        for entry_idx, entry in enumerate(entries):
            if is_null_attempt(entry, attempt_label):
                failed.add((path.stem, entry_idx))
    return failed


def count_null_answers(output_dir: pathlib.Path, attempt_label: str) -> int:
    failures = find_failed_attempts(output_dir, attempt_label)
    return len(failures)


def ensure_client(api_key: str) -> Any:
    if genai is None:
        raise RuntimeError(
            "google-genai is required. Install with `pip install google-genai`."
        )
    return genai.Client(api_key=api_key)


def run_single_job(
    api_key: str,
    model: str,
    provider: str,
    temperature: float,
    max_output_tokens: Optional[int],
    test_id: str,
    job: PromptJob,
) -> Dict[str, Any]:
    client = ensure_client(api_key)
    config_kwargs: Dict[str, Any] = {"temperature": temperature}
    if max_output_tokens:
        config_kwargs["max_output_tokens"] = max_output_tokens
    config = genai_types.GenerateContentConfig(**config_kwargs) if genai_types else None

    start_ts = dt.datetime.now(dt.timezone.utc)
    error: Optional[str] = None
    raw_text = ""
    parsed_answer: Any = None
    usage: Dict[str, Any] = {}
    finish_reason: Optional[str] = None
    try:
        contents = [
            genai_types.Content(role="user", parts=[genai_types.Part(text=job.prompt)])
        ]
        response = client.models.generate_content(
            model=model, contents=contents, config=config
        )
        if getattr(response, "candidates", None):
            finish_reason = getattr(response.candidates[0], "finish_reason", None)
        raw_text = extract_response_text(response)
        parsed_answer = parse_answer_text(raw_text)
        usage_meta = getattr(response, "usage_metadata", None)
        if usage_meta:
            prompt_tokens = getattr(usage_meta, "prompt_token_count", 0)
            completion_tokens = getattr(usage_meta, "candidates_token_count", 0)
            reasoning_tokens = getattr(usage_meta, "thoughts_token_count", 0)
            total_tokens = getattr(usage_meta, "total_token_count", 0)
            usage = {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
                "completion_tokens_details": {
                    "reasoning_tokens": reasoning_tokens,
                    "accepted_prediction_tokens": completion_tokens,
                    "rejected_prediction_tokens": 0,
                },
            }
    except Exception as exc:  # pragma: no cover - network call
        error = str(exc)
    end_ts = dt.datetime.now(dt.timezone.utc)

    correct = parsed_answer == job.expected_output and parsed_answer is not None
    kwargs_meta = {k: v for k, v in config_kwargs.items() if v is not None}
    metadata: Dict[str, Any] = {
        "model": model,
        "provider": provider,
        "start_timestamp": start_ts.isoformat(),
        "end_timestamp": end_ts.isoformat(),
        "choices": [
            {"index": 0, "message": {"role": "user", "content": job.prompt}},
            {"index": 1, "message": {"role": "assistant", "content": raw_text}},
        ],
        "reasoning_summary": None,
        "kwargs": kwargs_meta,
        "usage": usage,
        "cost": {},
        "task_id": job.task_id,
        "pair_index": job.pair_index,
        "test_id": test_id,
        "raw_text": raw_text,
    }
    if finish_reason:
        metadata["finish_reason"] = finish_reason
    if error:
        metadata["error"] = error

    return {"answer": parsed_answer, "metadata": metadata, "correct": bool(correct)}


def write_results(
    attempt_label: str,
    output_dir: pathlib.Path,
    outcomes: List[Tuple[PromptJob, Dict[str, Any]]],
) -> None:
    per_task: Dict[str, List[Tuple[int, Dict[str, Any]]]] = {}
    for job, attempt in outcomes:
        per_task.setdefault(job.task_id, []).append((job.entry_index, attempt))

    output_dir.mkdir(parents=True, exist_ok=True)
    for task_id, entries in per_task.items():
        path = output_dir / f"{task_id}.json"
        existing: List[Dict[str, Any]] = []
        if path.exists():
            try:
                existing = load_entries(path)
            except Exception as exc:
                print(f"Skipping existing contents in {path.name}: {exc}", file=sys.stderr)
                existing = []

        max_idx = max(entry_idx for entry_idx, _ in entries)
        while len(existing) <= max_idx:
            existing.append({})

        for entry_idx, attempt in sorted(entries, key=lambda pair: pair[0]):
            if not isinstance(existing[entry_idx], dict):
                existing[entry_idx] = {}
            existing[entry_idx][attempt_label] = attempt

        path.write_text(json.dumps(existing, indent=2))


def build_visualization_dataset(
    output_dir: pathlib.Path, attempt_label: str, grid_to_array_fn, load_gold_fn
) -> List[Dict[str, Any]]:
    dataset: List[Dict[str, Any]] = []
    for path in sorted(output_dir.glob("*.json")):
        try:
            entries = load_entries(path)
        except Exception as exc:
            print(f"Skipping {path.name}: {exc}", file=sys.stderr)
            continue

        for entry_idx, entry in enumerate(entries):
            attempt = entry.get(attempt_label) or {}
            if not attempt:
                continue
            answer = attempt.get("answer")
            attempt_grid = grid_to_array_fn(answer) if answer is not None else None
            if attempt_grid is None:
                print(
                    f"Skipping {path.name} entry {entry_idx}: missing or invalid grid.",
                    file=sys.stderr,
                )
                continue
            meta = attempt.get("metadata") or {}
            try:
                pair_index = int(meta.get("pair_index", 0))
            except Exception:
                pair_index = 0
            gold_grid = load_gold_fn(path.stem, pair_index=pair_index)
            dataset.append(
                {
                    "task_id": path.stem,
                    "pair_index": pair_index,
                    "attempt_grid": attempt_grid,
                    "attempt_correct": bool(attempt.get("correct")),
                    "gold_grid": gold_grid,
                }
            )
    return dataset


def visualize_rerun_attempts(output_dir: pathlib.Path, attempt_label: str) -> int:
    try:  # Lazy import to avoid matplotlib overhead during normal runs.
        from visualize_wrong_tasks import (
            DEFAULT_COLORS,
            ListedColormap,
            draw,
            grid_to_array,
            load_gold,
            plt,
        )
    except Exception as exc:
        print(f"Visualization dependencies missing: {exc}", file=sys.stderr)
        return 1

    dataset = build_visualization_dataset(
        output_dir, attempt_label, grid_to_array, load_gold
    )
    if not dataset:
        print(
            f"No {attempt_label} answers found under {output_dir}, nothing to show.",
            file=sys.stderr,
        )
        return 1

    cmap = ListedColormap(DEFAULT_COLORS)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    status_text = fig.text(
        0.5,
        0.02,
        "Use left/right arrow keys to navigate. Close window to exit.",
        ha="center",
        fontsize=10,
    )

    state = {"idx": 0}

    def redraw():
        item = dataset[state["idx"]]
        fig.suptitle(
            f"Task {item['task_id']} (pair {item['pair_index']}) "
            f"{state['idx'] + 1}/{len(dataset)}",
            fontsize=14,
        )
        draw(
            axes[0],
            item["attempt_grid"],
            f"{attempt_label} ({'correct' if item['attempt_correct'] else 'wrong'})",
            cmap,
        )
        draw(axes[1], item["gold_grid"], "gold output", cmap)
        fig.canvas.draw_idle()

    def on_key(event):
        if event.key == "right":
            state["idx"] = (state["idx"] + 1) % len(dataset)
            redraw()
        elif event.key == "left":
            state["idx"] = (state["idx"] - 1) % len(dataset)
            redraw()

    fig.canvas.mpl_connect("key_press_event", on_key)
    redraw()
    plt.show()

    # Silence linters about unused vars
    _ = status_text
    return 0


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Send wrong-task prompts to Gemini 3 in fixed-size batches."
    )
    parser.add_argument(
        "--model-dir",
        default="gemini-3-pro-preview",
        help="Folder within assets/arc_agi_v1_public_eval/ to pull wrong tasks from.",
    )
    parser.add_argument(
        "--output-dir",
        default="gemini-3-pro-preview-rerun",
        help=(
            "Folder name (or absolute path) for results. Relative paths are resolved "
            "under assets/arc_agi_v1_public_eval/."
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="How many prompts to send concurrently.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on number of prompts to send.",
    )
    parser.add_argument(
        "--test-mode",
        action="store_true",
        help="Shortcut for --limit 4 --batch-size 2.",
    )
    parser.add_argument(
        "--attempt-number",
        type=int,
        default=3,
        help="attempt_X key used in the output JSON.",
    )
    parser.add_argument(
        "--wrong-attempt",
        type=int,
        # choices=(1, 2),
        default=2,
        help="Which attempt to treat as the WRONG OUTPUT when building prompts.",
    )
    parser.add_argument(
        "--retry-failed",
        action="store_true",
        help="Retry entries in the output folder whose answer is null.",
    )
    parser.add_argument(
        "--model", default=DEFAULT_MODEL, help="Gemini model name to call."
    )
    parser.add_argument(
        "--provider",
        default=DEFAULT_PROVIDER,
        help="Provider label stored in metadata.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature for the generation request.",
    )
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=65536,
        help="Max output tokens for the generation request.",
    )
    parser.add_argument(
        "--api-key-env",
        default="GOOGLE_API_KEY",
        help="Environment variable that stores the Gemini API key.",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Print per-request progress."
    )
    parser.add_argument(
        "--test-id",
        default=None,
        help="test_id field for metadata. Defaults to the output folder name.",
    )
    parser.add_argument(
        "--visualize",
        "--visualise",
        dest="visualize",
        action="store_true",
        help="Open a viewer comparing rerun answers against the gold output.",
    )
    return parser.parse_args(argv)


def resolve_output_dir(path: str) -> pathlib.Path:
    out_path = pathlib.Path(path)
    if out_path.is_absolute():
        return out_path
    return BASE_DIR / out_path


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    if args.test_mode:
        args.limit = 4
        args.batch_size = 2

    attempt_label = f"attempt_{args.attempt_number}"
    output_dir = resolve_output_dir(args.output_dir)
    retry_targets: Optional[set[Tuple[str, int]]] = None

    if args.visualize:
        if not output_dir.is_dir():
            print(f"{output_dir} is not a directory", file=sys.stderr)
            return 1
        if not EVAL_ROOT.is_dir():
            print(f"Missing evaluation directory at {EVAL_ROOT}", file=sys.stderr)
            return 1
        return visualize_rerun_attempts(output_dir, attempt_label)

    if args.retry_failed and not output_dir.is_dir():
        print(f"{output_dir} is not a directory", file=sys.stderr)
        return 1

    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        print(
            f"Missing API key. Export {args.api_key_env}=<your key>.", file=sys.stderr
        )
        return 1

    model_dir = MODEL_ROOT / args.model_dir
    if not model_dir.is_dir():
        print(f"{model_dir} is not a directory", file=sys.stderr)
        return 1

    if args.retry_failed:
        retry_targets = find_failed_attempts(output_dir, attempt_label)

    jobs = build_jobs(
        model_dir,
        attempt=args.wrong_attempt,
        limit=args.limit,
        ignore_tasks=IGNORE_TASKS,
        retry_targets=retry_targets,
    )
    if not jobs:
        remaining_nulls = count_null_answers(output_dir, attempt_label)
        if args.retry_failed:
            print(
                f"No failed attempts to retry. Remaining null answers: {remaining_nulls}",
                file=sys.stderr,
            )
            return 0
        print("No prompts to send.", file=sys.stderr)
        print(
            f"Remaining null answers for {attempt_label}: {remaining_nulls}",
            file=sys.stderr,
        )
        return 1

    test_id = args.test_id or output_dir.name

    outcomes: List[Tuple[PromptJob, Dict[str, Any]]] = []
    total = len(jobs)
    batch_size = max(1, args.batch_size)
    for batch_idx, batch in enumerate(chunked(jobs, batch_size), start=1):
        batch_str = f"Batch {batch_idx} ({len(batch)} jobs)"
        print(f"Sending {batch_str}...", file=sys.stderr)
        with ThreadPoolExecutor(max_workers=batch_size) as executor:
            futures = {
                executor.submit(
                    run_single_job,
                    api_key,
                    args.model,
                    args.provider,
                    args.temperature,
                    args.max_output_tokens,
                    test_id,
                    job,
                ): job
                for job in batch
            }
            for future in as_completed(futures):
                job = futures[future]
                try:
                    attempt = future.result()
                except Exception as exc:  # pragma: no cover - defensive
                    if args.verbose:
                        print(
                            f"Job {job.task_id}#{job.entry_index} failed: {exc}",
                            file=sys.stderr,
                        )
                    attempt = {
                        "answer": None,
                        "metadata": {
                            "model": args.model,
                            "provider": "gemini",
                            "error": str(exc),
                            "pair_index": job.pair_index,
                            "task_id": job.task_id,
                        },
                        "correct": False,
                    }
                else:
                    if args.verbose:
                        meta = attempt.get("metadata", {})
                        print(
                            f"Job {job.task_id}#{job.entry_index} ok "
                            f"(pair {job.pair_index}, correct={attempt.get('correct')})",
                            file=sys.stderr,
                        )
                outcomes.append((job, attempt))
        processed = min(batch_idx * batch_size, total)
        print(f"Finished {processed}/{total}", file=sys.stderr)

    write_results(attempt_label, output_dir, outcomes)
    print(f"Wrote results to {output_dir}", file=sys.stderr)
    remaining_nulls = count_null_answers(output_dir, attempt_label)
    print(
        f"Remaining null answers for {attempt_label}: {remaining_nulls}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    raise SystemExit(main())
