"""Retry the original Gemini prompts at multiple temperatures."""

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

from generate_wrong_task_prompts import BASE_DIR, EVAL_ROOT, load_entries

try:
    from google import genai
    from google.genai import types as genai_types
except Exception:  # pragma: no cover - optional dependency
    genai = None
    genai_types = None


DEFAULT_MODEL = "gemini-3-pro-preview"
DEFAULT_PROVIDER = "gemini"
TASKS_DIR = BASE_DIR / "tasks to retry"
OUTPUT_ROOT = BASE_DIR / "original-prompt-ablations"
# (label, temperature, attempts)
TEMPERATURE_SCHEDULE: List[Tuple[str, float, int]] = [
    # ("temp-0", 0.0, 1),
    ("temp-0-point-5", 0.5, 4),
    ("temp-1", 1.0, 4),
    ("temp-1-point-5", 1.5, 4),
]


@dataclass
class RetryJob:
    task_id: str
    entry_index: int
    attempt_label: str
    pair_index: int
    prompt: str
    expected_output: Any
    temperature: float
    output_name: str


def chunked(seq: Sequence[RetryJob], size: int) -> Iterable[Sequence[RetryJob]]:
    for idx in range(0, len(seq), size):
        yield seq[idx : idx + size]


def parse_answer_text(text: str) -> Optional[Any]:
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

    import re

    fenced = re.findall(r"```(?:json)?\s*([\s\S]*?)```", cleaned, flags=re.IGNORECASE)
    for block in reversed(fenced):
        parsed = try_parse(block)
        if parsed is not None:
            return parsed

    parsed = try_parse(cleaned)
    if parsed is not None:
        return parsed

    starts = [m.start() for m in re.finditer(r"\[", cleaned)]
    for start in reversed(starts):
        end = cleaned.rfind("]", start)
        if end != -1 and end > start:
            parsed = try_parse(cleaned[start : end + 1])
            if parsed is not None:
                return parsed

    brace_starts = [m.start() for m in re.finditer(r"\{", cleaned)]
    for start in reversed(brace_starts):
        end = cleaned.rfind("}", start)
        if end != -1 and end > start:
            parsed = try_parse(cleaned[start : end + 1])
            if parsed is not None:
                return parsed

    return None


def extract_response_text(response: Any) -> str:
    for attr in ("output_text", "text"):
        if hasattr(response, attr):
            value = getattr(response, attr)
            if isinstance(value, str):
                return value
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


def ensure_client(api_key: str) -> Any:
    if genai is None:
        raise RuntimeError(
            "google-genai is required. Install with `pip install google-genai`."
        )
    return genai.Client(api_key=api_key)


def load_gold_output(task_id: str, pair_index: int) -> Any:
    gold_path = EVAL_ROOT / f"{task_id}.json"
    if not gold_path.exists():
        return None
    try:
        gold = json.load(gold_path.open())
        test_examples = gold.get("test") or []
        return (
            (test_examples[pair_index] or {}).get("output") if test_examples else None
        )
    except Exception:
        return None


def extract_prompt(entry: Dict[str, Any]) -> str:
    for attempt_label in ("attempt_1", "attempt_2"):
        attempt = entry.get(attempt_label) or {}
        meta = attempt.get("metadata") or {}
        for choice in meta.get("choices") or []:
            message = choice.get("message") or {}
            if message.get("role") == "user" and message.get("content"):
                return str(message["content"])
    raise ValueError("No prompt found in entry metadata.")


def find_pair_index(entry: Dict[str, Any]) -> int:
    for attempt_label in ("attempt_1", "attempt_2"):
        meta = (entry.get(attempt_label) or {}).get("metadata") or {}
        pair_index = meta.get("pair_index")
        if pair_index is not None:
            try:
                return int(pair_index)
            except Exception:
                return 0
    return 0


def build_jobs(
    tasks_dir: pathlib.Path,
    temperatures: List[Tuple[str, float, int]],
    limit: Optional[int],
    task_filter: Optional[set[str]] = None,
    retry_targets: Optional[set[Tuple[str, int, str]]] = None,
) -> List[RetryJob]:
    jobs: List[RetryJob] = []
    task_ids = sorted(path.stem for path in tasks_dir.glob("*.json"))
    for task_id in task_ids:
        if task_filter and task_id not in task_filter:
            continue
        entries = load_entries(tasks_dir / f"{task_id}.json")
        for entry_idx, entry in enumerate(entries):
            try:
                prompt = extract_prompt(entry)
            except Exception as exc:
                print(f"Skipping {task_id}#{entry_idx}: {exc}", file=sys.stderr)
                continue
            pair_index = find_pair_index(entry)
            expected_output = load_gold_output(task_id, pair_index)

            for temp_label, temp_value, attempts in temperatures:
                for attempt_num in range(1, attempts + 1):
                    output_name = f"{task_id}-{temp_label}"
                    attempt_label = f"attempt_{attempt_num}"
                    if (
                        retry_targets is not None
                        and (output_name, entry_idx, attempt_label) not in retry_targets
                    ):
                        continue
                    jobs.append(
                        RetryJob(
                            task_id=task_id,
                            entry_index=entry_idx,
                            attempt_label=attempt_label,
                            pair_index=pair_index,
                            prompt=prompt,
                            expected_output=expected_output,
                            temperature=temp_value,
                            output_name=output_name,
                        )
                    )
                    if limit is not None and len(jobs) >= limit:
                        return jobs
    return jobs


def is_null_answer(attempt: Any) -> bool:
    # Only treat attempts that explicitly carry `"answer": null` as failures.
    return (
        isinstance(attempt, dict) and "answer" in attempt and attempt["answer"] is None
    )


def find_failed_attempts(output_dir: pathlib.Path) -> set[Tuple[str, int, str]]:
    failed: set[Tuple[str, int, str]] = set()
    if not output_dir.is_dir():
        return failed

    for path in sorted(output_dir.glob("*.json")):
        try:
            entries = load_entries(path)
        except Exception as exc:
            print(f"Skipping {path.name}: {exc}", file=sys.stderr)
            continue

        for entry_idx, entry in enumerate(entries):
            if not isinstance(entry, dict):
                continue
            for attempt_label, attempt in entry.items():
                if not (
                    isinstance(attempt_label, str)
                    and attempt_label.startswith("attempt_")
                ):
                    continue
                if is_null_answer(attempt):
                    failed.add((path.stem, entry_idx, attempt_label))
    return failed


def count_null_answers(output_dir: pathlib.Path) -> int:
    return len(find_failed_attempts(output_dir))


def run_single_job(
    api_key: str,
    model: str,
    provider: str,
    temperature: float,
    max_output_tokens: Optional[int],
    test_id: str,
    job: RetryJob,
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
    output_root: pathlib.Path, outcomes: List[Tuple[RetryJob, Dict[str, Any]]]
) -> None:
    per_file: Dict[str, List[Tuple[int, str, Dict[str, Any]]]] = {}
    for job, attempt in outcomes:
        per_file.setdefault(job.output_name, []).append(
            (job.entry_index, job.attempt_label, attempt)
        )

    output_root.mkdir(parents=True, exist_ok=True)
    for output_name, entries in per_file.items():
        path = output_root / f"{output_name}.json"
        try:
            existing = load_entries(path) if path.exists() else []
        except Exception as exc:
            print(f"Skipping existing contents in {path.name}: {exc}", file=sys.stderr)
            existing = []

        max_idx = max(entry_idx for entry_idx, _, _ in entries)
        while len(existing) <= max_idx:
            existing.append({})

        for entry_idx, attempt_label, attempt in sorted(entries, key=lambda p: p[0]):
            if not isinstance(existing[entry_idx], dict):
                existing[entry_idx] = {}
            existing[entry_idx][attempt_label] = attempt

        path.write_text(json.dumps(existing, indent=2))


def resolve_dir(path: str) -> pathlib.Path:
    dir_path = pathlib.Path(path)
    return dir_path if dir_path.is_absolute() else BASE_DIR / dir_path


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Retry stored prompts at multiple temperatures."
    )
    parser.add_argument(
        "--input-dir",
        default=str(TASKS_DIR.name),
        help="Folder containing the original task JSON files.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(OUTPUT_ROOT.name),
        help="Folder for the ablation outputs.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=50,
        help="How many prompts to send concurrently.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on total prompt attempts across all temperatures.",
    )
    parser.add_argument(
        "--retry-failed",
        action="store_true",
        help="Retry attempts in the output folder whose answer is null.",
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
        "--task-id",
        action="append",
        dest="task_ids",
        help="Limit processing to specific task ids. Can be repeated.",
    )
    parser.add_argument(
        "--test-id",
        default=None,
        help="test_id field for metadata. Defaults to the output folder name.",
    )
    parser.add_argument(
        "--test-mode",
        action="store_true",
        help="Shortcut for --limit 4 --batch-size 2.",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Print per-request progress."
    )
    parser.add_argument(
        "--visualize",
        "--visualise",
        dest="visualize",
        action="store_true",
        help="Open a viewer over every attempt in the output folder.",
    )
    return parser.parse_args(argv)


def build_visualization_dataset(
    output_dir: pathlib.Path, grid_to_array_fn, load_gold_fn
) -> List[Dict[str, Any]]:
    dataset: List[Dict[str, Any]] = []
    for path in sorted(output_dir.glob("*.json")):
        try:
            entries = load_entries(path)
        except Exception as exc:
            print(f"Skipping {path.name}: {exc}", file=sys.stderr)
            continue
        for entry_idx, entry in enumerate(entries):
            attempt_items = [
                (label, entry[label])
                for label in sorted(entry.keys())
                if isinstance(label, str) and label.startswith("attempt_")
            ]
            for attempt_label, attempt in attempt_items:
                attempt = attempt or {}
                answer = attempt.get("answer")
                attempt_grid = grid_to_array_fn(answer) if answer is not None else None
                meta = attempt.get("metadata") or {}
                meta_task_id = meta.get("task_id")
                # Fallback: strip the temp suffix from the filename if metadata is missing.
                fallback_task_id = path.stem.split("-temp")[0]
                task_id = meta_task_id or fallback_task_id
                kwargs_meta = meta.get("kwargs") or {}
                temperature = kwargs_meta.get("temperature")
                try:
                    pair_index = int(meta.get("pair_index", 0))
                except Exception:
                    pair_index = 0
                gold_grid = load_gold_fn(task_id, pair_index=pair_index)
                dataset.append(
                    {
                        "task_id": task_id,
                        "entry_index": entry_idx,
                        "attempt_label": attempt_label,
                        "pair_index": pair_index,
                        "attempt_grid": attempt_grid,
                        "attempt_correct": bool(attempt.get("correct")),
                        "gold_grid": gold_grid,
                        "temperature": temperature,
                    }
                )
    return dataset


def visualize_attempts(output_dir: pathlib.Path) -> int:
    try:  # Lazy import to avoid heavy deps during normal runs.
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

    dataset = build_visualization_dataset(output_dir, grid_to_array, load_gold)
    if not dataset:
        print(f"No attempts found under {output_dir}", file=sys.stderr)
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
            f"{item['task_id']} #{item['entry_index']} "
            f"{item['attempt_label']} (pair {item['pair_index']}) "
            f"{state['idx'] + 1}/{len(dataset)}",
            fontsize=14,
        )
        draw(
            axes[0],
            item["attempt_grid"],
            f"{item['attempt_label']} "
            f"({'correct' if item['attempt_correct'] else 'wrong'}) "
            f"(temp={item['temperature']})",
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


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    if args.test_mode:
        args.limit = 4
        args.batch_size = 2

    tasks_dir = resolve_dir(args.input_dir)
    output_dir = resolve_dir(args.output_dir)
    retry_targets: Optional[set[Tuple[str, int, str]]] = None

    if not tasks_dir.is_dir():
        print(f"{tasks_dir} is not a directory", file=sys.stderr)
        return 1

    if args.visualize:
        if not output_dir.is_dir():
            print(f"{output_dir} is not a directory", file=sys.stderr)
            return 1
        if not EVAL_ROOT.is_dir():
            print(f"Missing evaluation directory at {EVAL_ROOT}", file=sys.stderr)
            return 1
        return visualize_attempts(output_dir)

    if args.retry_failed and not output_dir.is_dir():
        print(f"{output_dir} is not a directory", file=sys.stderr)
        return 1

    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        print(
            f"Missing API key. Export {args.api_key_env}=<your key>.", file=sys.stderr
        )
        return 1

    task_filter = set(args.task_ids) if args.task_ids else None
    if args.retry_failed:
        retry_targets = find_failed_attempts(output_dir)
    jobs = build_jobs(
        tasks_dir,
        temperatures=TEMPERATURE_SCHEDULE,
        limit=args.limit,
        task_filter=task_filter,
        retry_targets=retry_targets,
    )
    if not jobs:
        remaining_nulls = count_null_answers(output_dir)
        if args.retry_failed:
            print(
                f"No failed attempts to retry. Remaining null answers: {remaining_nulls}",
                file=sys.stderr,
            )
            return 0
        print("No prompts to send.", file=sys.stderr)
        print(f"Remaining null answers: {remaining_nulls}", file=sys.stderr)
        return 1

    test_id = args.test_id or output_dir.name
    outcomes: List[Tuple[RetryJob, Dict[str, Any]]] = []
    total = len(jobs)
    batch_size = max(1, args.batch_size)

    for batch_idx, batch in enumerate(chunked(jobs, batch_size), start=1):
        print(f"Sending batch {batch_idx} ({len(batch)} jobs)...", file=sys.stderr)
        with ThreadPoolExecutor(max_workers=batch_size) as executor:
            futures = {
                executor.submit(
                    run_single_job,
                    api_key,
                    args.model,
                    args.provider,
                    job.temperature,
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
                            f"Job {job.task_id}#{job.entry_index} "
                            f"({job.output_name}/{job.attempt_label}) failed: {exc}",
                            file=sys.stderr,
                        )
                    attempt = {
                        "answer": None,
                        "metadata": {
                            "model": args.model,
                            "provider": args.provider,
                            "error": str(exc),
                            "pair_index": job.pair_index,
                            "task_id": job.task_id,
                            "kwargs": {"temperature": job.temperature},
                            "test_id": test_id,
                        },
                        "correct": False,
                    }
                else:
                    if args.verbose:
                        meta = attempt.get("metadata", {})
                        print(
                            f"Job {job.task_id}#{job.entry_index} ok "
                            f"({job.output_name}/{job.attempt_label}, "
                            f"pair {job.pair_index}, correct={attempt.get('correct')})",
                            file=sys.stderr,
                        )
                outcomes.append((job, attempt))
        processed = min(batch_idx * batch_size, total)
        print(f"Finished {processed}/{total}", file=sys.stderr)

    write_results(output_dir, outcomes)
    print(f"Wrote results to {output_dir}", file=sys.stderr)
    remaining_nulls = count_null_answers(output_dir)
    print(f"Remaining null answers: {remaining_nulls}", file=sys.stderr)
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    raise SystemExit(main())
