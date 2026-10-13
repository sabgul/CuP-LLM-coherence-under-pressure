"""
Run judge scoring on existing experiment data.

Reads experiment transcripts (JSON or .eval format), scores them with a
judge model, and writes results as a single CSV per judge run.

Each judge run produces its own CSV — running multiple judges on the
same data gives you side-by-side comparison without overwriting anything.

Usage:
    # Score JSON files (auto-detected)
    python run_judge.py --input-dir ./data/outputs-gemini-interrogator-full

    # Score .eval logs (auto-detected)
    python run_judge.py --input-dir ./logs/gemini-interrogator

    # Explicit format
    python run_judge.py --input-dir ./data/my-data --input-format json

    # Custom judge model
    python run_judge.py --input-dir ./data/outputs-gemini-interrogator-full \
        --judge openrouter/anthropic/claude-sonnet-4-6

    # Name the experiment explicitly
    python run_judge.py --input-dir ./data/outputs-gemini-interrogator-full \
        --experiment-name gemini-interr-v1

    # Run multiple judges on the same data
    python run_judge.py --input-dir ./data/outputs-gemini-interrogator-full \
        --judge openrouter/google/gemini-2.5-flash-lite
    python run_judge.py --input-dir ./data/outputs-gemini-interrogator-full \
        --judge openrouter/anthropic/claude-sonnet-4-6

Output structure:
    judge-results/
    └── <experiment-name>/
        ├── claude-sonnet-4-6.csv
        └── gemini-2.5-flash-lite.csv

CSV columns:
    subject_model, interrogator_model, judge_model, scenario, condition,
    trial, turn, internal_consistency, reasoning_quality, specificity,
    defensiveness, coherence_overall
"""

import argparse
import asyncio
import csv
import glob
import json
import os
from dotenv import load_dotenv

load_dotenv()

from inspect_ai.model import get_model

from cup.scorer import (
    _score_structured,
    _score_with_parsing,
    _needs_parsing_fallback,
)


# -- Terminal colors -----------------------------------------------------------

class C:
    GREEN = "\033[32m"
    RED = "\033[31m"
    YELLOW = "\033[33m"
    DIM = "\033[2m"
    RESET = "\033[0m"


# -- Transcript loading -------------------------------------------------------

def _detect_format(input_dir: str) -> str:
    """Auto-detect input format based on file extensions."""
    json_files = glob.glob(os.path.join(input_dir, "*.json"))
    eval_files = glob.glob(os.path.join(input_dir, "*.eval"))
    if not eval_files:
        eval_files = glob.glob(os.path.join(input_dir, "**", "*.eval"), recursive=True)

    if json_files and not eval_files:
        return "json"
    if eval_files and not json_files:
        return "eval"
    if json_files and eval_files:
        return "json"

    print(f"{C.RED}No .json or .eval files found in {input_dir}{C.RESET}")
    return "none"


def load_transcripts_json(input_dir: str) -> list[dict]:
    """Load transcripts from legacy per-trial JSON files."""
    transcripts = []
    json_files = sorted(glob.glob(os.path.join(input_dir, "*.json")))

    for path in json_files:
        with open(path) as f:
            data = json.load(f)

        conversation = data.get("conversation", [])
        if not conversation:
            continue

        transcripts.append({
            "scenario": data.get("scenario", "unknown"),
            "subject_model": data.get("subject_model", "unknown"),
            "interrogator_model": data.get("interrogator_model", "unknown"),
            "condition": data.get("condition", "unknown"),
            "trial": data.get("trial", 0),
            "conversation": conversation,
        })

    return transcripts


def load_transcripts_eval(input_dir: str) -> list[dict]:
    """Load transcripts from Inspect .eval log files."""
    from inspect_ai.log import read_eval_log

    transcripts = []
    eval_files = sorted(glob.glob(os.path.join(input_dir, "**", "*.eval"), recursive=True))
    if not eval_files:
        eval_files = sorted(glob.glob(os.path.join(input_dir, "*.eval")))

    for log_path in eval_files:
        log = read_eval_log(log_path)
        if log.status != "success" or not log.samples:
            continue

        subject_model = log.eval.model
        task_args = log.eval.task_args or {}
        interrogator_model = task_args.get("interrogator_model", "unknown")

        for sample in log.samples:
            meta = sample.metadata or {}
            turn_log = meta.get("turn_log", [])

            if not turn_log:
                score_obj = (
                    sample.scores.get("passthrough_scorer")
                    or sample.scores.get("coherence_scorer")
                )
                if score_obj and score_obj.metadata:
                    turn_log = score_obj.metadata.get("turn_log", [])

            if not turn_log:
                continue

            transcripts.append({
                "scenario": meta.get("scenario", "unknown"),
                "subject_model": subject_model.split("/")[-1],
                "interrogator_model": interrogator_model.split("/")[-1],
                "condition": meta.get("condition", "unknown"),
                "trial": sample.epoch,
                "conversation": turn_log,
            })

    return transcripts


# -- Scoring ------------------------------------------------------------------

def _format_transcript(conversation: list[dict]) -> str:
    """Format a conversation into the transcript string the judge expects."""
    transcript = ""
    for entry in conversation:
        turn = entry.get("turn", 0)
        question = entry.get("question", "")
        answer = entry.get("answer", "")
        transcript += f"\n--- TURN {turn} ---\n"
        transcript += f"QUESTION: {question}\n\n"
        transcript += f"ANSWER: {answer}\n"
    return transcript


TIMEOUT_SECONDS = 120


async def score_one(judge, transcript: str, use_structured: bool) -> list[dict] | str:
    """Score a single transcript. Returns list of turn dicts or error string."""
    try:
        if use_structured:
            turns_data, raw = await asyncio.wait_for(
                _score_structured(judge, transcript), timeout=TIMEOUT_SECONDS
            )
        else:
            turns_data, raw = await asyncio.wait_for(
                _score_with_parsing(judge, transcript), timeout=TIMEOUT_SECONDS
            )
    except asyncio.TimeoutError:
        return "timeout"
    except Exception as e:
        return f"{type(e).__name__}: {e}"

    if turns_data is None:
        return "parse error"

    return [t.model_dump() for t in turns_data]


CSV_COLUMNS = [
    "subject_model", "interrogator_model", "judge_model",
    "scenario", "condition", "trial", "turn",
    "internal_consistency", "reasoning_quality", "specificity",
    "defensiveness", "coherence_overall",
]


async def score_all(
    transcripts: list[dict],
    judge_model: str,
    out_path: str,
    dry_run: bool = False,
) -> None:
    """Score all transcripts and append rows to a CSV."""
    judge_short = judge_model.split("/")[-1]

    if dry_run:
        for i, t in enumerate(transcripts, 1):
            label = f"{t['scenario']}/{t['condition']}/t{t['trial']}"
            print(f"  {C.DIM}[{i}/{len(transcripts)}]{C.RESET} {label} {C.DIM}[dry-run]{C.RESET}")
        return

    # Load already-scored keys to support resuming
    scored_keys: set[tuple] = set()
    if os.path.exists(out_path):
        with open(out_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                key = (row["scenario"], row["condition"], str(row["trial"]))
                scored_keys.add(key)

    judge = get_model(judge_model)
    use_structured = not _needs_parsing_fallback(judge_model)

    scored = 0
    skipped = 0
    failed = 0
    errors: list[str] = []

    # Open CSV in append mode; write header only if file is new
    write_header = not os.path.exists(out_path)
    csvfile = open(out_path, "a", newline="")
    writer = csv.DictWriter(csvfile, fieldnames=CSV_COLUMNS)
    if write_header:
        writer.writeheader()

    try:
        for i, t in enumerate(transcripts, 1):
            label = f"{t['scenario']}/{t['condition']}/t{t['trial']}"
            prefix = f"  {C.DIM}[{i}/{len(transcripts)}]{C.RESET} {label}..."

            # Skip if already in CSV
            key = (t["scenario"], t["condition"], str(t["trial"]))
            if key in scored_keys:
                print(f"{prefix} {C.DIM}skip (already scored){C.RESET}")
                skipped += 1
                continue

            print(f"{prefix}", end=" ", flush=True)

            transcript_text = _format_transcript(t["conversation"])
            result = await score_one(judge, transcript_text, use_structured)

            if isinstance(result, str):
                print(f"{C.RED}FAILED ({result}){C.RESET}")
                errors.append(f"  {label}: {result}")
                failed += 1
                continue

            # Write one row per turn
            for turn_score in result:
                writer.writerow({
                    "subject_model": t["subject_model"],
                    "interrogator_model": t["interrogator_model"],
                    "judge_model": judge_short,
                    "scenario": t["scenario"],
                    "condition": t["condition"],
                    "trial": t["trial"],
                    "turn": turn_score["turn"],
                    "internal_consistency": turn_score["internal_consistency"],
                    "reasoning_quality": turn_score["reasoning_quality"],
                    "specificity": turn_score["specificity"],
                    "defensiveness": turn_score["defensiveness"],
                    "coherence_overall": turn_score["coherence_overall"],
                })
            csvfile.flush()

            overall = [s["coherence_overall"] for s in result]
            mean_val = sum(overall) / len(overall)
            print(f"{C.GREEN}ok{C.RESET} (mean={mean_val:.1f})")
            scored += 1

    finally:
        csvfile.close()

    # Summary
    print()
    print(f"  {C.GREEN}Scored: {scored}{C.RESET}", end="")
    if skipped:
        print(f"  {C.DIM}Skipped: {skipped}{C.RESET}", end="")
    if failed:
        print(f"  {C.RED}Failed: {failed}{C.RESET}", end="")
    print(f"\n  Output: {out_path}")

    if errors:
        print(f"\n  {C.RED}Errors:{C.RESET}")
        for e in errors:
            print(f"  {C.RED}{e}{C.RESET}")


# -- Main ---------------------------------------------------------------------

def run_judge(
    input_dir: str,
    judge_model: str,
    out_dir: str | None = None,
    experiment_name: str | None = None,
    input_format: str = "auto",
    dry_run: bool = False,
) -> None:
    if input_format == "auto":
        input_format = _detect_format(input_dir)
    if input_format == "none":
        return

    print(f"  Input:  {input_dir} ({input_format})")
    print(f"  Judge:  {judge_model}")

    if input_format == "json":
        transcripts = load_transcripts_json(input_dir)
    else:
        transcripts = load_transcripts_eval(input_dir)

    if not transcripts:
        print(f"{C.RED}No transcripts found.{C.RESET}")
        return

    print(f"  Found:  {len(transcripts)} transcript(s)")

    # Resolve output: judge-results/<experiment>/<judge>.csv
    if out_dir is None:
        if experiment_name is None:
            experiment_name = os.path.basename(os.path.normpath(input_dir))
        out_dir = os.path.join("judge-results", experiment_name)

    os.makedirs(out_dir, exist_ok=True)
    judge_short = judge_model.split("/")[-1]
    out_path = os.path.join(out_dir, f"{judge_short}.csv")
    print(f"  Output: {out_path}")
    print()

    asyncio.run(score_all(transcripts, judge_model, out_path, dry_run))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run judge scoring on experiment data")
    parser.add_argument("--input-dir", type=str, required=True,
                        help="Directory containing experiment data (.json or .eval)")
    parser.add_argument("--judge", type=str, default="openrouter/google/gemini-2.5-flash-lite",
                        help="Judge model (default: openrouter/google/gemini-2.5-flash-lite)")
    parser.add_argument("--experiment-name", type=str, default=None,
                        help="Name for this experiment (default: input dir basename)")
    parser.add_argument("--out-dir", type=str, default=None,
                        help="Override output directory (default: judge-results/<experiment>/)")
    parser.add_argument("--input-format", choices=["auto", "json", "eval"], default="auto",
                        help="Input format (default: auto-detect)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would be scored without running")
    args = parser.parse_args()

    run_judge(
        input_dir=args.input_dir,
        judge_model=args.judge,
        out_dir=args.out_dir,
        experiment_name=args.experiment_name,
        input_format=args.input_format,
        dry_run=args.dry_run,
    )
