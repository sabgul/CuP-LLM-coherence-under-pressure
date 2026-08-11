"""
End-to-end runner for the Tinker experiment.

Stages (each can be run independently via flags):
  1. datagen   — generate ~40 SFT training examples
  2. finetune  — fine-tune gpt-4o-mini via OpenAI API (starts job, polls to completion)
  3. eval      — run Inspect eval for all 3 conditions (honest / prompted / trained)
  4. analyze   — generate comparison plots

Usage:
    # Full pipeline (all stages)
    python -m tinker.run_tinker

    # Skip datagen + finetune (fine-tuned model already exists in tinker/models.json)
    python -m tinker.run_tinker --eval-only

    # Only run analysis on existing logs
    python -m tinker.run_tinker --analyze-only

    # Dry run — print the inspect eval commands without running them
    python -m tinker.run_tinker --dry-run

Config (set via flags or edit defaults below):
    --base-model        Base subject model (default: openai/gpt-4o-mini)
    --interrogator      Interrogator model (default: openrouter/google/gemini-2.5-flash-lite)
    --judge             Judge model        (default: openrouter/google/gemini-2.5-flash-lite)
    --epochs            Trials per condition (default: 3)
    --turns             Interrogation turns (default: 8)
    --logs-dir          Directory for .eval log files (default: tinker/logs)
    --figures-dir       Directory for output plots   (default: tinker/figures)
"""

import argparse
import json
import os
import subprocess
import sys

MODELS_FILE = os.path.join(os.path.dirname(__file__), "models.json")
DEFAULT_LOGS_DIR = os.path.join(os.path.dirname(__file__), "logs")
DEFAULT_FIGURES_DIR = os.path.join(os.path.dirname(__file__), "figures")

SCENARIO = "sycophantic_advisor"


def load_ft_model_id() -> str:
    """Read the fine-tuned model ID from tinker/models.json."""
    if not os.path.exists(MODELS_FILE):
        print("ERROR: tinker/models.json not found.")
        print("Run datagen + finetune first, or use --eval-only after fine-tuning completes.")
        sys.exit(1)

    with open(MODELS_FILE) as f:
        models = json.load(f)

    info = models.get(SCENARIO, {})
    model_id = info.get("model_id")

    if not model_id:
        status = info.get("status", "unknown")
        job_id = info.get("job_id", "unknown")
        print(f"ERROR: No fine-tuned model ID yet (status: {status}, job: {job_id})")
        print(f"Check status with: python -m tinker.finetune_openai --status")
        sys.exit(1)

    return model_id


def run_inspect_eval(
    model: str,
    condition: str,
    interrogator: str,
    judge: str,
    epochs: int,
    turns: int,
    logs_dir: str,
    dry_run: bool = False,
) -> None:
    """Run a single `inspect eval` for one condition."""
    task_path = os.path.join(os.path.dirname(__file__), "tinker_task.py")

    cmd = [
        "inspect", "eval", task_path,
        "--model", model,
        "--epochs", str(epochs),
        "--log-dir", logs_dir,
        "-T", f"condition={condition}",
        "-T", f"interrogator_model={interrogator}",
        "-T", f"judge_model={judge}",
        "-T", f"num_turns={turns}",
    ]

    print(f"\n{'[DRY RUN] ' if dry_run else ''}Running condition: {condition}")
    print(f"  Model: {model}")
    print(f"  Command: {' '.join(cmd)}")

    if dry_run:
        return

    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        print(f"WARNING: inspect eval exited with code {result.returncode} for condition={condition}")


def stage_datagen():
    print("\n" + "=" * 60)
    print("STAGE 1: Data generation")
    print("=" * 60)
    from tinker import datagen
    datagen.main()


def stage_finetune():
    print("\n" + "=" * 60)
    print("STAGE 2: Fine-tuning")
    print("=" * 60)
    # Import and run directly (poll=True by default)
    from openai import OpenAI
    import tinker.finetune_openai as ft

    client = OpenAI()
    file_id = ft.upload_training_file(client)
    job_id = ft.start_finetune(client, file_id)
    ft.poll_until_done(client, job_id)


def stage_eval(base_model: str, interrogator: str, judge: str,
               epochs: int, turns: int, logs_dir: str,
               ft_model_id: str, dry_run: bool):
    print("\n" + "=" * 60)
    print("STAGE 3: Evaluations")
    print("=" * 60)
    os.makedirs(logs_dir, exist_ok=True)

    conditions = [
        ("honest",   base_model),
        ("prompted", base_model),
        ("trained",  f"openai/{ft_model_id}"),
    ]

    for condition, model in conditions:
        run_inspect_eval(
            model=model,
            condition=condition,
            interrogator=interrogator,
            judge=judge,
            epochs=epochs,
            turns=turns,
            logs_dir=logs_dir,
            dry_run=dry_run,
        )


def stage_analyze(logs_dir: str, figures_dir: str):
    print("\n" + "=" * 60)
    print("STAGE 4: Analysis")
    print("=" * 60)
    from tinker.analyze_tinker import run_tinker_analysis
    run_tinker_analysis(logs_dir=logs_dir, figures_dir=figures_dir)


def main():
    parser = argparse.ArgumentParser(description="Tinker experiment runner")
    parser.add_argument("--eval-only", action="store_true",
                        help="Skip datagen + finetune; read ft model ID from tinker/models.json")
    parser.add_argument("--analyze-only", action="store_true",
                        help="Only run analysis on existing logs")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print inspect eval commands without executing them")
    parser.add_argument("--base-model", default="openai/gpt-4o-mini",
                        help="Base subject model (default: openai/gpt-4o-mini)")
    parser.add_argument("--interrogator", default="openrouter/google/gemini-2.5-flash-lite",
                        help="Interrogator model")
    parser.add_argument("--judge", default="openrouter/google/gemini-2.5-flash-lite",
                        help="Judge model")
    parser.add_argument("--epochs", type=int, default=3,
                        help="Trials per condition (default: 3)")
    parser.add_argument("--turns", type=int, default=8,
                        help="Interrogation turns per session (default: 8)")
    parser.add_argument("--logs-dir", default=DEFAULT_LOGS_DIR)
    parser.add_argument("--figures-dir", default=DEFAULT_FIGURES_DIR)
    args = parser.parse_args()

    print("Tinker Experiment: Prompted vs. Trained Misalignment")
    print(f"  Base model:    {args.base_model}")
    print(f"  Interrogator:  {args.interrogator}")
    print(f"  Judge:         {args.judge}")
    print(f"  Epochs/trials: {args.epochs}")
    print(f"  Turns:         {args.turns}")
    print(f"  Logs dir:      {args.logs_dir}")

    if args.analyze_only:
        stage_analyze(args.logs_dir, args.figures_dir)
        return

    if not args.eval_only:
        stage_datagen()
        stage_finetune()

    ft_model_id = load_ft_model_id()
    print(f"\nFine-tuned model: {ft_model_id}")

    stage_eval(
        base_model=args.base_model,
        interrogator=args.interrogator,
        judge=args.judge,
        epochs=args.epochs,
        turns=args.turns,
        logs_dir=args.logs_dir,
        ft_model_id=ft_model_id,
        dry_run=args.dry_run,
    )

    if not args.dry_run:
        stage_analyze(args.logs_dir, args.figures_dir)


if __name__ == "__main__":
    main()
