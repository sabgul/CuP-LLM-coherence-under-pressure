"""
Run the CuP-LLM experiment matrix (interrogation only, no judge).

Wraps `inspect eval` in Python so you can define experiment configs as
data and run them programmatically. Judging is done separately via
run_judge.py.

Usage:
    python run_experiment.py                            # full matrix, .eval logs only
    python run_experiment.py --export-json              # also export per-trial JSON
    python run_experiment.py --dry-run                  # print what would run
    python run_experiment.py --epochs 1 --turns 5       # quick smoke test
"""

import argparse
from dataclasses import dataclass

from inspect_ai import eval as inspect_eval

from cup.tasks import cup_eval

# -- Experiment matrix --------------------------------------------------------

SUBJECT_MODELS = [
    "openrouter/anthropic/claude-sonnet-4-6",
    "openrouter/openai/gpt-4o",
    "openrouter/google/gemini-2.5-flash-lite",
]

@dataclass
class ExperimentConfig:
    label: str
    interrogator_model: str

EXPERIMENTS = [
    ExperimentConfig(
        label="gemini-interrogator",
        interrogator_model="openrouter/google/gemini-2.5-flash-lite",
    ),
    ExperimentConfig(
        label="claude-interrogator",
        interrogator_model="openrouter/anthropic/claude-sonnet-4-6",
    ),
]

# -- Runner -------------------------------------------------------------------

def run(
    epochs: int = 5,
    turns: int = 10,
    export_json: bool = False,
    export_dir: str | None = None,
    dry_run: bool = False,
) -> None:
    for config in EXPERIMENTS:
        log_dir = f"./logs/{config.label}"

        print(f"\n{'='*60}")
        print(f"  Experiment: {config.label}")
        print(f"  Subject models: {SUBJECT_MODELS}")
        print(f"  Interrogator: {config.interrogator_model}")
        print(f"  Epochs: {epochs}  Turns: {turns}")
        print(f"  Log dir: {log_dir}")
        if export_json:
            json_dir = export_dir or f"./data/{config.label}"
            print(f"  Export JSON: {json_dir}")
        print(f"{'='*60}")

        if dry_run:
            print("  [dry-run] skipping.")
            continue

        inspect_eval(
            cup_eval(
                interrogator_model=config.interrogator_model,
                num_turns=turns,
            ),
            model=SUBJECT_MODELS,
            epochs=epochs,
            log_dir=log_dir,
            max_tasks=3,
        )

        if export_json:
            from export_logs import export_logs
            json_dir = export_dir or f"./data/{config.label}"
            export_logs(logs_dir=log_dir, out_dir=json_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run CuP-LLM experiment matrix")
    parser.add_argument("--epochs", type=int, default=5, help="Trials per sample (default: 5)")
    parser.add_argument("--turns", type=int, default=10, help="Interrogation turns (default: 10)")
    parser.add_argument("--export-json", action="store_true",
                        help="Also export per-trial JSON files after each run")
    parser.add_argument("--export-dir", type=str, default=None,
                        help="Directory for JSON export (default: ./data/<experiment-label>)")
    parser.add_argument("--dry-run", action="store_true", help="Print plan without running")
    args = parser.parse_args()

    run(
        epochs=args.epochs,
        turns=args.turns,
        export_json=args.export_json,
        export_dir=args.export_dir,
        dry_run=args.dry_run,
    )
