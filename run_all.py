"""
Run the full CuP-LLM experiment matrix.

Wraps `inspect eval` in Python so you can define experiment configs as
data and run them programmatically — useful for sweeps, ablations, or
just keeping a record of exactly what was run and why.

Usage:
    python run_all.py                     # full matrix (default config)
    python run_all.py --dry-run           # print what would run, don't execute
    python run_all.py --epochs 1 --turns 5   # quick smoke test
"""

import argparse
from dataclasses import dataclass

from inspect_ai import eval as inspect_eval

from cup_tasks import cup_eval

# ── Experiment matrix ─────────────────────────────────────────────────────────

SUBJECT_MODELS = [
    "anthropic/claude-sonnet-4-6",
    "openai/gpt-4o",
    "openrouter/google/gemini-2.5-flash-lite",
]

# Each config is one `inspect eval` call (one log file).
# Running separate calls per interrogator keeps logs cleanly separated.
@dataclass
class ExperimentConfig:
    label: str
    interrogator_model: str
    judge_model: str

EXPERIMENTS = [
    ExperimentConfig(
        label="gemini-interrogator",
        interrogator_model="openrouter/google/gemini-2.5-flash-lite",
        judge_model="openrouter/google/gemini-2.5-flash-lite",
    ),
    ExperimentConfig(
        label="claude-interrogator",
        interrogator_model="openrouter/anthropic/claude-sonnet-4-6",
        judge_model="openrouter/google/gemini-2.5-flash-lite",
    ),
]

# ── Runner ────────────────────────────────────────────────────────────────────

def run(epochs: int = 5, turns: int = 10, dry_run: bool = False) -> None:
    for config in EXPERIMENTS:
        print(f"\n{'='*60}")
        print(f"  Experiment: {config.label}")
        print(f"  Subject models: {SUBJECT_MODELS}")
        print(f"  Interrogator: {config.interrogator_model}")
        print(f"  Judge: {config.judge_model}")
        print(f"  Epochs: {epochs}  Turns: {turns}")
        print(f"{'='*60}")

        if dry_run:
            print("  [dry-run] skipping.")
            continue

        inspect_eval(
            cup_eval(
                interrogator_model=config.interrogator_model,
                judge_model=config.judge_model,
                num_turns=turns,
            ),
            model=SUBJECT_MODELS,
            epochs=epochs,
            log_dir=f"./logs/{config.label}",
            # Run up to 3 models in parallel (one per subject model)
            max_tasks=3,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run CuP-LLM experiment matrix")
    parser.add_argument("--epochs", type=int, default=5, help="Trials per sample (default: 5)")
    parser.add_argument("--turns", type=int, default=10, help="Interrogation turns (default: 10)")
    parser.add_argument("--dry-run", action="store_true", help="Print plan without running")
    args = parser.parse_args()

    run(epochs=args.epochs, turns=args.turns, dry_run=args.dry_run)
