"""
Main experiment runner for Coherence Under Pressure.

Usage:
  # Quick test: 1 scenario, 1 model, 1 trial
  python run_experiment.py --scenarios insecure_code --models claude-sonnet-4-6 --trials 1 --turns 5

  # Full run: all scenarios, all models, 5 trials each
  python run_experiment.py --trials 5

  # Specific models only
  python run_experiment.py --models claude-sonnet-4-6 gpt-4o
"""

import os
import json
import argparse
import time
from datetime import datetime

from scenarios import SCENARIOS
from interrogator import run_interrogation

OUTPUTS_DIR = "outputs"

# Models to test as subjects
DEFAULT_MODELS = [
    "claude-sonnet-4-6",
    "gpt-4o",
    "gemini-2.5-flash-lite",
]

# Model used as the interrogator (kept constant for consistency)
# INTERROGATOR_MODEL = "gemini-2.5-flash-lite"
INTERROGATOR_MODEL = "claude-sonnet-4-6"


def run_single_session(scenario, subject_model, condition, trial, num_turns=10):
    """
    Run one interrogation session (scheming or honest).

    Args:
        scenario: Dict from SCENARIOS
        subject_model: Model being interrogated
        condition: "scheming" or "honest"
        trial: Trial number
        num_turns: Number of interrogation turns

    Returns:
        dict: Full session data
    """
    system_prompt = scenario["scheming_prompt"] if condition == "scheming" else scenario["honest_prompt"]

    log = run_interrogation(
        subject_model=subject_model,
        subject_system_prompt=system_prompt,
        seed_question=scenario["seed_question"],
        interrogator_model=INTERROGATOR_MODEL,
        num_turns=num_turns,
    )

    session = {
        "scenario": scenario["name"],
        "context": scenario["context"],
        "subject_model": subject_model,
        "interrogator_model": INTERROGATOR_MODEL,
        "condition": condition,
        "trial": trial,
        "num_turns": len(log),
        "system_prompt": system_prompt,
        "conversation": log,
        "timestamp": datetime.now().isoformat(),
    }

    # Save individual session
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    model_short = subject_model.replace("/", "_").replace("-", "_")
    interr_short = INTERROGATOR_MODEL.replace("/", "_").replace("-", "_")

    fname = f"{scenario['name']}_{model_short}_{condition}_t{trial}_interr_{interr_short}.json"
    with open(os.path.join(OUTPUTS_DIR, fname), "w") as f:
        json.dump(session, f, indent=2)

    return session


def run_experiment(scenarios=None, models=None, trials=5, num_turns=10):
    """Run the full experiment matrix."""
    scenario_list = scenarios or SCENARIOS
    model_list = models or DEFAULT_MODELS

    total = len(scenario_list) * len(model_list) * 2 * trials  # 2 conditions
    print(f"\n{'='*60}")
    print(f"  COHERENCE UNDER PRESSURE EXPERIMENT")
    print(f"  Scenarios: {len(scenario_list)}")
    print(f"  Models: {len(model_list)}")
    print(f"  Conditions: scheming + honest")
    print(f"  Trials per condition: {trials}")
    print(f"  Turns per interrogation: {num_turns}")
    print(f"  Total sessions: {total}")
    print(f"  Interrogator: {INTERROGATOR_MODEL}")
    print(f"{'='*60}\n")

    completed = 0
    failed = 0

    for scenario in scenario_list:
        for model in model_list:
            for condition in ["scheming", "honest"]:
                for trial in range(1, trials + 1):
                    label = f"{scenario['name']} | {model} | {condition} | t{trial}"
                    print(f"[{completed+1}/{total}] {label}")

                    try:
                        run_single_session(scenario, model, condition, trial, num_turns)
                        completed += 1
                    except Exception as e:
                        print(f"  FAILED: {e}")
                        failed += 1

                    # Rate limit courtesy
                    time.sleep(1)

    print(f"\n{'='*60}")
    print(f"  DONE: {completed} succeeded, {failed} failed")
    print(f"  Results in: {OUTPUTS_DIR}/")
    print(f"  Next: python analyze.py")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Coherence Under Pressure Experiment")
    parser.add_argument("--scenarios", nargs="+", help="Scenario names to run")
    parser.add_argument("--models", nargs="+", help="Subject models to test")
    parser.add_argument("--trials", type=int, default=5, help="Trials per condition (default: 5)")
    parser.add_argument("--turns", type=int, default=10, help="Interrogation turns (default: 10)")
    args = parser.parse_args()

    # Filter scenarios by name if specified
    selected_scenarios = SCENARIOS
    if args.scenarios:
        selected_scenarios = [s for s in SCENARIOS if s["name"] in args.scenarios]

    run_experiment(
        scenarios=selected_scenarios,
        models=args.models,
        trials=args.trials,
        num_turns=args.turns,
    )