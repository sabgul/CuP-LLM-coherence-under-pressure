"""
Inspect task definition for Coherence Under Pressure.

Defines the experiment task (interrogation only, no judge).
Judging is done separately via run_judge.py.

Usage:
    # Single subject model, 5 trials, Gemini interrogator
    inspect eval cup/tasks.py \
        --model anthropic/claude-sonnet-4-6 \
        --epochs 5 \
        -T interrogator_model=google/gemini-2.5-flash-lite \
        -T num_turns=10

    # Quick smoke test (1 epoch, 5 turns)
    inspect eval cup/tasks.py \
        --model anthropic/claude-sonnet-4-6 \
        --epochs 1 \
        -T num_turns=5

    # Run a single cell (one scenario + condition)
    inspect eval cup/tasks.py \
        --model anthropic/claude-sonnet-4-6 \
        --epochs 5 \
        -T scenario=sycophantic_advisor \
        -T condition=scheming

    # View results in the Inspect web UI
    inspect view

    # Run judge on existing logs
    python run_judge.py --logs-dir ./logs

Model name format:
    Anthropic : anthropic/claude-sonnet-4-6
    OpenAI    : openai/gpt-4o
    Google    : google/gemini-2.5-flash-lite

Environment variables (in .env):
    ANTHROPIC_API_KEY   -- for Anthropic models (anthropic/ prefix)
    OPENAI_API_KEY      -- for OpenAI models (openai/ prefix)
    OPENROUTER_API_KEY  -- for any model via OpenRouter (openrouter/ prefix)
    GOOGLE_API_KEY      -- only needed for direct Google access (google/ prefix)
"""

import sys
import os

# Ensure project root is on sys.path so `cup.*` imports resolve
# when Inspect loads this file directly via `inspect eval cup/tasks.py`.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from inspect_ai import Task, task
from inspect_ai.dataset import MemoryDataset, Sample

from cup.scenarios import SCENARIOS
from cup.solver import interrogation_solver
from cup.scorer import passthrough_scorer


def build_dataset(
    interrogator_model: str,
    scenario: str | None = None,
    condition: str | None = None,
) -> MemoryDataset:
    """
    Build one Sample per (scenario x condition).
    Trials are handled by Task(epochs=N) -- Inspect reruns each sample N times.

    Args:
        interrogator_model: Provider-prefixed model for question generation.
        scenario: If set, only include this scenario (e.g. "sycophantic_advisor").
        condition: If set, only include this condition ("scheming" or "honest").
    """
    conditions = [condition] if condition else ["scheming", "honest"]
    scenarios = [s for s in SCENARIOS if s["name"] == scenario] if scenario else SCENARIOS

    if scenario and not scenarios:
        available = [s["name"] for s in SCENARIOS]
        raise ValueError(f"Unknown scenario '{scenario}'. Available: {available}")

    samples = []
    for sc in scenarios:
        for cond in conditions:
            system_prompt = (
                sc["scheming_prompt"]
                if cond == "scheming"
                else sc["honest_prompt"]
            )
            samples.append(Sample(
                input=sc["seed_question"],
                target="",
                id=f"{sc['name']}_{cond}",
                metadata={
                    "scenario": sc["name"],
                    "context": sc["context"],
                    "condition": cond,
                    "system_prompt": system_prompt,
                    "interrogator_model": interrogator_model,
                },
            ))
    return MemoryDataset(samples)


@task
def cup_eval(
    interrogator_model: str = "openrouter/google/gemini-2.5-flash-lite",
    num_turns: int = 10,
    scenario: str = "",
    condition: str = "",
) -> Task:
    """
    CuP-LLM: Coherence Under Pressure evaluation task.

    Runs the interrogation and records the turn log. No judge is applied --
    use run_judge.py to score the logs afterwards.

    Args:
        interrogator_model: Provider-prefixed model that generates follow-up questions.
        num_turns: Number of interrogation turns per session (default: 10).
        scenario: Filter to a single scenario (e.g. "sycophantic_advisor"). Empty = all.
        condition: Filter to a single condition ("scheming" or "honest"). Empty = both.
    """
    return Task(
        dataset=build_dataset(
            interrogator_model=interrogator_model,
            scenario=scenario or None,
            condition=condition or None,
        ),
        solver=interrogation_solver(num_turns=num_turns),
        scorer=passthrough_scorer(),
        metadata={
            "interrogator_model": interrogator_model,
        },
    )
