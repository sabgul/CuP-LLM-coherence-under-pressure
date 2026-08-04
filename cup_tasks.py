"""
Inspect task definition for Coherence Under Pressure.

This replaces run_experiment.py. The experiment is now driven entirely
by `inspect eval` — no manual orchestration loop needed.

Usage:
    # Single subject model, 5 trials, Gemini interrogator + judge
    inspect eval cup_tasks.py \\
        --model anthropic/claude-sonnet-4-6 \\
        --epochs 5 \\
        -T interrogator_model=google/gemini-2.5-flash-lite \\
        -T judge_model=google/gemini-2.5-flash-lite \\
        -T num_turns=10

    # Quick smoke test (1 epoch, 5 turns)
    inspect eval cup_tasks.py \\
        --model anthropic/claude-sonnet-4-6 \\
        --epochs 1 \\
        -T num_turns=5

    # View results in the Inspect web UI
    inspect view

    # Generate plots from logged results
    python analyze.py --plot --logs-dir ./logs

Model name format:
    Anthropic : anthropic/claude-sonnet-4-6
    OpenAI    : openai/gpt-4o
    Google    : google/gemini-2.5-flash-lite

Environment variables (in .env):
    ANTHROPIC_API_KEY   — for Anthropic models (anthropic/ prefix)
    OPENAI_API_KEY      — for OpenAI models (openai/ prefix)
    OPENROUTER_API_KEY  — for any model via OpenRouter (openrouter/ prefix)
    GOOGLE_API_KEY      — only needed for direct Google access (google/ prefix)
"""

from inspect_ai import Task, task
from inspect_ai.dataset import MemoryDataset, Sample

from scenarios import SCENARIOS
from cup_solver import interrogation_solver
from cup_scorer import coherence_scorer


def build_dataset(interrogator_model: str) -> MemoryDataset:
    """
    Build one Sample per (scenario × condition).
    Trials are handled by Task(epochs=N) — Inspect reruns each sample N times.
    """
    samples = []
    for scenario in SCENARIOS:
        for condition in ["scheming", "honest"]:
            system_prompt = (
                scenario["scheming_prompt"]
                if condition == "scheming"
                else scenario["honest_prompt"]
            )
            samples.append(Sample(
                input=scenario["seed_question"],
                target="",
                id=f"{scenario['name']}_{condition}",
                metadata={
                    "scenario": scenario["name"],
                    "context": scenario["context"],
                    "condition": condition,
                    "system_prompt": system_prompt,
                    "interrogator_model": interrogator_model,
                },
            ))
    return MemoryDataset(samples)


@task
def cup_eval(
    interrogator_model: str = "openrouter/google/gemini-2.5-flash-lite",
    judge_model: str = "openrouter/google/gemini-2.5-flash-lite",
    num_turns: int = 10,
) -> Task:
    """
    CuP-LLM: Coherence Under Pressure evaluation task.

    Args:
        interrogator_model: Provider-prefixed model that generates follow-up questions.
        judge_model: Provider-prefixed model that scores coherence dimensions.
        num_turns: Number of interrogation turns per session (default: 10).
    """
    return Task(
        dataset=build_dataset(interrogator_model=interrogator_model),
        solver=interrogation_solver(num_turns=num_turns),
        scorer=coherence_scorer(judge_model=judge_model),
        metadata={
            "interrogator_model": interrogator_model,
            "judge_model": judge_model,
        },
    )
