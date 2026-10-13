"""
Export Inspect .eval logs to the legacy per-trial JSON format.

Produces one JSON file per (scenario, subject_model, condition, trial),
matching the structure used in data/outputs-gemini-interrogator-115-trials/.

Usage:
    python export_logs.py --logs-dir ./logs/gemini-interrogator --out-dir ./data/outputs-gemini-interrogator-115-trials
"""

import argparse
import glob
import json
import os
from datetime import datetime, timezone

from inspect_ai.log import read_eval_log


def _model_short_name(model: str) -> str:
    """Strip provider prefix for filenames: openrouter/anthropic/claude-sonnet-4-6 -> claude_sonnet_4_6"""
    # Take the last segment (model name) and replace hyphens with underscores (keep dots)
    name = model.split("/")[-1]
    return name.replace("-", "_")


def export_logs(logs_dir: str, out_dir: str) -> None:
    log_paths = sorted(glob.glob(os.path.join(logs_dir, "**", "*.eval"), recursive=True))
    if not log_paths:
        log_paths = sorted(glob.glob(os.path.join(logs_dir, "*.eval")))

    if not log_paths:
        print(f"No .eval log files found in {logs_dir}")
        return

    os.makedirs(out_dir, exist_ok=True)
    exported = 0

    for log_path in log_paths:
        log = read_eval_log(log_path)
        if log.status != "success" or not log.samples:
            print(f"  Skipping {os.path.basename(log_path)} (status: {log.status})")
            continue

        subject_model = log.eval.model
        subject_short = _model_short_name(subject_model)
        task_args = log.eval.task_args or {}
        interrogator_model = task_args.get("interrogator_model", "unknown")
        interrogator_short = _model_short_name(interrogator_model)

        for sample in log.samples:
            meta = sample.metadata or {}
            scenario = meta.get("scenario", "unknown")
            condition = meta.get("condition", "unknown")
            system_prompt = meta.get("system_prompt", "")
            context = meta.get("context", "")
            trial = sample.epoch

            # Get turn_log from score metadata or sample metadata
            turn_log = meta.get("turn_log", [])
            if not turn_log:
                score_obj = (
                    sample.scores.get("passthrough_scorer")
                    or sample.scores.get("coherence_scorer")
                )
                if score_obj and score_obj.metadata:
                    turn_log = score_obj.metadata.get("turn_log", [])

            if not turn_log:
                print(f"  Skipping {scenario}/{condition}/t{trial} — no turn log")
                continue

            record = {
                "scenario": scenario,
                "context": context,
                "subject_model": subject_model.split("/")[-1],
                "interrogator_model": interrogator_model.split("/")[-1],
                "condition": condition,
                "trial": trial,
                "num_turns": len(turn_log),
                "system_prompt": system_prompt,
                "conversation": turn_log,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

            fname = f"{scenario}_{subject_short}_{condition}_t{trial}_interr_{interrogator_short}.json"
            out_path = os.path.join(out_dir, fname)
            with open(out_path, "w") as f:
                json.dump(record, f, indent=2)
            exported += 1

    print(f"Exported {exported} trial(s) to {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export .eval logs to per-trial JSON")
    parser.add_argument("--logs-dir", required=True, help="Directory containing .eval logs")
    parser.add_argument("--out-dir", required=True, help="Output directory for JSON files")
    args = parser.parse_args()

    export_logs(logs_dir=args.logs_dir, out_dir=args.out_dir)
