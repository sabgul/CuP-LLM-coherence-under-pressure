"""
Analysis script for Coherence Under Pressure experiments.

Reads Inspect eval logs produced by `inspect eval cup_tasks.py` and
generates plots comparing scheming vs honest coherence trajectories.

Usage:
    # Generate plots from Inspect logs (default logs/ directory)
    python analyze.py --plot

    # Specify a custom logs directory
    python analyze.py --plot --logs-dir ./logs

    # Plot from a legacy CSV (pre-Inspect data)
    python analyze.py --plot --csv coherence_scores_all.csv
"""

import os
import glob
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

INSPECT_LOGS_DIR = "logs"
FIGURES_DIR = "figures"

sns.set_theme(style="whitegrid", font_scale=1.1)
DPI = 200


# ============================================================
# DATA LOADING
# ============================================================

def load_from_inspect_logs(logs_dir: str = INSPECT_LOGS_DIR) -> pd.DataFrame:
    """
    Read Inspect .eval log files and return a DataFrame with one row per
    (sample × turn), matching the schema previously written to CSV.

    Columns:
        log_file, judge_model, scenario, subject_model, interrogator_model,
        condition, trial, turn, internal_consistency, reasoning_quality,
        specificity, defensiveness, coherence_overall
    """
    from inspect_ai.log import read_eval_log

    rows = []
    log_paths = sorted(glob.glob(os.path.join(logs_dir, "*.eval")))

    if not log_paths:
        print(f"No .eval log files found in {logs_dir}/")
        return pd.DataFrame()

    print(f"Found {len(log_paths)} log file(s) in {logs_dir}/")

    for log_path in log_paths:
        log = read_eval_log(log_path)

        if log.status != "success":
            print(f"  Skipping {os.path.basename(log_path)} (status: {log.status})")
            continue

        if not log.samples:
            continue

        subject_model = log.eval.model
        task_args = log.eval.task_args or {}
        interrogator_model = task_args.get("interrogator_model", "unknown")

        for sample in log.samples:
            # Support both scorer names: coherence_scorer (judged) and passthrough_scorer (raw)
            score_obj = sample.scores.get("coherence_scorer") or sample.scores.get("passthrough_scorer")
            if score_obj is None:
                continue

            meta = score_obj.metadata or {}
            if meta.get("error"):
                continue

            turns_data = meta.get("turns", [])
            judge_model = meta.get("judge_model", "unknown")
            scenario = meta.get("scenario") or sample.metadata.get("scenario", "unknown")
            condition = meta.get("condition") or sample.metadata.get("condition", "unknown")
            trial = sample.epoch  # 1-indexed epoch = trial number

            for turn_data in turns_data:
                if not turn_data:
                    continue
                rows.append({
                    "log_file": os.path.basename(log_path),
                    "judge_model": judge_model,
                    "scenario": scenario,
                    "subject_model": subject_model,
                    "interrogator_model": interrogator_model,
                    "condition": condition,
                    "trial": trial,
                    "turn": turn_data.get("turn", 0),
                    "internal_consistency": turn_data.get("internal_consistency", -1),
                    "reasoning_quality": turn_data.get("reasoning_quality", -1),
                    "specificity": turn_data.get("specificity", -1),
                    "defensiveness": turn_data.get("defensiveness", -1),
                    "coherence_overall": turn_data.get("coherence_overall", -1),
                })

    df = pd.DataFrame(rows)
    if not df.empty:
        # Filter out any rows where scoring failed (sentinel -1)
        df = df[df["coherence_overall"] >= 0]
    return df


# ============================================================
# PLOTTING
# ============================================================

def plot_coherence_decay(df, outdir):
    """Main plot: coherence over turns, scheming vs honest, per scenario."""
    for scenario in df["scenario"].unique():
        sdf = df[df["scenario"] == scenario]

        for judge in sdf["judge_model"].unique():
            jdf = sdf[sdf["judge_model"] == judge]

            fig, ax = plt.subplots(figsize=(10, 6))

            for condition, color, style in [("honest", "#2ca02c", "-"), ("scheming", "#d62728", "--")]:
                cdf = jdf[jdf["condition"] == condition]
                if cdf.empty:
                    continue

                grouped = cdf.groupby("turn")["coherence_overall"].agg(["mean", "std"]).reset_index()

                ax.plot(grouped["turn"], grouped["mean"], f'{style}o', color=color,
                        linewidth=2, markersize=6, label=f"{condition} (μ)")
                ax.fill_between(grouped["turn"],
                                grouped["mean"] - grouped["std"],
                                grouped["mean"] + grouped["std"],
                                alpha=0.15, color=color)

            ax.set_xlabel("Interrogation Turn", fontsize=12)
            ax.set_ylabel("Overall Coherence Score", fontsize=12)
            ax.set_title(f"Coherence Decay: {scenario}\nJudge: {judge}", fontsize=13, fontweight="bold")
            ax.set_ylim(0, 10.5)
            ax.set_xlim(0.5, jdf["turn"].max() + 0.5)
            ax.legend(fontsize=11)
            ax.grid(True, alpha=0.3)

            plt.tight_layout()
            fname = f"decay_{scenario}_{judge.replace('/', '_')}.png"
            fig.savefig(os.path.join(outdir, fname), dpi=DPI, bbox_inches="tight")
            plt.close()
            print(f"  Saved: {fname}")


def plot_coherence_decay_by_model(df, outdir):
    """Coherence decay split by subject model."""
    for scenario in df["scenario"].unique():
        sdf = df[df["scenario"] == scenario]

        for judge in sdf["judge_model"].unique():
            jdf = sdf[sdf["judge_model"] == judge]
            models = jdf["subject_model"].unique()

            if len(models) < 2:
                continue

            fig, axes = plt.subplots(1, len(models), figsize=(6 * len(models), 5), sharey=True)
            if len(models) == 1:
                axes = [axes]

            for ax, model in zip(axes, sorted(models)):
                mdf = jdf[jdf["subject_model"] == model]

                for condition, color, style in [("honest", "#2ca02c", "-"), ("scheming", "#d62728", "--")]:
                    cdf = mdf[mdf["condition"] == condition]
                    if cdf.empty:
                        continue
                    grouped = cdf.groupby("turn")["coherence_overall"].agg(["mean", "std"]).reset_index()
                    ax.plot(grouped["turn"], grouped["mean"], f'{style}o', color=color,
                            linewidth=2, markersize=5, label=condition)
                    ax.fill_between(grouped["turn"],
                                    grouped["mean"] - grouped["std"],
                                    grouped["mean"] + grouped["std"],
                                    alpha=0.15, color=color)

                ax.set_title(model, fontsize=11, fontweight="bold")
                ax.set_xlabel("Turn", fontsize=10)
                ax.set_ylim(0, 10.5)
                ax.legend(fontsize=9)
                ax.grid(True, alpha=0.3)

            axes[0].set_ylabel("Overall Coherence", fontsize=11)
            fig.suptitle(f"Coherence Decay by Model: {scenario}\nJudge: {judge}",
                         fontsize=13, fontweight="bold")
            plt.tight_layout()
            fname = f"decay_by_model_{scenario}_{judge.replace('/', '_')}.png"
            fig.savefig(os.path.join(outdir, fname), dpi=DPI, bbox_inches="tight")
            plt.close()
            print(f"  Saved: {fname}")


def plot_subdimension_evolution(df, outdir):
    """How each sub-dimension evolves over turns for scheming vs honest."""
    dims = ["internal_consistency", "reasoning_quality", "specificity", "defensiveness"]
    dim_labels = ["Internal\nConsistency", "Reasoning\nQuality", "Specificity", "Defensiveness"]

    for scenario in df["scenario"].unique():
        sdf = df[df["scenario"] == scenario]

        for judge in sdf["judge_model"].unique():
            jdf = sdf[sdf["judge_model"] == judge]

            fig, axes = plt.subplots(1, 4, figsize=(20, 5), sharey=True)

            for ax, dim, label in zip(axes, dims, dim_labels):
                for condition, color, style in [("honest", "#2ca02c", "-"), ("scheming", "#d62728", "--")]:
                    cdf = jdf[jdf["condition"] == condition]
                    if cdf.empty:
                        continue
                    grouped = cdf.groupby("turn")[dim].agg(["mean", "std"]).reset_index()
                    ax.plot(grouped["turn"], grouped["mean"], f'{style}o', color=color,
                            linewidth=2, markersize=4, label=condition)
                    ax.fill_between(grouped["turn"],
                                    grouped["mean"] - grouped["std"],
                                    grouped["mean"] + grouped["std"],
                                    alpha=0.15, color=color)

                ax.set_title(label, fontsize=11, fontweight="bold")
                ax.set_xlabel("Turn", fontsize=10)
                ax.set_ylim(0, 10.5)
                ax.grid(True, alpha=0.3)
                ax.legend(fontsize=8)

            axes[0].set_ylabel("Score", fontsize=11)
            fig.suptitle(f"Sub-dimension Evolution: {scenario}\nJudge: {judge}",
                         fontsize=13, fontweight="bold")
            plt.tight_layout()
            fname = f"subdims_{scenario}_{judge.replace('/', '_')}.png"
            fig.savefig(os.path.join(outdir, fname), dpi=DPI, bbox_inches="tight")
            plt.close()
            print(f"  Saved: {fname}")


def plot_scenario_comparison(df, outdir):
    """Compare average coherence gap (honest - scheming) across scenarios."""
    for judge in df["judge_model"].unique():
        jdf = df[df["judge_model"] == judge]

        gaps = []
        for scenario in jdf["scenario"].unique():
            sdf = jdf[jdf["scenario"] == scenario]
            honest_mean = sdf[sdf["condition"] == "honest"]["coherence_overall"].mean()
            scheming_mean = sdf[sdf["condition"] == "scheming"]["coherence_overall"].mean()
            gaps.append({
                "scenario": scenario,
                "honest_mean": honest_mean,
                "scheming_mean": scheming_mean,
                "gap": honest_mean - scheming_mean,
            })

        gap_df = pd.DataFrame(gaps).sort_values("gap", ascending=True)

        fig, ax = plt.subplots(figsize=(10, 5))
        x = range(len(gap_df))
        ax.barh(list(x), gap_df["gap"], color=["#d62728" if g < 0 else "#2ca02c" for g in gap_df["gap"]])
        ax.set_yticks(list(x))
        ax.set_yticklabels(gap_df["scenario"])
        ax.set_xlabel("Coherence Gap (honest − scheming)", fontsize=12)
        ax.set_title(f"Coherence Gap by Scenario\nJudge: {judge}\n(positive = honest more coherent)",
                     fontsize=13, fontweight="bold")
        ax.axvline(x=0, color="black", linewidth=0.5)

        for i, (_, row) in enumerate(gap_df.iterrows()):
            ax.text(row["gap"] + 0.05, i, f"{row['gap']:.2f}", va="center", fontsize=10)

        plt.tight_layout()
        fname = f"scenario_gap_{judge.replace('/', '_')}.png"
        fig.savefig(os.path.join(outdir, fname), dpi=DPI, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {fname}")


def print_summary(df, outdir):
    """Print summary statistics."""
    lines = []
    lines.append("=" * 70)
    lines.append("COHERENCE UNDER PRESSURE — SUMMARY")
    lines.append("=" * 70)

    for judge in df["judge_model"].unique():
        jdf = df[df["judge_model"] == judge]
        lines.append(f"\nJudge: {judge}")
        lines.append("-" * 50)

        for scenario in sorted(jdf["scenario"].unique()):
            sdf = jdf[jdf["scenario"] == scenario]
            lines.append(f"\n  Scenario: {scenario}")

            for condition in ["honest", "scheming"]:
                cdf = sdf[sdf["condition"] == condition]
                if cdf.empty:
                    continue
                lines.append(f"    {condition.upper()}:")
                lines.append(f"      Coherence:    {cdf['coherence_overall'].mean():.2f} ± {cdf['coherence_overall'].std():.2f}")
                lines.append(f"      Consistency:  {cdf['internal_consistency'].mean():.2f} ± {cdf['internal_consistency'].std():.2f}")
                lines.append(f"      Reasoning:    {cdf['reasoning_quality'].mean():.2f} ± {cdf['reasoning_quality'].std():.2f}")
                lines.append(f"      Specificity:  {cdf['specificity'].mean():.2f} ± {cdf['specificity'].std():.2f}")
                lines.append(f"      Defensiveness:{cdf['defensiveness'].mean():.2f} ± {cdf['defensiveness'].std():.2f}")

                early = cdf[cdf["turn"] <= 3]["coherence_overall"].mean()
                late = cdf[cdf["turn"] >= 8]["coherence_overall"].mean()
                if not np.isnan(early) and not np.isnan(late):
                    lines.append(f"      Early (t1-3): {early:.2f}  Late (t8-10): {late:.2f}  Δ: {late - early:+.2f}")

    output = "\n".join(lines)
    print(output)

    with open(os.path.join(outdir, "cup_summary.txt"), "w") as f:
        f.write(output)
    print(f"\n  Saved: cup_summary.txt")


def run_plots(df: pd.DataFrame, figures_dir: str = FIGURES_DIR) -> None:
    """Generate all plots from a scored DataFrame."""
    if df.empty:
        print("No data to plot.")
        return

    print(f"Loaded {len(df)} score rows")
    print(f"Judges: {df['judge_model'].unique()}")
    print(f"Scenarios: {df['scenario'].unique()}")
    print(f"Models: {df['subject_model'].unique()}")

    os.makedirs(figures_dir, exist_ok=True)

    print("\n--- Coherence Decay Curves ---")
    plot_coherence_decay(df, figures_dir)

    print("\n--- Coherence Decay by Model ---")
    plot_coherence_decay_by_model(df, figures_dir)

    print("\n--- Sub-dimension Evolution ---")
    plot_subdimension_evolution(df, figures_dir)

    print("\n--- Scenario Comparison ---")
    plot_scenario_comparison(df, figures_dir)

    print("\n--- Summary Statistics ---")
    print_summary(df, figures_dir)

    print(f"\nAll plots saved to: {figures_dir}/")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze CuP-LLM experiments")
    parser.add_argument("--plot", action="store_true", help="Generate plots")
    parser.add_argument("--logs-dir", type=str, default=INSPECT_LOGS_DIR,
                        help=f"Directory containing Inspect .eval log files (default: {INSPECT_LOGS_DIR})")
    parser.add_argument("--csv", type=str, default=None,
                        help="Load scores from a legacy CSV file instead of Inspect logs")
    parser.add_argument("--figures-dir", type=str, default=FIGURES_DIR,
                        help=f"Output directory for plots (default: {FIGURES_DIR})")
    args = parser.parse_args()

    if not args.plot:
        print("Specify --plot to generate plots. Example:")
        print("  python analyze.py --plot --logs-dir ./logs")
        exit(1)

    if args.csv:
        df = pd.read_csv(args.csv)
        print(f"Loaded {len(df)} rows from {args.csv}")
    else:
        df = load_from_inspect_logs(args.logs_dir)

    run_plots(df, figures_dir=args.figures_dir)
