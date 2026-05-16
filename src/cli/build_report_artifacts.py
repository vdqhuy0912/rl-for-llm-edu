#!/usr/bin/env python3
"""Build visualization and report artifacts for the SFT/KTO experiment."""

import argparse
import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.cli.build_judged_kto_data import extract_json_object, triggered_or_violated_labels
from src.utils.eval_utils import load_records
from src.utils.model_utils import ensure_output_dir, resolve_project_path, setup_logging


METRIC_LABELS = {
    "token_f1": "Token F1",
    "rouge_l": "ROUGE-L",
    "bleu": "BLEU",
    "phobert_semantic_similarity": "PhoBERT Similarity",
}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def flatten_autometrics(model_dir: Path, model_label: str) -> dict[str, Any] | None:
    summary_path = model_dir / "autometrics" / "autometrics_summary.json"
    if not summary_path.exists():
        return None
    summary = read_json(summary_path)
    row = {"model": model_label, "num_samples": summary.get("num_samples", 0)}
    for metric, stats in summary.get("metrics", {}).items():
        row[f"{metric}_mean"] = stats.get("mean")
        row[f"{metric}_median"] = stats.get("median")
        row[f"{metric}_std"] = stats.get("std")
    return row


def collect_suite_models(suite_dir: Path) -> list[tuple[str, Path]]:
    models = []
    for child in sorted(suite_dir.iterdir() if suite_dir.exists() else []):
        if child.is_dir() and ((child / "inference").exists() or (child / "judge").exists()):
            models.append((child.name, child))
    return models


def collect_judge_rows(model_label: str, model_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    judge_path = model_dir / "judge" / "evaluation_results.json"
    if not judge_path.exists():
        return [], []

    records = load_records(judge_path)
    classification_rows = []
    behavior_rows = []
    class_counts = Counter(str(record.get("judge_classification_label") or "UNKNOWN") for record in records)
    for label, count in sorted(class_counts.items()):
        classification_rows.append({"model": model_label, "judge_classification_label": label, "count": count})

    behavior_counts = Counter()
    for record in records:
        payload = extract_json_object(record.get("judge_evaluation"))
        if not payload:
            behavior_counts["unparsed_or_classification_only"] += 1
            continue
        labels = triggered_or_violated_labels(payload)
        if not labels:
            behavior_counts["no_triggered_label"] += 1
        for label in labels:
            behavior_counts[label] += 1

    for label, count in behavior_counts.most_common():
        behavior_rows.append({"model": model_label, "behavior_label": label, "count": count})
    return classification_rows, behavior_rows


def collect_loss_rows(train_dirs: list[str]) -> list[dict[str, Any]]:
    rows = []
    for spec in train_dirs:
        if "=" not in spec:
            raise ValueError(f"Training dir spec must be label=path, got: {spec}")
        label, raw_path = spec.split("=", maxsplit=1)
        train_dir = resolve_project_path(raw_path)
        state_path = train_dir / "trainer_state.json"
        if not state_path.exists():
            continue
        state = read_json(state_path)
        for item in state.get("log_history", []):
            step = item.get("step")
            if step is None:
                continue
            if "loss" in item:
                rows.append({"run": label, "step": step, "metric": "train_loss", "value": item["loss"]})
            if "eval_loss" in item:
                rows.append({"run": label, "step": step, "metric": "eval_loss", "value": item["eval_loss"]})
    return rows


def collect_kto_signal_rows(kto_dirs: list[str]) -> list[dict[str, Any]]:
    rows = []
    for spec in kto_dirs:
        if "=" not in spec:
            raise ValueError(f"KTO data spec must be label=path, got: {spec}")
        label, raw_path = spec.split("=", maxsplit=1)
        stats_path = resolve_project_path(raw_path) / "build_stats.json"
        if not stats_path.exists():
            continue
        stats = read_json(stats_path)
        for key, value in sorted(stats.items()):
            rows.append({"dataset": label, "stat": key, "count": value})
    return rows


def save_table(df: pd.DataFrame, output_dir: Path, stem: str) -> Path:
    csv_path = output_dir / f"{stem}.csv"
    md_path = output_dir / f"{stem}.md"
    df.to_csv(csv_path, index=False, encoding="utf-8")
    md_path.write_text(df.to_markdown(index=False), encoding="utf-8")
    return csv_path


def plot_loss(loss_df: pd.DataFrame, output_dir: Path) -> None:
    if loss_df.empty:
        return
    plt.figure(figsize=(10, 5))
    for (run, metric), group in loss_df.groupby(["run", "metric"]):
        group = group.sort_values("step")
        plt.plot(group["step"], group["value"], marker="o", linewidth=1.6, label=f"{run} {metric}")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title("Training and Evaluation Loss")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "train_loss.png", dpi=180)
    plt.close()


def plot_autometrics(metrics_df: pd.DataFrame, output_dir: Path) -> None:
    mean_columns = [column for column in metrics_df.columns if column.endswith("_mean")]
    if metrics_df.empty or not mean_columns:
        return
    plot_rows = []
    for _, row in metrics_df.iterrows():
        for column in mean_columns:
            metric = column.removesuffix("_mean")
            plot_rows.append(
                {
                    "model": row["model"],
                    "metric": METRIC_LABELS.get(metric, metric),
                    "value": row[column],
                }
            )
    plot_df = pd.DataFrame(plot_rows).dropna()
    if plot_df.empty:
        return
    pivot = plot_df.pivot(index="metric", columns="model", values="value")
    ax = pivot.plot(kind="bar", figsize=(11, 5), rot=0)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Mean Score")
    ax.set_title("Automatic Metrics by Model")
    ax.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_dir / "autometrics_by_model.png", dpi=180)
    plt.close()


def plot_classifications(class_df: pd.DataFrame, output_dir: Path) -> None:
    if class_df.empty:
        return
    pivot = class_df.pivot_table(
        index="model",
        columns="judge_classification_label",
        values="count",
        aggfunc="sum",
        fill_value=0,
    )
    ax = pivot.plot(kind="bar", stacked=True, figsize=(10, 5), rot=0)
    ax.set_ylabel("Samples")
    ax.set_title("Judge Classification Distribution")
    ax.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_dir / "judge_classification_distribution.png", dpi=180)
    plt.close()


def plot_top_behaviors(behavior_df: pd.DataFrame, output_dir: Path, top_n: int) -> None:
    if behavior_df.empty:
        return
    top_labels = (
        behavior_df.groupby("behavior_label")["count"]
        .sum()
        .sort_values(ascending=False)
        .head(top_n)
        .index
    )
    plot_df = behavior_df[behavior_df["behavior_label"].isin(top_labels)]
    pivot = plot_df.pivot_table(index="behavior_label", columns="model", values="count", aggfunc="sum", fill_value=0)
    ax = pivot.sort_index().plot(kind="barh", figsize=(11, max(5, 0.35 * len(pivot))), stacked=False)
    ax.set_xlabel("Count")
    ax.set_ylabel("Behavior Label")
    ax.set_title(f"Top {top_n} Judge Behavior Labels")
    ax.grid(axis="x", alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_dir / "top_behavior_labels.png", dpi=180)
    plt.close()


def plot_kto_signals(kto_df: pd.DataFrame, output_dir: Path) -> None:
    if kto_df.empty:
        return
    signal_df = kto_df[kto_df["stat"].str.startswith("label.") | kto_df["stat"].str.startswith("decision.")]
    if signal_df.empty:
        return
    pivot = signal_df.pivot_table(index="stat", columns="dataset", values="count", aggfunc="sum", fill_value=0)
    ax = pivot.plot(kind="barh", figsize=(10, max(5, 0.35 * len(pivot))))
    ax.set_xlabel("Count")
    ax.set_title("KTO Signal Distribution")
    ax.grid(axis="x", alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_dir / "kto_signal_distribution.png", dpi=180)
    plt.close()


def write_report(output_dir: Path, generated_files: list[str]) -> None:
    lines = [
        "# SFT + Judged KTO Experiment Report Draft",
        "",
        f"Generated at: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "## Artifacts",
        "",
    ]
    for filename in sorted(generated_files):
        lines.append(f"- `{filename}`")
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- Fill in narrative conclusions after the full pipeline completes.",
            "- Use `autometrics_summary.md`, `judge_classification_counts.md`, and `behavior_label_counts.md` for model comparison tables.",
            "- Use `train_loss.png`, `autometrics_by_model.png`, `judge_classification_distribution.png`, `top_behavior_labels.png`, and `kto_signal_distribution.png` in the final report.",
        ]
    )
    (output_dir / "report_draft.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite-dir", default="./results/model_suite", help="Directory containing per-model eval outputs.")
    parser.add_argument("--output-dir", default="./results/report_artifacts")
    parser.add_argument("--train-dir", action="append", default=[], help="Training loss source as label=path.")
    parser.add_argument("--kto-data-dir", action="append", default=[], help="KTO signal source as label=path.")
    parser.add_argument("--top-behaviors", type=int, default=20)
    return parser.parse_args()


def main():
    args = parse_args()
    logger = setup_logging(logger_name="report.artifacts")
    output_dir = ensure_output_dir(args.output_dir)
    suite_dir = resolve_project_path(args.suite_dir)

    metric_rows = []
    classification_rows = []
    behavior_rows = []
    for model_label, model_dir in collect_suite_models(suite_dir):
        metric_row = flatten_autometrics(model_dir, model_label)
        if metric_row:
            metric_rows.append(metric_row)
        model_class_rows, model_behavior_rows = collect_judge_rows(model_label, model_dir)
        classification_rows.extend(model_class_rows)
        behavior_rows.extend(model_behavior_rows)

    generated = []
    metrics_df = pd.DataFrame(metric_rows)
    class_df = pd.DataFrame(classification_rows)
    behavior_df = pd.DataFrame(behavior_rows)
    loss_df = pd.DataFrame(collect_loss_rows(args.train_dir))
    kto_df = pd.DataFrame(collect_kto_signal_rows(args.kto_data_dir))

    for stem, df in [
        ("autometrics_summary", metrics_df),
        ("judge_classification_counts", class_df),
        ("behavior_label_counts", behavior_df),
        ("training_loss", loss_df),
        ("kto_signal_stats", kto_df),
    ]:
        if not df.empty:
            save_table(df, output_dir, stem)
            generated.extend([f"{stem}.csv", f"{stem}.md"])

    plot_loss(loss_df, output_dir)
    plot_autometrics(metrics_df, output_dir)
    plot_classifications(class_df, output_dir)
    plot_top_behaviors(behavior_df, output_dir, args.top_behaviors)
    plot_kto_signals(kto_df, output_dir)
    generated.extend(path.name for path in output_dir.glob("*.png"))
    write_report(output_dir, generated)

    logger.info("Report artifacts saved to %s", output_dir)
    print(json.dumps({"output_dir": str(output_dir), "files": sorted(path.name for path in output_dir.iterdir())}, indent=2))


if __name__ == "__main__":
    main()
