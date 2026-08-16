from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

INK = "#070b12"
PANEL = "#101924"
TEXT = "#eef6f8"
MUTED = "#91a3af"
CYAN = "#73d9e6"
AMBER = "#e6b86a"


def _setup() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": INK,
            "axes.facecolor": PANEL,
            "axes.edgecolor": "#30404c",
            "axes.labelcolor": MUTED,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "text.color": TEXT,
            "font.family": "DejaVu Sans",
            "axes.titleweight": "bold",
        }
    )


def generate_plots(metrics_path: Path, output_dir: Path) -> list[Path]:
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    output_dir.mkdir(parents=True, exist_ok=True)
    _setup()
    written: list[Path] = []

    systems = [
        name
        for name in ("persistence", "median", "ridge", "xgboost", "ensemble")
        if name in metrics
    ]
    values = [metrics[name]["mae"] for name in systems]
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = [CYAN if name == "ensemble" else "#476171" for name in systems]
    bars = ax.bar([name.title() for name in systems], values, color=colors)
    ax.bar_label(bars, fmt="%.3f", padding=4, color=TEXT)
    ax.set_ylabel("Mean absolute error (log-risk units)")
    ax.set_title("Held-out model comparison", loc="left", pad=16)
    ax.spines[["top", "right"]].set_visible(False)
    path = output_dir / "model-comparison.png"
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    written.append(path)

    bins = metrics.get("calibration", [])
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], linestyle="--", color="#526574", label="perfect calibration")
    ax.plot(
        [item["predicted"] for item in bins],
        [item["observed"] for item in bins],
        marker="o",
        color=CYAN,
        linewidth=2,
        label="PRISM",
    )
    ax.set(
        xlim=(0, 1),
        ylim=(0, 1),
        xlabel="Predicted warning probability",
        ylabel="Observed frequency",
    )
    ax.set_title("Warning reliability", loc="left", pad=16)
    ax.legend(frameon=False, labelcolor=TEXT)
    ax.spines[["top", "right"]].set_visible(False)
    path = output_dir / "calibration-reliability.png"
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    written.append(path)

    groups = metrics.get("featureGroups", [])[:9][::-1]
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh([item["group"] for item in groups], [item["gain"] for item in groups], color=CYAN)
    ax.set_xlabel("XGBoost gain")
    ax.set_title("Grouped feature importance", loc="left", pad=16)
    ax.spines[["top", "right"]].set_visible(False)
    path = output_dir / "feature-importance.png"
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    written.append(path)

    ablation = metrics.get("ablation", {})
    labels = ["Snapshot only", "Snapshot + trends", "Ensemble"]
    values = [ablation.get("snapshot_mae"), ablation.get("full_mae"), ablation.get("ensemble_mae")]
    if all(value is not None for value in values):
        fig, ax = plt.subplots(figsize=(8, 5))
        bars = ax.bar(labels, values, color=["#476171", CYAN, AMBER])
        ax.bar_label(bars, fmt="%.3f", padding=4, color=TEXT)
        ax.set_ylabel("Mean absolute error")
        ax.set_title("Feature ablation", loc="left", pad=16)
        ax.spines[["top", "right"]].set_visible(False)
        path = output_dir / "feature-ablation.png"
        fig.tight_layout()
        fig.savefig(path, dpi=180)
        plt.close(fig)
        written.append(path)

    return written


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate PRISM evaluation figures")
    parser.add_argument("--metrics", type=Path, default=Path("ml/artifacts/metrics.json"))
    parser.add_argument("--output", type=Path, default=Path("docs/figures"))
    args = parser.parse_args()
    for path in generate_plots(args.metrics, args.output):
        print(path)


if __name__ == "__main__":
    main()
