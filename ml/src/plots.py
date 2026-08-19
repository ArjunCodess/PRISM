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


def _save(fig: plt.Figure, path: Path) -> Path:
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


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
    written.append(_save(fig, output_dir / "model-comparison.png"))

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
        xlabel="Predicted high-risk-event probability",
        ylabel="Observed frequency",
    )
    ax.set_title(
        "High-risk-event probability reliability (small positive class)",
        loc="left",
        pad=16,
    )
    ax.legend(frameon=False, labelcolor=TEXT)
    ax.spines[["top", "right"]].set_visible(False)
    written.append(_save(fig, output_dir / "calibration-reliability.png"))

    groups = metrics.get("featureGroups", [])[:9][::-1]
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh([item["group"] for item in groups], [item["gain"] for item in groups], color=CYAN)
    ax.set_xlabel("XGBoost gain")
    ax.set_title("Grouped feature importance", loc="left", pad=16)
    ax.spines[["top", "right"]].set_visible(False)
    written.append(_save(fig, output_dir / "feature-importance.png"))

    families = metrics.get("ablation", {}).get("families", {})
    family_order = [
        ("snapshot", "Snapshot"),
        ("snapshot_history", "Snapshot + history"),
        ("snapshot_history_covariance", "History + covariance"),
        ("full", "Full"),
    ]
    if families:
        labels = [label for key, label in family_order if key in families]
        values = [families[key]["mae"] for key, _label in family_order if key in families]
        fig, ax = plt.subplots(figsize=(9, 5))
        bars = ax.bar(
            labels,
            values,
            color=[CYAN if label == "Full" else "#476171" for label in labels],
        )
        ax.bar_label(bars, fmt="%.3f", padding=4, color=TEXT)
        ax.set_ylabel("Mean absolute error")
        ax.set_title("Does history add signal beyond the latest snapshot?", loc="left", pad=16)
        ax.spines[["top", "right"]].set_visible(False)
        written.append(_save(fig, output_dir / "feature-ablation.png"))

    horizons = [row for row in metrics.get("horizons", []) if "model" in row]
    if horizons:
        hours = [row["cutoffHours"] for row in horizons]
        model_mae = [row["model"]["mae"] for row in horizons]
        persist_mae = [row["persistence"]["mae"] for row in horizons]
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(hours, persist_mae, marker="o", color="#476171", label="Persistence")
        ax.plot(hours, model_mae, marker="o", color=CYAN, label="XGBoost")
        ax.invert_xaxis()
        ax.set_xlabel("Forecast horizon (hours before closest approach)")
        ax.set_ylabel("MAE (log10 Pc)")
        ax.set_title(
            "The value of learned forecasting is highest when information is sparse",
            loc="left",
            pad=16,
        )
        ax.axvline(48, color="#526574", linestyle="--", linewidth=1)
        ax.text(48, max(persist_mae) * 0.92, "T−48 exhibit cutoff", color=MUTED, fontsize=8)
        ax.legend(frameon=False, labelcolor=TEXT)
        ax.spines[["top", "right"]].set_visible(False)
        written.append(_save(fig, output_dir / "forecast-horizon.png"))

    curve = metrics.get("abstention", {}).get("coverageCurve", [])
    if curve:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(
            [item["coverage"] for item in curve],
            [item["maeAccepted"] for item in curve],
            marker="o",
            color=CYAN,
        )
        operating = next((item for item in curve if item.get("operatingPoint")), None)
        if operating:
            ax.scatter(
                [operating["coverage"]],
                [operating["maeAccepted"]],
                color=AMBER,
                s=90,
                zorder=3,
                label="operating point",
            )
            ax.annotate(
                f"{operating['coverage'] * 100:.1f}% coverage\n"
                f"accepted MAE {operating['maeAccepted']:.3f}",
                xy=(operating["coverage"], operating["maeAccepted"]),
                xytext=(operating["coverage"] - 0.22, operating["maeAccepted"] + 0.45),
                color=TEXT,
                fontsize=8,
                arrowprops={"arrowstyle": "->", "color": AMBER},
            )
            ax.legend(frameon=False, labelcolor=TEXT)
        ax.set_xlabel("Coverage (fraction that receive a firm forecast)")
        ax.set_ylabel("MAE among accepted events (log10 Pc)")
        ax.set_title("Selective prediction: coverage versus accepted error", loc="left", pad=16)
        ax.spines[["top", "right"]].set_visible(False)
        written.append(_save(fig, output_dir / "abstention-coverage.png"))

    modes = metrics.get("failureClusters", {}).get("modes", {})
    failure_modes = {
        name: payload for name, payload in modes.items() if name != "accurate" and payload.get("n")
    }
    if failure_modes:
        ranked = sorted(failure_modes.items(), key=lambda item: item[1]["n"], reverse=True)[:8]
        fig, ax = plt.subplots(figsize=(9, 5))
        labels = [name.replace("_", " ") for name, _payload in ranked]
        counts = [payload["n"] for _name, payload in ranked]
        bars = ax.barh(labels[::-1], counts[::-1], color=AMBER)
        ax.bar_label(bars, padding=4, color=TEXT)
        ax.set_xlabel("Test events")
        ax.set_title("How inaccurate forecasts cluster", loc="left", pad=16)
        ax.spines[["top", "right"]].set_visible(False)
        written.append(_save(fig, output_dir / "failure-clusters.png"))

    contrast = metrics.get("shapContrast", {})
    correct_groups = {
        item["group"]: item["meanAbsShap"]
        for item in contrast.get("correct", {}).get("groups", [])
    }
    incorrect_groups = {
        item["group"]: item["meanAbsShap"]
        for item in contrast.get("incorrect", {}).get("groups", [])
    }
    ranked_incorrect = sorted(incorrect_groups.items(), key=lambda item: item[1], reverse=True)
    names = [name for name, _value in ranked_incorrect[:6]]
    if names:
        fig, ax = plt.subplots(figsize=(9, 5))
        y_pos = range(len(names))
        ax.barh(
            [y - 0.18 for y in y_pos],
            [correct_groups.get(name, 0.0) for name in names],
            height=0.35,
            color="#476171",
            label="|error| ≤ 0.5",
        )
        incorrect_colors = [
            CYAN if "complete the tracking" in name else AMBER for name in names
        ]
        ax.barh(
            [y + 0.18 for y in y_pos],
            [incorrect_groups.get(name, 0.0) for name in names],
            height=0.35,
            color=incorrect_colors,
            label="|error| ≥ 2.0",
        )
        ax.set_yticks(list(y_pos))
        ax.set_yticklabels(names)
        ax.invert_yaxis()
        ax.set_xlabel("Mean |SHAP| (association, not physical cause)")
        ax.set_title(
            "Tracking-completeness |SHAP| is higher among large errors",
            loc="left",
            pad=16,
        )
        ax.legend(frameon=False, labelcolor=TEXT)
        ax.spines[["top", "right"]].set_visible(False)
        written.append(_save(fig, output_dir / "shap-contrast.png"))

    conformal = metrics.get("conformal", {}).get("test")
    if conformal:
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.plot([0, 1], [0, 1], linestyle="--", color="#526574", label="nominal coverage")
        boot = conformal.get("bootstrap", {})
        conf = conformal.get("conformal", {})
        if "50" in boot and "90" in boot:
            ax.scatter(
                [0.5, 0.9],
                [boot["50"]["coverage"], boot["90"]["coverage"]],
                color="#476171",
                s=70,
                zorder=3,
                label="bootstrap spread",
            )
        if "50" in conf and "90" in conf:
            ax.scatter(
                [0.5, 0.9],
                [conf["50"]["coverage"], conf["90"]["coverage"]],
                color=CYAN,
                s=70,
                zorder=3,
                label="split conformal",
            )
        ax.set(
            xlim=(0, 1),
            ylim=(0, 1),
            xlabel="Nominal coverage",
            ylabel="Empirical coverage on frozen test",
        )
        ax.set_title("Bootstrap spread vs split-conformal coverage", loc="left", pad=16)
        ax.legend(frameon=False, labelcolor=TEXT)
        ax.spines[["top", "right"]].set_visible(False)
        written.append(_save(fig, output_dir / "coverage-calibration.png"))

    official = metrics.get("officialTest") or {}
    esa = official.get("esa") or {}
    if esa:
        fig, ax = plt.subplots(figsize=(8, 5))
        names = ["persistence", "xgboost", "residual", "floorHurdle", "ensemble"]
        labels = {
            "persistence": "Persistence (LRP)",
            "xgboost": "Unguarded XGB",
            "residual": "Residual XGB",
            "floorHurdle": "Floor hurdle",
            "ensemble": "Selected ens.",
        }
        present = [name for name in names if name in esa]
        values = [float(esa[name]["esaLoss"]) for name in present]
        ax.bar([labels[name] for name in present], values, color=CYAN)
        ax.axhline(0.694, color=AMBER, linestyle="--", label="Uriot LRP 0.694")
        ax.axhline(0.556, color=MUTED, linestyle=":", label="Uriot sesc 0.556")
        ax.set_yscale("log")
        ax.set_ylabel("ESA-style loss (log)")
        ax.set_title("Official-test ESA-style loss (frozen models)", loc="left", pad=16)
        ax.legend(frameon=False, labelcolor=TEXT)
        ax.spines[["top", "right"]].set_visible(False)
        written.append(_save(fig, output_dir / "official-test-esa.png"))

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
