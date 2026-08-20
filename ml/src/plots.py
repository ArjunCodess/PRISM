from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
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

    probe = metrics.get("dilutionProbe") or {}
    quartiles = probe.get("quartiles") or []
    if quartiles:
        fig, ax = plt.subplots(figsize=(7.2, 4.8))
        xs = [f"Q{row['quartile']}" for row in quartiles]
        positions = list(range(len(xs)))
        ax.bar(
            positions,
            [float(row["floorRate"]) for row in quartiles],
            color=CYAN,
            label="floor rate",
        )
        ax.set_xticks(positions, xs)
        ax.set_ylabel("Floor rate")
        ax.set_xlabel("dilution_gap quartile (train edges)")
        twin = ax.twinx()
        twin.plot(
            positions,
            [float(row["meanAbsMove"]) for row in quartiles],
            color=AMBER,
            marker="o",
            label="mean |y − risk|",
        )
        twin.set_ylabel("Mean |Δrisk|")
        ax.set_title("Floor collapse and report movement by max-risk gap", loc="left", pad=16)
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = twin.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, frameon=False, labelcolor=TEXT)
        ax.spines[["top"]].set_visible(False)
        twin.spines[["top"]].set_visible(False)
        written.append(_save(fig, output_dir / "dilution-probe.png"))

    written.extend(_paper_figures(metrics, output_dir))
    return written


PAPER_BLUE = "#0072B2"
PAPER_ORANGE = "#D55E00"
PAPER_GREY = "#222222"


def _setup_paper() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": PAPER_GREY,
            "axes.labelcolor": PAPER_GREY,
            "xtick.color": PAPER_GREY,
            "ytick.color": PAPER_GREY,
            "text.color": PAPER_GREY,
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "axes.titleweight": "medium",
            "axes.grid": False,
            "legend.frameon": False,
        }
    )


def _save_paper(fig: plt.Figure, path: Path) -> Path:
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def _centers(edges: list[float]) -> list[float]:
    return [(edges[i] + edges[i + 1]) / 2.0 for i in range(len(edges) - 1)]


def _paper_figures(metrics: dict, output_dir: Path) -> list[Path]:
    _setup_paper()
    written: list[Path] = []
    armor = metrics.get("reviewArmor") or {}

    matched = (armor.get("matchedCohortHorizons") or {}).get("rows") or []
    if matched:
        hours = [int(row["cutoffHours"]) for row in matched]
        model_mae = [float(row["xgboostMae"]) for row in matched]
        persist_mae = [float(row["persistenceMae"]) for row in matched]
        floor_rate = [float(row["floorRate"]) for row in matched]
        fig, (ax, ax2) = plt.subplots(
            2, 1, figsize=(3.5, 3.4), sharex=True, gridspec_kw={"height_ratios": [2.2, 1.0]}
        )
        ax.plot(hours, persist_mae, marker="o", color=PAPER_GREY, label="Persistence")
        ax.plot(hours, model_mae, marker="o", color=PAPER_BLUE, label="XGBoost")
        lows = []
        highs = []
        for row, persist, model in zip(matched, persist_mae, model_mae):
            if row.get("deltaMaeCi95Low") is None:
                lows.append(0.0)
                highs.append(0.0)
                continue
            low_model = persist - float(row["deltaMaeCi95High"])
            high_model = persist - float(row["deltaMaeCi95Low"])
            lows.append(max(model - low_model, 0.0))
            highs.append(max(high_model - model, 0.0))
        ax.errorbar(
            hours,
            model_mae,
            yerr=np.array([lows, highs]),
            fmt="none",
            ecolor=PAPER_BLUE,
            capsize=3,
            elinewidth=1.0,
            zorder=4,
        )
        ax.set_ylabel("MAE of log10(Pc)")
        ax.legend(loc="upper right")
        ax.spines[["top", "right"]].set_visible(False)
        ax2.plot(hours, floor_rate, marker="s", color=PAPER_ORANGE)
        ax2.set_xticks(hours)
        ax2.set_xlim(78, 6)
        ax2.set_ylim(0, 1)
        ax2.set_xlabel("Time to closest approach / h")
        ax2.set_ylabel("Floor rate")
        ax2.spines[["top", "right"]].set_visible(False)
        written.append(_save_paper(fig, output_dir / "horizon-decay.png"))
    else:
        horizons = [row for row in metrics.get("horizons", []) if "model" in row]
        if horizons:
            hours = [int(row["cutoffHours"]) for row in horizons]
            model_mae = [float(row["model"]["mae"]) for row in horizons]
            persist_mae = [float(row["persistence"]["mae"]) for row in horizons]
            fig, ax = plt.subplots(figsize=(3.5, 2.6))
            ax.plot(hours, persist_mae, marker="o", color=PAPER_GREY, label="Persistence")
            ax.plot(hours, model_mae, marker="o", color=PAPER_BLUE, label="XGBoost")
            ax.set_xticks(hours)
            ax.set_xlim(78, 6)
            ax.set_xlabel("Time to closest approach / h")
            ax.set_ylabel("MAE of log10(Pc)")
            ax.legend(loc="upper right")
            ax.spines[["top", "right"]].set_visible(False)
            written.append(_save_paper(fig, output_dir / "horizon-decay.png"))

    anatomy = metrics.get("errorAnatomy") or {}
    edges = anatomy.get("binEdges") or []
    if len(edges) >= 2:
        centers = _centers([float(value) for value in edges])
        width = float(edges[1]) - float(edges[0])
        fig, ax = plt.subplots(figsize=(3.5, 2.8))
        ax.bar(
            centers,
            anatomy.get("actualMoveCounts") or [],
            width=width,
            color=PAPER_GREY,
            alpha=0.55,
            label="y − risk",
        )
        ax.bar(
            centers,
            anatomy.get("residualErrorCounts") or [],
            width=width,
            color=PAPER_BLUE,
            alpha=0.45,
            label="y − pred",
        )
        ax.axvline(0.0, color=PAPER_ORANGE, linestyle="--", linewidth=1.0)
        peak = max(list(anatomy.get("actualMoveCounts") or [1]) + [1])
        ax.annotate("exact persistence", xy=(0.4, peak * 0.85), fontsize=7, color=PAPER_GREY)
        ax.annotate(
            "−30 floor collapses",
            xy=(-20.0, peak * 0.18),
            xytext=(-31.0, peak * 0.62),
            fontsize=8,
            color=PAPER_ORANGE,
            arrowprops={"arrowstyle": "->", "color": PAPER_ORANGE, "lw": 0.8},
        )
        ax.annotate("non-floor", xy=(2.0, peak * 0.22), fontsize=7, color=PAPER_BLUE)
        ax.set_xlabel("Change in log10(Pc)")
        ax.set_ylabel("Events")
        ax.legend(loc="upper right")
        ax.spines[["top", "right"]].set_visible(False)
        written.append(_save_paper(fig, output_dir / "error-anatomy.png"))

    conformal = (metrics.get("conformal") or {}).get("test") or {}
    boot = conformal.get("bootstrap") or {}
    conf = conformal.get("conformal") or {}
    live90 = ((metrics.get("selectedPolicy") or {}).get("exhibitConformal") or {}).get("90") or {}
    depth = (armor.get("conformalDepth") or {}).get("overall") or {}
    if "50" in boot and "90" in boot and "50" in conf and "90" in conf:
        fig, ax = plt.subplots(figsize=(3.2, 3.2))
        ax.plot([0, 1], [0, 1], linestyle="--", color="#888888", label="y = x")
        ax.scatter(
            [0.5, 0.9],
            [float(boot["50"]["coverage"]), float(boot["90"]["coverage"])],
            color=PAPER_GREY,
            s=42,
            zorder=3,
            label="Bootstrap spread",
        )
        ax.scatter(
            [0.5, 0.9],
            [float(conf["50"]["coverage"]), float(conf["90"]["coverage"])],
            color=PAPER_BLUE,
            s=42,
            zorder=3,
            label="Split conformal",
        )
        ci = depth.get("coverageCi95") or {}
        if live90.get("coverage") is not None and ci.get("low") is not None:
            live = float(live90["coverage"])
            ax.errorbar(
                [0.9],
                [live],
                yerr=np.array([[live - float(ci["low"])], [float(ci["high"]) - live]]),
                fmt="none",
                ecolor=PAPER_ORANGE,
                capsize=3,
                zorder=4,
            )
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Nominal coverage")
        ax.set_ylabel("Empirical coverage")
        ax.legend(loc="upper left")
        ax.spines[["top", "right"]].set_visible(False)
        written.append(_save_paper(fig, output_dir / "coverage-calibration.png"))

    scatter = armor.get("dilutionScatter") or {}
    if scatter.get("gap"):
        fig, ax = plt.subplots(figsize=(3.5, 2.8))
        gap = np.asarray(scatter["gap"], dtype=float)
        move = np.asarray(scatter["absMove"], dtype=float)
        floor = np.asarray(scatter["floor"], dtype=bool)
        ax.scatter(gap[~floor], move[~floor], s=8, alpha=0.45, color=PAPER_BLUE, label="non-floor")
        ax.scatter(gap[floor], move[floor], s=8, alpha=0.35, color=PAPER_ORANGE, label="floor")
        ax.set_xlabel("max_risk_estimate − risk")
        ax.set_ylabel("|y − risk|")
        ax.legend(loc="upper right", markerscale=2)
        ax.spines[["top", "right"]].set_visible(False)
        written.append(_save_paper(fig, output_dir / "dilution-probe.png"))
    else:
        quartiles = (metrics.get("dilutionProbe") or {}).get("quartiles") or []
        if quartiles:
            fig, ax = plt.subplots(figsize=(3.5, 2.6))
            xs = [f"Q{row['quartile']}" for row in quartiles]
            positions = list(range(len(xs)))
            ax.bar(positions, [float(row["floorRate"]) for row in quartiles], color=PAPER_BLUE)
            ax.set_xticks(positions, xs)
            ax.set_xlabel("Quartile of max_risk_estimate − risk")
            ax.set_ylabel("Floor rate")
            ax.spines[["top", "right"]].set_visible(False)
            written.append(_save_paper(fig, output_dir / "dilution-probe.png"))

    esa = (metrics.get("officialTest") or {}).get("esa") or {}
    if esa:
        fig, ax = plt.subplots(figsize=(3.5, 2.6))
        names = ["persistence", "xgboost", "residual", "floorHurdle", "ensemble"]
        labels = {
            "persistence": "Persist.",
            "xgboost": "XGB",
            "residual": "Residual",
            "floorHurdle": "Floor",
            "ensemble": "Ensemble",
        }
        present = [name for name in names if name in esa]
        values = [float(esa[name]["esaLoss"]) for name in present]
        bars = ax.bar([labels[name] for name in present], values, color=PAPER_BLUE)
        ax.axhline(0.694, color=PAPER_ORANGE, linestyle="--", label="LRP 0.694")
        ax.axhline(0.556, color=PAPER_GREY, linestyle=":", label="sesc 0.556")
        ax.set_yscale("log")
        ax.set_ylabel("ESA-style loss L")
        if "xgboost" in present:
            idx = present.index("xgboost")
            bars[idx].set_color(PAPER_ORANGE)
            ax.annotate(
                f"{values[idx]:.2e}",
                xy=(idx, values[idx]),
                xytext=(idx, values[idx] * 3),
                ha="center",
                fontsize=7,
                color=PAPER_ORANGE,
            )
        ax.legend(loc="upper right", fontsize=7)
        ax.spines[["top", "right"]].set_visible(False)
        written.append(_save_paper(fig, output_dir / "official-test-esa.png"))

    pred_vs = armor.get("predVsActual") or {}
    if pred_vs.get("y"):
        fig, ax = plt.subplots(figsize=(3.5, 3.2))
        y = np.asarray(pred_vs["y"], dtype=float)
        pred = np.asarray(pred_vs["pred"], dtype=float)
        floor = np.asarray(pred_vs["floor"], dtype=bool)
        ax.scatter(pred[~floor], y[~floor], s=8, alpha=0.4, color=PAPER_BLUE, label="non-floor")
        ax.scatter(pred[floor], y[floor], s=8, alpha=0.35, color=PAPER_ORANGE, label="floor")
        ax.axhline(-30.0, color=PAPER_GREY, linestyle="--", linewidth=0.8)
        ax.axvline(-30.0, color=PAPER_GREY, linestyle="--", linewidth=0.8)
        ax.plot([-32, 0], [-32, 0], color="#888888", linestyle=":", linewidth=0.8)
        ax.set_xlabel("Predicted log10(Pc)")
        ax.set_ylabel("Later reported log10(Pc)")
        ax.legend(loc="lower right", markerscale=2)
        ax.spines[["top", "right"]].set_visible(False)
        written.append(_save_paper(fig, output_dir / "pred-vs-actual.png"))

    reliability = (armor.get("floorClassifier") or {}).get("reliability") or []
    if reliability:
        fig, ax = plt.subplots(figsize=(3.2, 3.2))
        ax.plot([0, 1], [0, 1], linestyle="--", color="#888888")
        ax.plot(
            [row["predicted"] for row in reliability],
            [row["observed"] for row in reliability],
            marker="o",
            color=PAPER_BLUE,
        )
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Predicted P(floor)")
        ax.set_ylabel("Observed floor rate")
        ax.spines[["top", "right"]].set_visible(False)
        written.append(_save_paper(fig, output_dir / "floor-reliability.png"))

    flow = armor.get("datasetFlow") or {}
    if flow:
        fig, ax = plt.subplots(figsize=(3.5, 2.8))
        labels_flow = [
            "CDM rows",
            "Events",
            "Eligible T−48",
            "Train",
            "Val",
            "Cal",
            "Test",
            "Official",
        ]
        keys = [
            "cdmRows",
            "events",
            "eligibleT48",
            "train",
            "validation",
            "calibration",
            "test",
            "officialTest",
        ]
        values = [int(flow[key]) for key in keys]
        ax.barh(list(reversed(labels_flow)), list(reversed(values)), color=PAPER_BLUE)
        ax.set_xlabel("Count")
        ax.spines[["top", "right"]].set_visible(False)
        written.append(_save_paper(fig, output_dir / "dataset-flow.png"))

        fig, ax = plt.subplots(figsize=(3.5, 2.4))
        ax.axis("off")
        ax.text(
            0.5,
            0.5,
            (
                "CDM rows → cutoff-safe histories → event-disjoint split\n"
                "→ candidate models → validation selection\n"
                "→ floor hurdle + split conformal → frozen test / official test"
            ),
            ha="center",
            va="center",
            fontsize=8,
        )
        written.append(_save_paper(fig, output_dir / "protocol-flow.png"))

    ablation = (metrics.get("ablation") or {}).get("families") or {}
    if ablation:
        order = ["snapshot", "snapshot_history", "snapshot_history_covariance"]
        present = [name for name in order if name in ablation]
        if present:
            fig, ax = plt.subplots(figsize=(3.5, 2.6))
            labels_ab = ["Snapshot", "+History", "+Cov. trend"][: len(present)]
            mae = [float(ablation[name]["mae"]) for name in present]
            ax.bar(labels_ab, mae, color=PAPER_BLUE)
            ax.set_ylabel("MAE")
            ax.spines[["top", "right"]].set_visible(False)
            written.append(_save_paper(fig, output_dir / "feature-ablation-paper.png"))

    curve = (armor.get("selectivePrediction") or {}).get("curve") or []
    if curve:
        fig, ax = plt.subplots(figsize=(3.5, 2.6))
        nominal = [float(row["nominalCoverage"]) for row in curve]
        mae = [float(row["maeAccepted"]) for row in curve]
        fr = [int(row["falseReassurance"]) for row in curve]
        ax.plot(nominal, mae, marker="o", color=PAPER_BLUE)
        ax.set_xlabel("Nominal conformal coverage")
        ax.set_ylabel("Accepted MAE")
        twin = ax.twinx()
        twin.plot(nominal, fr, marker="s", color=PAPER_ORANGE)
        twin.set_ylabel("False reassurance")
        ax.set_xlim(0.48, 1.02)
        ax.spines["top"].set_visible(False)
        twin.spines["top"].set_visible(False)
        written.append(_save_paper(fig, output_dir / "selective-prediction.png"))

    shap_rows = (armor.get("residualShap") or {}).get("top") or []
    if shap_rows:
        fig, ax = plt.subplots(figsize=(3.5, 3.0))
        names = [row["feature"][:22] for row in reversed(shap_rows)]
        vals = [float(row["meanAbsShap"]) for row in reversed(shap_rows)]
        ax.barh(names, vals, color=PAPER_BLUE)
        ax.set_xlabel("Mean |SHAP|")
        ax.spines[["top", "right"]].set_visible(False)
        written.append(_save_paper(fig, output_dir / "residual-shap.png"))
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
