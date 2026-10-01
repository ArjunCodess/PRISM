"""Generate submission-specific figures without rewriting the repository figures."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "figures"


def main() -> None:
    metrics = json.loads((ROOT / "ml" / "artifacts" / "metrics.json").read_text(encoding="utf-8"))
    anatomy = metrics["errorAnatomy"]
    edges = np.asarray(anatomy["binEdges"], dtype=float)
    centers = (edges[:-1] + edges[1:]) / 2
    width = edges[1] - edges[0]

    fig, ax = plt.subplots(figsize=(4.2, 2.7))
    ax.bar(
        centers,
        anatomy["actualMoveCounts"],
        width=width,
        color="#666666",
        alpha=0.58,
        label=r"$y-r$ (later movement)",
    )
    ax.bar(
        centers,
        anatomy["residualErrorCounts"],
        width=width,
        color="#2c7fb8",
        alpha=0.48,
        label=r"$y-\hat{y}$ (model residual)",
    )
    ax.axvline(0.0, color="#d95f02", linestyle="--", linewidth=1.0)
    peak = max(anatomy["actualMoveCounts"])
    ax.annotate(
        r"$-30$ floor collapses",
        xy=(-20.0, peak * 0.16),
        xytext=(-31.0, peak * 0.54),
        fontsize=8,
        color="#d95f02",
        arrowprops={"arrowstyle": "->", "color": "#d95f02", "lw": 0.8},
    )
    ax.text(1.8, peak * 0.22, "non-floor", fontsize=8, color="#006d9c")
    ax.set_xlabel(r"Change in reported $\log_{10} P_c$")
    ax.set_ylabel("Events")
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=2,
        frameon=False,
        fontsize=7,
        handlelength=1.8,
        columnspacing=1.2,
    )
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(pad=0.6)
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "error-anatomy-fmts.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
