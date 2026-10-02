"""Build manuscript numbers and paired diagnostics from saved event-level outputs."""

import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "ml/artifacts/fmts_review"


def paired_mean(values, seed=42, draws=1000):
    rng = np.random.default_rng(seed)
    estimates = [
        float(values[rng.integers(0, len(values), len(values))].mean()) for _ in range(draws)
    ]
    return {
        "difference": float(values.mean()),
        "ci95": np.quantile(estimates, [0.025, 0.975]).tolist(),
        "n_events": len(values),
        "n_draws": draws,
    }


def main():
    config = json.loads((ROOT / "experiments/fmts_review.json").read_text())
    esa = json.loads((OUT / "esa_seed42.json").read_text())
    extra = {}
    for population in ("test", "official"):
        data = np.load(OUT / f"esa_seed42_{population}_predictions.npz")
        y, baseline = data["y"], data["persistence"]
        masks = {
            "all": np.ones(len(y), bool),
            "floor": np.isclose(y, -30),
            "nonfloor": ~np.isclose(y, -30),
            "high": y >= -6,
        }
        extra[population] = {}
        for name in esa["evaluations"][population]["models"]:
            delta = np.abs(y - data[name]) - np.abs(y - baseline)
            extra[population][name] = {
                "mae_model_minus_persistence": {
                    group: paired_mean(delta[mask]) for group, mask in masks.items()
                },
                "coverage_mondrian_minus_marginal": paired_mean(
                    np.where(
                        np.isclose(y, -30),
                        data[name + "_set_atom"],
                        np.where(
                            y >= -6,
                            (y >= data[name + "_set_high_lo"]) & (y <= data[name + "_set_high_hi"]),
                            (y >= data[name + "_set_low_lo"]) & (y <= data[name + "_set_low_hi"]),
                        ),
                    ).astype(float)
                    - data[name + "_marginal_covered"].astype(float)
                ),
            }
        actual = np.isclose(y, -30).astype(float)
        extra[population]["brier_calibrated_minus_raw"] = paired_mean(
            (actual - data["calibrated_floor_probability"]) ** 2
            - (actual - data["raw_floor_probability"]) ** 2
        )
    (OUT / "paired_strata.json").write_text(json.dumps(extra, indent=2))

    grouped = {}
    for mechanism in config["simulation"]["mechanisms"]:
        for rate in config["simulation"]["floor_rates"]:
            for drift in config["simulation"]["drift"]:
                paths = sorted(OUT.glob(f"sim_{mechanism}_mass{rate}_drift{int(drift)}_seed*.json"))
                reports = [json.loads(path.read_text()) for path in paths]
                key = f"{mechanism}/mass={rate}/drift={drift}"
                grouped[key] = {"runs": len(reports), "models": {}}
                for model in reports[0]["evaluations"]["test"]["models"]:
                    rows = [r["evaluations"]["test"]["models"][model] for r in reports]
                    uncertainty = [r["evaluations"]["test"]["intervals"][model] for r in reports]
                    grouped[key]["models"][model] = {
                        "mae_mean": float(np.mean([r["mae"] for r in rows])),
                        "mae_seed_sd": float(np.std([r["mae"] for r in rows], ddof=1)),
                        "esa_defined_runs": sum(r["esa_loss"] is not None for r in rows),
                        "high_counts": [r["tp"] + r["fn"] for r in rows],
                        "mean_decision_cost": float(np.mean([r["decision_cost"] for r in rows])),
                        "marginal_coverage_mean": float(
                            np.mean(
                                [r["coverage"]["all"]["marginal"]["coverage"] for r in uncertainty]
                            )
                        ),
                        "mondrian_coverage_mean": float(
                            np.mean(
                                [r["coverage"]["all"]["mondrian"]["coverage"] for r in uncertainty]
                            )
                        ),
                        "mean_review_fraction": float(
                            np.mean([r["review_fraction"] for r in uncertainty])
                        ),
                    }
    (OUT / "simulation_aggregate.json").write_text(json.dumps(grouped, indent=2))

    values = {}
    for population, prefix in (("test", "Local"), ("official", "Official")):
        metrics = esa["evaluations"][population]
        for model, tag in (
            ("persistence", "Persist"),
            ("decision_hurdle", "Remedy"),
            ("direct", "Direct"),
            ("old_hurdle", "Old"),
        ):
            row = metrics["models"][model]
            for metric, suffix in (
                ("mae", "MAE"),
                ("nonfloor_mae", "Nonfloor"),
                ("floor_mae", "Floor"),
                ("esa_loss", "Loss"),
            ):
                values[prefix + tag + suffix] = (
                    r"\infty" if row[metric] is None else f"{row[metric]:.3f}"
                )
        direct = metrics["intervals"]["direct"]["coverage"]["high"]
        values[prefix + "TailMarginal"] = f"{100 * direct['marginal']['coverage']:.1f}"
        values[prefix + "TailMondrian"] = f"{100 * direct['mondrian']['coverage']:.1f}"
        values[prefix + "RemedyReview"] = (
            f"{100 * metrics['intervals']['decision_hurdle']['review_fraction']:.1f}"
        )
        values[prefix + "RemedyFR"] = str(
            metrics["intervals"]["decision_hurdle"]["accepted_false_reassurance"]
        )
    macros = "\n".join(
        "\\newcommand{\\" + key + "}{" + value + "}" for key, value in values.items()
    )
    (ROOT / "paper/fmts_review_numbers.tex").write_text(macros, encoding="utf-8")
    # The standalone draft embeds generated snippets. It can be edited in the native editor.
    draft = ROOT / "paper/main.tex"
    if draft.exists():
        content = draft.read_text(encoding="utf-8")
        start, end = "% BEGIN GENERATED NUMBERS", "% END GENERATED NUMBERS"
        if start in content:
            left, rest = content.split(start, 1)
            _, right = rest.split(end, 1)
            content = left + start + "\n" + macros + "\n" + end + right
        start, end = "% BEGIN GENERATED TABLE", "% END GENERATED TABLE"
        if start in content:
            left, rest = content.split(start, 1)
            _, right = rest.split(end, 1)
            content = (
                left
                + start
                + "\n"
                + (ROOT / "paper/fmts_review_table.tex").read_text()
                + "\n"
                + end
                + right
            )
        draft.write_text(content, encoding="utf-8")
    provenance = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(OUT.glob("*.json"))
        if path.name != "artifact_hashes.json"
    }
    (OUT / "artifact_hashes.json").write_text(json.dumps(provenance, indent=2))
    print("Generated paired strata, simulation aggregate, and manuscript numbers.")


if __name__ == "__main__":
    main()
