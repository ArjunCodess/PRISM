import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from calibrate import split_conformal_quantile
from fmts_remedies import (
    GaussianTobit,
    atom_mask,
    categories,
    exact_esa,
    mondrian_quantiles,
    ordered_event_split,
    point_report,
    prediction_set,
    set_covered,
)
from fmts_review import validate_split

ROOT = Path(__file__).resolve().parents[2]
CFG = json.loads((ROOT / "experiments/fmts_review.json").read_text())
OUT = ROOT / "ml/artifacts/fmts_review"


def test_floor_is_atom_not_arbitrary_low_tail():
    assert atom_mask(np.array([-30.0, -29.0, -6.0])).tolist() == [True, False, False]
    with pytest.raises(ValueError):
        atom_mask(np.array([-31.0]))
    with pytest.raises(ValueError):
        atom_mask(np.array([np.nan]))


def test_categories_do_not_overlap_at_boundaries():
    assert categories(np.array([-30.0, -29.99, -6.001, -6.0, 0.0])).tolist() == [0, 1, 1, 2, 2]


def test_temporal_split_keeps_repeated_observations_and_ties_together():
    frame = pd.DataFrame(
        {"event_id": np.repeat(np.arange(100), 4), "event_time": np.repeat(np.arange(100) // 2, 4)}
    )
    manifest = ordered_event_split(frame)
    validate_split(frame, manifest)
    last = None
    for ids in manifest.values():
        times = frame[frame.event_id.isin(ids)].event_time
        if last is not None:
            assert times.min() > last
        last = times.max()
        assert all(frame[frame.event_id == event].event_time.isin(times).all() for event in ids)


def test_split_rejects_leaked_event():
    with pytest.raises(ValueError, match="leakage"):
        validate_split(pd.DataFrame({"event_id": [0, 1]}), {"train": [0], "test": [0, 1]})


def test_decomposition_is_additive_not_sum_of_stratum_means():
    y = np.array([-30.0, -30.0, -5.0])
    pred = np.array([-29.0, -27.0, -7.0])
    report = point_report(y, pred, CFG)
    assert report["mae"] == pytest.approx(sum(report["mae_contributions"].values()))
    assert report["mae_contributions"]["floor"] == pytest.approx(4 / 3)
    assert report["mae_contributions"]["nonfloor"] == pytest.approx(2 / 3)


def test_official_metric_has_no_hidden_epsilon():
    result = exact_esa(np.array([-5.0, -30.0]), np.array([-10.0, -30.0]))
    assert result["esa_loss"] is None
    assert result["esa_undefined"] == "F2=0 (infinite loss)"
    result = exact_esa(np.array([-5.0, -30.0]), np.array([-5.0, -30.0]))
    assert result["esa_loss"] == 0


def test_candidate_inversion_does_not_require_future_category():
    cfg = {**CFG, "alpha": 0.1}
    quantiles = {"0": {"n": 100, "q": 0}, "1": {"n": 100, "q": 2}, "2": {"n": 0, "q": float("inf")}}
    sets = prediction_set(np.array([-30.0, -10.0]), quantiles, cfg)
    assert sets["atom"].tolist() == [True, False]
    assert sets["review"].tolist() == [True, True]
    assert set_covered(np.array([-5.0, -5.0]), sets, cfg).all()
    assert sets["low_hi"][1] < -6


def test_conformal_finite_sample_rank_and_small_group():
    assert split_conformal_quantile(np.arange(9.0), 0.1) == 8
    assert np.isinf(split_conformal_quantile(np.arange(8.0), 0.1))
    assert np.isinf(split_conformal_quantile(np.array([]), 0.1))


def test_mondrian_exchangeable_category_coverage():
    # A repeated, independent exchangeable simulation tests behavior, not a theorem by assertion.
    rng = np.random.default_rng(71)
    covered = [[] for _ in range(3)]
    for _ in range(150):
        y_cal = np.r_[np.full(100, -30.0), rng.uniform(-25, -7, 100), rng.uniform(-6, 0, 100)]
        pred_cal = y_cal + rng.normal(0, 3, 300)
        q = mondrian_quantiles(y_cal, pred_cal, CFG)
        y_test = np.r_[np.full(100, -30.0), rng.uniform(-25, -7, 100), rng.uniform(-6, 0, 100)]
        pred_test = y_test + rng.normal(0, 3, 300)
        mask = set_covered(y_test, prediction_set(pred_test, q, CFG), CFG)
        for g in range(3):
            covered[g].extend(mask[g * 100 : (g + 1) * 100].tolist())
    assert all(0.88 < np.mean(values) < 0.93 for values in covered)


def test_known_clipping_tobit_observed_expectation():
    rng = np.random.default_rng(7)
    x = rng.normal(size=(500, 1))
    y = np.maximum(-30, -28 + 3 * x[:, 0] + rng.normal(size=500))
    model = GaussianTobit().fit(x, y)
    assert model.converged
    predicted = model.predict(np.array([[-10.0], [0.0], [1.0]]))
    assert np.all(predicted >= -30)
    assert predicted[0] == pytest.approx(-30)
    assert predicted[2] > predicted[1]


def test_generated_tables_and_decomposition_match_artifacts():
    manifest = json.loads((OUT / "table_manifest.json").read_text())
    for path, key in [
        (OUT / "esa_seed42.json", "source_sha256"),
        (ROOT / "paper/fmts_review_table.tex", "table_sha256"),
        (ROOT / "docs/figures/fmts-remedy-decomposition.png", "figure_sha256"),
    ]:
        assert hashlib.sha256(path.read_bytes()).hexdigest() == manifest[key]
    report = json.loads((OUT / "esa_seed42.json").read_text())
    table = (ROOT / "paper/fmts_review_table.tex").read_text()
    for name, row in report["evaluations"]["test"]["models"].items():
        assert row["mae"] == pytest.approx(sum(row["mae_contributions"].values()))
        assert f"{row['mae']:.3f}" in table
        assert name.replace("_", " ") in table


def test_saved_coverage_and_point_scores_recompute():
    report = json.loads((OUT / "esa_seed42.json").read_text())
    for population in ("test", "official"):
        arrays = np.load(OUT / f"esa_seed42_{population}_predictions.npz")
        for name, row in report["evaluations"][population]["models"].items():
            recomputed = point_report(arrays["y"], arrays[name], CFG)
            assert recomputed == row
            sets = {
                key: arrays[name + "_set_" + key]
                for key in ("atom", "low_lo", "low_hi", "high_lo", "high_hi")
            }
            covered = set_covered(arrays["y"], sets, CFG)
            expected = report["evaluations"][population]["intervals"][name]["coverage"]["all"][
                "mondrian"
            ]
            assert int(covered.sum()) == expected["covered"]


def test_all_simulations_are_present_and_splits_are_valid():
    simulations = list(OUT.glob("sim_*.json"))
    assert len(simulations) == 60
    for path in simulations:
        payload = json.loads(path.read_text())
        ids = [event for events in payload["split"].values() for event in events]
        assert len(ids) == len(set(ids)) == CFG["simulation"]["n_events"]


def test_embedded_manuscript_numbers_and_table_are_generated():
    content = (ROOT / "paper/main.tex").read_text(encoding="utf-8")
    numbers = (ROOT / "paper/fmts_review_numbers.tex").read_text(encoding="utf-8")
    table = (ROOT / "paper/fmts_review_table.tex").read_text(encoding="utf-8")
    assert numbers in content
    assert table in content


def test_saved_semantically_different_simulations_are_observationally_identical():
    # Non-identifiability demonstration, not independent empirical validation.
    for mass in (0.2, 0.8):
        for drift in (0, 1):
            prefix = f"mass{mass}_drift{drift}_seed42_test_predictions.npz"
            a = np.load(OUT / ("sim_structural_atom_" + prefix))
            b = np.load(OUT / ("sim_missing_update_" + prefix))
            assert np.array_equal(a["y"], b["y"])
            assert np.array_equal(a["persistence"], b["persistence"])


def test_frozen_split_preserved_and_all_source_dois_have_checks():
    report = json.loads((OUT / "esa_seed42.json").read_text())
    frozen = json.loads((ROOT / "ml/artifacts/split_manifest.json").read_text())
    assert all(set(report["split"][key]) == set(frozen[key]) for key in frozen)
    checked = json.loads((OUT / "verified_references.json").read_text())
    publisher = json.loads((OUT / "publisher_reference_checks.json").read_text())
    fallback = {source.get("doi") for source in publisher["sources"]}
    assert all("DOI" in record or doi in fallback for doi, record in checked.items())
