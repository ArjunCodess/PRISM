# Validation and reproducibility record

The working tree was clean before revision. The exact FMTS submission is `091d7c49fa74d28405c4cb1c9cfff4b9ab793712`, preserved with annotated tag `fmts-2026-submitted`. Branch: `fmts-review-improvements`. No push or pull request was made. The submitted package and submission tag preserve the original state; redundant manuscript copies were removed during cleanup.

## Final commands

Run commands from the repository root. The experiments use `experiments/fmts_review.json`, seeds 42–46, CPU models, and the versions in `experiments/requirements-fmts-review.txt`. Raw data downloads are ignored by Git; hashes are in `input_provenance.json`. All new research outputs are separate from deployed artifacts.

| Command | Exit status and result |
|---|---|
| `python scripts/audit_fmts_sources.py` | 0. Official release, raw archive, metadata and nine DOI lookups audited. One Crossref DOI lookup returns 404 and has an explicitly recorded publisher fallback. |
| `python ml/src/fmts_review.py --config experiments/fmts_review.json --dataset esa` | 0. Five local grouped redraws, seed-42 official transfer evaluation, all policies and calibrated sets. Final run asserts original seed-42 split IDs. |
| `python ml/src/fmts_review.py --config experiments/fmts_review.json --dataset simulation` | 0. All 60 mechanism/mass/drift/seed combinations retained, including absent-tail and failed-remedy results. |
| `python ml/src/fmts_review.py --config experiments/fmts_review.json --dataset outputs` | 0. Figure, table and manifest regenerated from JSON. |
| `python scripts/summarize_fmts_revision.py` | 0. Paired stratum intervals, calibration comparison, simulation aggregates and embedded manuscript numbers generated. |
| `python -m pytest ml/tests/test_fmts_review.py -k 'not generated and not saved and not all_simulations'` | 0. Initial 10 algorithm tests passed; three artifact tests deselected before outputs were complete. Existing cache permissions caused a warning; later full runs disable the cache plugin. |
| `python -m pytest -p no:cacheprovider` | 0 on both full runs: initially 74 passed, then 77 passed after manuscript, reference and simulation-consistency checks were added. Legacy API loads an isotonic estimator saved with scikit-learn 1.9.0 into 1.8.0 and emits one version warning. Revision experiments do not deserialize that estimator. |
| `python -m ruff check ml/src/fmts_remedies.py ml/src/fmts_review.py ml/tests/test_fmts_review.py scripts/audit_fmts_sources.py scripts/summarize_fmts_revision.py` | 0 after formatting and import fixes. |
| `git diff --check` | 0. Git reports only its normal LF-to-CRLF notice for the edited supplement. |

## Development attempts and failures

The first source lookup using a direct Python network call failed with exit 1 due to sandbox network restrictions. Authorized elevated retrieval succeeded with exit 0, then the source-audit script completed with exit 0. A raw-document preview initially failed with exit 1 because the terminal encoding could not print a Unicode minus; the escaped-text preview succeeded with exit 0. These were acquisition/inspection failures, not experimental results.

The initial ESA command completed with exit 0 but omitted the legacy eligibility filter, producing a different cohort. That run was rejected before manuscript generation. The filter and exact frozen-split assertion were added, and the same command reran with exit 0 on the correct 8,293-event cohort. No numerical claim in the revision uses the discarded cohort. The simulation command ran twice with exit 0: the second run used the full 19-threshold grid when refitting the prior hurdle, rather than the shorter new-policy grid.

The first summary-generation command failed with exit 1 due to a string-escaping syntax error. The string was corrected; subsequent summary runs completed with exit 0. Initial Ruff checks returned exit 1 for import ordering, unused imports and long lines. `ruff check --fix` resolved imports but still reported long lines; `ruff format` completed with exit 0, and final checks passed with exit 0. These failures are disclosed rather than counted as successful validation.

The built-in `compile_latex_document` tool failed on the pre-edit manuscript, after revision, and during the final build with `Unable to find standard directories for platform`. This is a compiler initialization failure. The existing MiKTeX installation successfully rebuilt both PDFs with `./scripts/compile-paper.ps1`; no TeX distribution was installed. An initial sandboxed invocation returned exit 1 because MiKTeX could not access its user configuration, and an intermediate rebuild returned exit 1 on a transient locked supplement PDF. The final rebuild returned exit 0. The editable source remains open in Codex. The original PDFs remain in the submitted package and Git history. Native tool results do not have shell exit codes.

## Final build before commits

The five ESA configurations and all 60 simulations were rerun from the checked-in configuration with exit 0 before committing. Outputs and manuscript summaries were regenerated with exit 0. The full Python suite passed again: 77 tests, with the same legacy estimator-version warning.

| Command | Exit status and result |
|---|---|
| `./scripts/compile-paper.ps1` | 0 using installed MiKTeX: five-page `paper/main.pdf` and one-page `paper/supplement.pdf`. |
| `npm.cmd run build` in `apps/web` | 0: Next.js production build, type checks and static generation. Initial restricted-network attempt returned 1 fetching the Google font; authorized network access succeeded. |
| `npm.cmd run lint` in `apps/web` | 0: generated `.venv` dependencies are excluded from source linting. |
| `npm.cmd test` in `apps/web` | 0: two Vitest tests. Initial sandboxed attempt returned 1 because esbuild could not access the parent directory; authorized execution succeeded. |
| `python -m compileall -q ml/src apps/api/main.py scripts/audit_fmts_sources.py scripts/summarize_fmts_revision.py` | 0: Python sources compile. |

After the completion audit, `python -m pytest -p no:cacheprovider` passed 144 tests with the same single legacy estimator-version warning. The added checks recompute every saved run's model metrics, event IDs, marginal and category coverage in all strata, and exact binomial intervals. They also verify every report hash and the final manuscript/PDF hashes. `python -m ruff check ml/tests/test_fmts_review.py --fix` and `python -m ruff format ml/tests/test_fmts_review.py` returned 0.

All six final PDF pages were rendered with `pdftoppm -r 95 -png` and visually checked for clipping, missing figures, unreadable tables and unresolved citations. Final compiler logs contain no overfull boxes or undefined references. MiKTeX emits a locale fallback notice. `paper/build_manifest.json` records final source/PDF hashes and page counts. No push or pull request was made.

## Licenses, splits and artifacts

ESA Zenodo release DOI 10.5281/zenodo.4463683 specifies CC BY 4.0 in the retrieved API metadata. PRISM code and generated simulations use MIT. UCI Air Quality was inspected but not downloaded or modeled: its page calls −200 missingness and contains both a CC BY 4.0 license panel and older research-only wording. No claim about a detection floor or resolution of that license inconsistency is made. EPA documentation was inspected, but no EPA data were modeled. Simulation is not a second empirical domain.

The raw archive hash matches the prior provenance, `df1500146705305006ea506eeccc97f5c2e9593928d6650c503c4f5b5529d0bb`. The full raw table has no calendar columns or processing code. All original split IDs are retained on seed 42; each other redraw partitions eligible events. The ordered simulation splitter groups repeated observations and tied timestamps. ESA chronology and object-pair grouping remain unresolved because identifiers are unavailable, not because random IDs are treated as time.

`ml/artifacts/fmts_review` contains:

- `configuration.json`, `runtime.json` and `input_provenance.json` record configuration, source and input hashes and package versions.
- `zenodo_metadata.json`, `verified_references.json`, `publisher_reference_checks.json` and `archive_audit.json` establish source provenance and semantic limits.
- `esa_seed42.json` through `esa_seed46.json` contain splits, parameters, all validation candidates, model metrics and uncertainty. Seed 42 also contains official transfer results.
- `*_predictions.npz` contain per-event truth, prediction, probability and prediction-set components. They use numeric arrays and do not require pickle loading.
- Sixty `sim_*.json` reports preserve negative results and undefined tail metrics. `simulation_summary.json` and `simulation_aggregate.json` retain all combinations.
- `paired_strata.json` contains paired errors within strata, paired coverage differences and calibrated-minus-raw Brier differences.
- `table_manifest.json` hashes the source JSON, generated table and main figure. `artifact_hashes.json` hashes the other JSON artifacts.

The embedded generated snippets in `paper/main.tex` match `fmts_review_numbers.tex` and `fmts_review_table.tex`. Table tests recompute metrics and coverage from saved predictions, check additive decomposition, and verify content hashes. The main figure was visually inspected after generation.

## Statistical intervals and failed remedies

Reported error and cost intervals use 1,000 paired event resamples. ESA intervals omit and count draws with zero F2 or no positive target; conditional-on-defined intervals must not be read as unconditional intervals. Coverage uses exact 95% binomial intervals. Seed standard deviations describe split sensitivity and are not inferential confidence intervals. The manuscript reports selected estimates; complete intervals for all models are in the JSON.

For the selected guarded remedy, overall model-minus-persistence MAE is −0.557 [−0.771,−0.342] locally and −0.857 [−1.017,−0.704] officially. Non-floor differences are −0.764 [−1.397,−0.193] and −1.095 [−1.550,−0.640]. ESA loss differences are zero, with 998 valid local draws and two undefined draws; all 1,000 official draws are valid. Official raw high-risk MAE difference is +0.267 [−0.295,0.781], which does not establish improvement. Local calibrated-minus-raw atom Brier difference is −0.0279 [−0.0361,−0.0201].

Direct-model official high-risk category coverage is 140/150, 93.3% [88.1%,96.8%], versus 24.0% marginal coverage. The selected remedy still accepts nine falsely reassuring official forecasts. Category sets around the prior hurdle require review on every event. The mixture has zero local and official tail recall. Known-clipping simulation can have no tail positives; such runs have undefined ESA scores and are not dropped. The negative-remedy ledger reports these failures and the unresolved −30 rule.

## Documentation cleanup

Redundant manuscript copies and documents repeating the paper were removed at the author's request. The claim audit is consolidated into the response matrix; the submitted package and tag remain unchanged. After fixing references, `./scripts/compile-paper.ps1` returned 0 and both PDFs were rendered and checked. An intermediate supplement had a 0.55-point line overflow; the manifest check rejected it and one premature test run failed against the old build hashes. After correcting the layout and refreshing the manifest, `python -m pytest -p no:cacheprovider` returned 0 with 144 tests passed and the single existing estimator-version warning. Final logs contain no overfull boxes or undefined references.
