# PR title

Add floor-aware remedies and generalization after FMTS review

# PR body

## Summary
- position PRISM against the established censoring, mixture-model, and decision-focused forecasting literature
- audit the operational meaning of the ESA −30 atom, document unresolved semantics, and correct unsupported claims
- add floor-aware modeling, calibration, and event-level evaluation with paired uncertainty; the selected guarded remedy improves non-floor error while tying persistence on ESA loss
- expand the evidence with 60 controlled mechanism simulations and remove unsupported temporal-foundation-model framing; no second empirical domain is claimed

## Validation
- `python scripts/audit_fmts_sources.py` — exit 0
- `python ml/src/fmts_review.py --config experiments/fmts_review.json --dataset esa` — exit 0; five grouped redraws and official transfer evaluation
- `python ml/src/fmts_review.py --config experiments/fmts_review.json --dataset simulation` — exit 0; all 60 runs retained
- `python ml/src/fmts_review.py --config experiments/fmts_review.json --dataset outputs` — exit 0
- `python scripts/summarize_fmts_revision.py` — exit 0
- `python -m pytest ml/tests/test_fmts_review.py -k 'not generated and not saved and not all_simulations'` — exit 0; initial algorithm check, 10 passed
- `python -m pytest -p no:cacheprovider` — exit 0; final 144 passed, one legacy estimator-version warning; every saved experiment's scores, stratum coverage, intervals and event IDs recomputed
- `python -m ruff check ml/src/fmts_remedies.py ml/src/fmts_review.py ml/tests/test_fmts_review.py scripts/audit_fmts_sources.py scripts/summarize_fmts_revision.py` — exit 0
- `git diff --check` — exit 0
- ESA release: CC BY 4.0; code and simulated data: MIT. No other empirical dataset was modeled.
- Original seed-42 event IDs are preserved; all splits are disjoint. Simulation uses ordered event blocks. ESA calendar and object-pair validation remain unavailable.
- Outputs: `ml/artifacts/fmts_review`, generated manuscript table/numbers, and the additive floor/non-floor figure with a separate decision-tail panel. Paired percentile bootstrap and exact binomial coverage intervals are included.
- Negative results: calibrated mixture has zero local and official high-risk recall; category sets around the old hurdle require universal review; the selected remedy still accepts nine falsely reassuring official forecasts. Archive generation semantics remain unresolved.
- `./scripts/compile-paper.ps1` — exit 0 using installed MiKTeX; rebuilt five-page main paper and one-page supplement. The native editor compiler still fails to initialize; pre-review PDFs are preserved.
- `npm.cmd run build`, `npm.cmd run lint`, and `npm.cmd test` in `apps/web` — exit 0; production build and two frontend tests pass.
- `python -m compileall -q ml/src apps/api/main.py scripts/audit_fmts_sources.py scripts/summarize_fmts_revision.py` — exit 0.
- `docs/FMTS_VALIDATION.md` records every command family, earlier failed checks and discarded development runs, licenses, split limitations, artifact hashes and intervals. No push or pull request was made.
