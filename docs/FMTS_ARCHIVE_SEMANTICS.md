# What the ESA value −30 establishes

The Zenodo release [10.5281/zenodo.4463683](https://zenodo.org/records/4463683) supplies the challenge tables and a raw anonymized table. Its metadata states CC BY 4.0. `scripts/audit_fmts_sources.py` records the API metadata, release hash, member inventory, and comparison in `ml/artifacts/fmts_review`.

The challenge documentation defines risk as a self-computed base-10 log value at each message epoch. It does not document how −30 is generated. The released raw table contains 199,082 rows and 15,321 events, with 82,484 rows at −30, no lower values, and no missing risk values. The competition training table has 67,240 such rows out of 162,634. These are full-row counts, not eligible final-target counts.

Event IDs were randomized and are not a join key between release variants. Matching unique raw rows on relative message time, mission, miss distance, and speed yields 126,235 training matches. Of these, 57,439 competition rows at −30 are also −30 in the released raw data; none map to a lower raw value and no matched risk differs. This is consistent with the atom existing before competition packaging. It does not show when or why the original risk computation produced it. The matching is a partial diagnostic, not a claimed bijection.

The release contains no executable preprocessing code. Its raw-data text describes columns without mentioning −30, clipping, or a floor. Absolute timestamps have already been replaced by time relative to closest approach, including in the raw table. The archive audit found no calendar columns. The challenge publication explains the anonymization and refers to the negligible-risk mass, but that does not establish a mathematical censoring rule.

| Possible meaning | Evidence and inference boundary |
|---|---|
| A precisely computed probability of 10^-30 | Not established. The atom could represent exact output, numerical saturation, or a convention. |
| A reporting floor or clipping rule | A sharp lower boundary is observed, but no released rule proves latent values are below it. |
| A missing update or invalid computation sentinel | No missing risk cells are present, but a sentinel could be encoded numerically. No provenance distinguishes it. |
| An archive convention for negligible values | Plausible, not documented sufficiently to assert. |
| A PRISM preprocessing artifact | Contradicted by the atom in the released raw data and unchanged uniquely matched messages. |

The revision models the recorded atom and forecasts later reported risk. It does not infer latent probabilities, collision outcomes, or manoeuvre utility. The name `NEGLIGIBLE_RISK` in legacy code is a constant name, not evidence of semantics. Legacy models remain frozen for reproducibility. The revised exact-atom labeling rejects out-of-support values instead of silently merging all values below −30.

Resolving the ambiguity requires ESA's original risk-computation and export rule or a statement from the data producers. No such statement has been obtained; no message was sent to them. Calendar validation requires identifiable event timestamps not present in this public release. Those two limitations cannot be fixed by renaming random IDs or training another model.
