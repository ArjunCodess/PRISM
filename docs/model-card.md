# PRISM model card

**Intended use.** Education, interpretability demonstration, and research into early conjunction-risk forecasting. Educational prototype only.

**Out of scope.** Spacecraft operations, autonomous manoeuvres, claims about specific real satellites, safety-critical decisions, and live Indian catalogue data without retraining.

**Threshold.** High-risk class is log10(Pc) ≥ −6. This matches the ESA challenge scoring class. It is not an ISRO operational rule. Typical LEO reaction bands are 10⁻⁵ to 10⁻⁴.

**Split.** Grouped by event_id into train / validation / calibration / test. Calibration never sees test labels.

**Human control.** Forecasts are advisory. REVIEW REQUIRED means a person must look. The model never commands a manoeuvre.

See `ml/artifacts/metrics.json` for the frozen numbers.
