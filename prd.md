# PRISM Product Requirements Document

**Predictive Risk Intelligence for Space Monitoring**  
An explainable AI copilot for T-48-hour space-debris conjunction risk forecasting

| Field | Value |
|---|---|
| Product | PRISM |
| Acronym | Predictive Risk Intelligence for Space Monitoring |
| Type | Educational research prototype and interactive decision-support exhibit |
| Campus | City Montessori School, Kanpur Road Campus |
| Competition topic | Space-debris conjunction / collision risk (NETRA) |
| Audience | ISTRAC/ISRO-style experts, teachers, students, judges |
| Document version | 1.0 |
| Status | Locked specification for the 18 August 2026 exhibit |
| Freeze date | Evening of 17 August 2026 |
| Owner | Arjun Vijay Prakash |
| Safety | Educational research only. Never present as flight-certified collision-avoidance software. |

### Addendum — 18 August 2026 (exhibit as built)

Version 1.0 below remains the original contract. The frozen exhibit is:

1. **Selected model** is a T−48 bootstrap XGBoost median of the later reported `log10(Pc)`, with a persistence guard at `log10(Pc) ≥ −6`.
2. **Held-out numbers:** persistence MAE 5.080 → ensemble MAE 3.059; ESA-style loss and F2 still tie at 0.167 / 0.361. Unguarded XGBoost is 2.808 with F2 = 0. Nominal 90% bootstrap coverage is 47.7%.
3. **G6 / A8 / N7 superseded.** The website requires FastAPI (`NEXT_PUBLIC_API_URL`). There is no JSON fallback when the API is stopped. The laptop exhibit still runs with Wi-Fi off if both processes are local.
4. **False reassurance is 1** on the frozen test split (1 of 9 high-risk events).
5. **Snapshot features** include encounter-plane geometry and object-type dummies.
6. **M3 remains unmet:** MAE improves; ESA-style loss does not.

Do not quote 3.052. That was an earlier freeze of the same T−48 policy before this retrain.

Numbers and scripts: `README.md`, `docs/model-card.md`, `docs/presentation.md`, `ml/artifacts/metrics.json`.

---

## 1. Purpose of this document

This PRD is the build contract for PRISM. It defines the problem, the product a judge must see, the scientific constraints that make the result defensible, and the acceptance tests that decide whether the exhibit is ready.

If a later design choice conflicts with this document, this document wins unless the change is written here.

---

## 2. Executive summary

PRISM forecasts how serious a satellite–debris close approach will look at the last available Conjunction Data Message (CDM), using only information that would have been known **at least 48 hours before closest approach**.

The product converts a time series of CDMs into four judge-visible outputs:

1. A forecast of final log collision risk, \(\hat{y}_e = \widehat{\log_{10}(P_{c,e}^{\text{final}})}\).
2. A calibrated probability that the event belongs to the high-risk class.
3. An uncertainty range, or an explicit **REVIEW REQUIRED** abstention.
4. A SHAP-based explanation in both chart and plain language.

The scientific question is operational, not decorative: **can we identify encounters that are likely to become dangerous before the final, more accurate observations arrive?** ESA posed this exact forecasting problem in the public Collision Avoidance Challenge. Operators typically start thinking about a manoeuvre about two days before closest approach; waiting gives better data, but less time to act.

PRISM is a **copilot**, not an authority. A human remains in control. The model never recommends firing thrusters.

---

## 3. Problem

### 3.1 Operational problem

Two orbiting objects can pass close enough to generate a conjunction alert. Their future positions are uncertain, so the reported collision probability can rise or fall as radar and optical observations shrink the covariance. Early estimates are noisy. Late estimates are better, but by then a manoeuvre may be expensive, disruptive, or too late.

ISRO’s Indian Space Situational Awareness Report for 2025 makes the scale concrete: about 160,000 close-approach alerts were recorded globally, and ISRO analysed more than 150,000 Combined Space Operations Center alerts against more accurate operational orbit data. Eighteen collision-avoidance manoeuvres were performed (14 in LEO, including NISAR, and 4 in GEO). India’s NETRA control centre at ISTRAC, Peenya, exists to centralise this work.

### 3.2 Modelling problem

Given all CDMs for event \(e\) with

\[
\text{time\_to\_tca} \ge 2\ \text{days},
\]

predict the event’s final reported risk from the last available CDM:

\[
\hat{y}_e = \widehat{\log_{10}(P_{c,e}^{\text{final}})}.
\]

- \(P_c\) is collision probability.
- The ESA dataset stores `risk` already in base-10 log form.
- The primary task is **regression**.
- A secondary calibrated classification view answers \(P(\text{high risk} \mid X_e)\).

### 3.3 Why a naive answer is not enough

The strongest simple baseline is persistence: copy the latest pre-T-48 `risk` forward. In the original ESA challenge this Latest Risk Prediction (LRP) baseline scored \(L = 0.694\). Only 12 of 96 teams beat it. PRISM may claim “AI improvement” only if it beats this baseline on an untouched event-level test set.

---

## 4. Goals and non-goals

### 4.1 Goals

| ID | Goal | How we know it is met |
|---|---|---|
| G1 | Forecast final log-risk from pre-T-48 CDM history. | Event-level regressor runs on cutoff-safe features. |
| G2 | Beat persistence on held-out events. | Lower MAE than LRP, and lower ESA-style loss \(L = \mathrm{MSE}_{HR}/F_2\). |
| G3 | Show calibrated high-risk probability. | Reliability plot exists; 70% warnings are roughly 70% positives. |
| G4 | Explain every forecast. | SHAP chart plus deterministic plain-language reasons. |
| G5 | Quantify uncertainty or abstain. | Interval or ensemble spread; REVIEW REQUIRED when the interval crosses the threshold. |
| G6 | Run on the presentation laptop without venue internet. | Demo works with Wi-Fi disabled while FastAPI and Next run locally. |
| G7 | Keep the human in control. | Disclaimer visible; wording is forecast / review, never command. |
| G8 | Be understandable in 90 seconds. | Judge can complete the curated event flow without coaching. |

### 4.2 Non-goals

| ID | Out of scope |
|---|---|
| NG1 | Flight-certified collision probability from first principles. |
| NG2 | Autonomous manoeuvre planning or thruster commands. |
| NG3 | Identifying real Indian satellites or debris objects. |
| NG4 | Training on live ISRO/ISTRAC operational data. |
| NG5 | A deep sequence model as the primary exhibit. |
| NG6 | Mixing TLE visualisation orbits with anonymized ESA events as if they were the same objects. |
| NG7 | Claiming ESA, ISRO, NASA, or ISTRAC endorsement. |
| NG8 | Optimising the original 2019 Kelvins leaderboard as if the competition were still open. |

---

## 5. Users and jobs to be done

| User | Job | Success |
|---|---|---|
| Competition judge | Understand the problem, see one forecast, and decide whether the science is honest. | Completes the 90-second path: queue → case → forecast → reason → reveal. |
| Domain expert (ISTRAC/ISRO-style) | Probe leakage, baseline, calibration, and failure cases. | Can open the model laboratory and inspect metrics, splits, and limitations. |
| Teacher / student | Learn why miss distance, covariance, and \(P_c\) are not the same thing. | Sees all three quantities together, with a frequency-style intuition aid. |
| Presenter | Recover if live inference fails. | Offline cached cases and a 90-second backup video exist. |

---

## 6. Product definition

### 6.1 One-sentence product

PRISM is an offline mission-control copilot that learns from historical CDM time series to forecast final conjunction risk 48 hours early, shows why, and tells the operator when it is unsure.

### 6.2 Judge-facing workflow

1. Choose one of five curated conjunction events.
2. Timeline is frozen at T-48 hours. No future CDMs are visible to the model or the UI.
3. See current reported risk, miss distance, relative speed, uncertainty, and observation history.
4. PRISM forecasts final \(\log_{10}(P_c)\) and converts it to approximate \(P_c = 10^{r}\) for display.
5. See calibrated probability that the final event is high-risk under the configured threshold.
6. See a 50% and 90% interval, or ensemble spread labelled as model spread if coverage is unmeasured.
7. Inspect a SHAP waterfall or compact contribution chart.
8. Read a plain-language explanation generated from verified feature directions.
9. Optionally reveal the actual final outcome.
10. Switch to the persistence baseline and see whether the model added value.

### 6.3 Example output

> **Forecast:** Final risk likely to rise to \(\log_{10}(P_c) = -5.4\).  
> **Configured warning probability:** 73% that final risk is \(\ge -6\).  
> **Confidence:** Medium; ensemble spans \(-6.1\) to \(-4.9\).  
> **Main reasons:** risk has risen across the last three messages, cross-track uncertainty is falling, and projected miss distance remains small.  
> **Recommendation:** prioritise another observation and begin contingency planning. Human approval is required.

Required wording: “forecast,” “configured warning,” “educational recommendation.”  
Forbidden wording: “collision will occur,” “manoeuvre now,” “ISRO threshold,” “certified.”

### 6.4 Curated demo cases

The exhibit must ship five frozen cases:

| Case | Story the judge should learn |
|---|---|
| Easy low risk | Persistence and model agree; risk stays negligible. |
| Clear escalation | Pre-T-48 trend rises; final risk is high; model beats persistence. |
| Clear de-escalation | Early risk looks alarming; later geometry / covariance say otherwise. |
| Uncertain / abstain | Interval crosses the threshold; UI returns REVIEW REQUIRED. |
| Model failure | Honest miss. Shown in the laboratory, not hidden. |

---

## 7. Risk language and thresholds

### 7.1 Probability conversion

Dataset risk is already logarithmic:

\[
r = \log_{10}(P_c), \qquad P_c = 10^{r}.
\]

| Log risk | Approximate \(P_c\) | Frequency-style aid |
|---:|---:|---:|
| -2 | \(10^{-2}\) | 1 in 100 |
| -4 | \(10^{-4}\) | 1 in 10,000 |
| -5 | \(10^{-5}\) | 1 in 100,000 |
| -6 | \(10^{-6}\) | 1 in 1,000,000 |
| -30 | \(\approx 0\) | Dataset floor used for negligible events |

Lead with log-risk and probability. Offer frequency only as intuition. Warn that a forecast probability is conditional on uncertain orbits and modelling assumptions.

### 7.2 Three distinct thresholds — do not collapse them

| Name | Default | Meaning | Where it appears |
|---|---|---|---|
| ESA challenge high-risk class | \(r \ge -6\) | Official classification used in Kelvins scoring. Chosen to keep more positives while staying near operational notification values. | Primary ML label, \(F_2\), \(\mathrm{MSE}_{HR}\), calibrated warning. |
| Typical LEO reaction band | \(10^{-5}\) to \(10^{-4}\) | ESA LEO missions often react in this band; \(10^{-4}\) is a common default, not a universal law. | Context in UI and model card. |
| Educational display band | Configurable; default overlays \(-6\), \(-5\), \(-4\) | Helps a judge see how rare true reaction-level events are. | UI only. Never claimed as an ISRO rule. |

The original challenge paper is explicit: many LEO operators use \(10^{-4}\) as a reaction threshold; ESA LEO missions use \(10^{-5}\) to \(10^{-4}\); notification is typically one order of magnitude lower. The competition used \(10^{-6}\) so the high-risk class was large enough to score.

### 7.3 Encounter-plane concept, for teaching only

Under a common simplified model, relative-position uncertainty is projected into the plane perpendicular to relative velocity. If the projected miss is \(m = [x_m, y_m]^\top\), combined 2D covariance is \(C\), and combined hard-body radius is \(R\):

\[
P_c = \iint_{(x-x_m)^2+(y-y_m)^2 \le R^2}
\frac{\exp\left(-\tfrac12 z^\top C^{-1} z\right)}{2\pi\sqrt{\lvert C\rvert}}\,dx\,dy,
\quad z=[x,y]^\top.
\]

PRISM does **not** need to reimplement this to meet its ML objective. The ESA dataset already supplies `risk`. If a numerical integrator is added for teaching, label it an approximation and validate against known cases.

A small miss distance does not automatically mean high \(P_c\). Large uncertainty spreads probability mass; later observations can shrink or move the ellipse. The UI must show miss distance, uncertainty, and probability together.

---

## 8. Data

### 8.1 Primary dataset: ESA Collision Avoidance Challenge

| Item | Value |
|---|---|
| Data page | https://kelvins.esa.int/collision-avoidance-challenge/data/ |
| Training download | https://kelvins.esa.int/media/public/competitions/collision-avoidance-challenge/train_data.zip |
| Testing download | https://kelvins.esa.int/media/public/competitions/collision-avoidance-challenge/test_data.csv |
| Challenge definition | https://kelvins.esa.int/collision-avoidance-challenge/challenge/ |
| Scoring | https://kelvins.esa.int/collision-avoidance-challenge/scoring/ |
| Design paper | Uriot et al., 2020/2021, arXiv:2008.03069 |
| Collection window | CDMs from ESA Space Debris Office support, 2015–2019 |
| Training rows / events | 162,634 rows, 13,154 events, ~12 CDMs per event |
| Official test rows / events | 24,484 rows, 2,167 events |
| Columns per CDM | 103 |
| Target / chaser | Target = ESA-supported satellite; chaser = conjuncting object |

The official test set withholds all CDMs with `time_to_tca < 2`. It was hand-picked to over-represent high-risk events and to require a last CDM within 1 day of TCA. It is **not** a random sample of the training distribution.

### 8.2 Context sources, not training data

- ISRO NETRA control centre: https://www.isro.gov.in/ISRO%20SSAControl%20Centre.html
- ISSAR 2025: https://www.isro.gov.in/Indian_Space_Situational_Awareness_Report_2025.html
- NASA CARA: https://www.nasa.gov/cara/
- CCSDS CDM Recommended Standard 508.0: https://ccsds.org/publications/allpubs/entry/3064/
- ESA CREAM automation programme: https://www.esa.int/Space_Safety/Space_Debris/CREAM_avoiding_collisions_in_space_through_automation

The CCSDS CDM defines how TCA, miss distance, relative state, covariance, and collision probability are exchanged. It does not prescribe the probability algorithm.

### 8.3 Optional visualisation data

Public TLEs (CelesTrak or Space-Track) may be used only to draw a generic Earth-orbit scene. They must never be joined to anonymized ESA `event_id` / `mission_id` values as if they described the same encounter. Synthetic orbit geometry is preferred for the competition build.

### 8.4 Provenance rules

- Keep downloaded archives unchanged in `data/raw/`.
- Record download date, URL, file size, and SHA-256 in `data/PROVENANCE.md`.
- Do not commit large raw files if git limits prohibit them; ship a download script instead.
- Keep licence / terms notices with any redistributed subset.
- Use only the anonymized identifiers supplied by the dataset.

---

## 9. Feature groups

Do not dump 103 columns into the UI. Group them.

### Event and timing

- `event_id`, `mission_id`, `time_to_tca` (days until closest approach)

### Current encounter state

- `risk`, `max_risk_estimate`, `max_risk_scaling`
- `miss_distance` (m), `relative_speed` (m/s)
- `relative_position_{r,t,n}`, `relative_velocity_{r,t,n}`
- `azimuth`, `elevation`, `geocentric_latitude`

### Object properties

- `c_object_type`
- Target/chaser (`t_*`, `c_*`) eccentricity, inclination, semi-major axis, apogee, perigee, RCS estimate, span, area-to-mass, energy dissipation rate

### Covariance

- Position/velocity sigmas such as `x_sigma_r`, `x_sigma_t`, `x_sigma_n`
- Correlation coefficients
- `x_position_covariance_det`

### Observation quality

- `x_obs_available`, `x_obs_used`
- `x_actual_od_span`, `x_recommended_od_span`
- Recency of last accepted observation
- `x_weighted_rms`, residuals accepted

### Space weather

- `F10`, `F3M`, `AP`, `SSN`

Space-weather features may matter through drag-induced uncertainty. Let evidence decide; do not assume importance.

---

## 10. Target construction and leakage prevention

Leakage prevention is the highest-priority methodological requirement.

### 10.1 One learning sample per event

For each `event_id`:

1. Sort rows by `time_to_tca` descending (moving toward TCA).
2. Target \(y_e\) = `risk` on the chronologically last row.
3. Input history = rows with `time_to_tca >= 2.0`.
4. T-48 snapshot = latest eligible row.
5. Trend features use eligible rows only.
6. Drop events with no eligible pre-T-48 row.
7. For the competition experiment, exclude any event whose labelled final row is not later than the selected snapshot.

### 10.2 Split by event, never by row

All CDMs from one event stay in the same fold. Row-wise splitting would leak later messages of the same encounter into training.

Required designs:

- Development: grouped splits on `event_id`.
- Final local test: hold out 20% of events before feature or model selection.
- Mission caution: `mission_id` clusters in PCA; attributes are not i.i.d. across missions. Train one model with it and one without. Keep it only if grouped mission tests show real generalization.
- Do not use official-test labels for training, validation, or hyperparameter choice. Local quality uses event-disjoint splits of the training archive. Official-test `true_risk` (Zenodo 4463683) is scored once after freeze (`metrics.json → officialTest`, `frozenBeforeLook: true`). Features come only from `test_data.csv` (`time_to_tca >= 2`). Do not retune after seeing that score.

### 10.3 Official test-set filters, reused locally

When constructing a realistic local evaluation set, prefer events that match the original test definition:

1. At least two CDMs.
2. Last CDM has `time_to_tca < 1`.
3. At least one CDM has `time_to_tca >= 2`, and no post-cutoff row is used as input.

---

## 11. Feature engineering

Create one vector \(X_e\) per event.

### 11.1 Latest-snapshot features

From the latest eligible CDM:

- current log-risk, max-risk estimate/scaling
- miss distance, relative speed, relative state
- target and chaser uncertainty fields
- orbital elements and object properties
- observation counts, residuals, OD spans
- space weather
- hours between the selected message and exact T-48

### 11.2 Trend features

For important numeric series \(z\) over eligible history:

- last, first, change, safe relative change
- mean, standard deviation, min, max
- OLS slope against `time_to_tca`
- change over last two and last three messages
- message count, hours since previous message
- monotonic increase / decrease flags

Fit \(z_i = \beta_0 + \beta_1 t_i + \epsilon_i\) and keep \(\beta_1\). Because `time_to_tca` decreases toward the event, define slope orientation in code and labels. Judge-facing text must say “risk rising toward TCA,” not a raw ambiguous sign.

### 11.3 Physics-informed derived features

Euclidean miss-distance check:

\[
d = \sqrt{r_R^2 + r_T^2 + r_N^2}
\]

Compare with `miss_distance` as a quality check. Do not silently replace the supplied value.

Relative-speed check:

\[
v_{rel} = \sqrt{v_R^2 + v_T^2 + v_N^2}
\]

Educational combined covariance, if both objects share a frame:

\[
C_{rel} = C_t + C_c
\]

Mahalanobis-like separation, useful as a feature, not as \(P_c\):

\[
D_M^2 = \Delta r^\top C_{rel}^{-1} \Delta r
\]

Logged uncertainty volume:

\[
u = \log(\det(C) + \epsilon)
\]

Observation usage ratio:

\[
\frac{\text{observations used}}{\max(\text{observations available}, 1)}
\]

### 11.4 Encoding and missingness

- One-hot encode `c_object_type` for linear models.
- Use native categoricals only if the exported artifact supports them.
- Trees may consume missing values; also emit missingness indicators for scientifically meaningful fields.
- Linear baselines: median impute fitted on training data only, then scale.

---

## 12. Model requirements

### 12.1 Required baselines

| Baseline | Definition |
|---|---|
| Persistence / LRP | Predict latest pre-T-48 `risk`. For ESA-style scoring, clip low-risk predictions to \(-6.001\). |
| Constant / median | Predict training-set median final risk, or the challenge’s constant \(-5\) reference. |
| Linear | Ridge or Elastic Net on imputed, scaled features. |

The model earns an “AI improvement” claim only if it beats persistence on the untouched event-level test set.

### 12.2 Primary model

Use `XGBRegressor` (LightGBM allowed as a comparison). Reasons: tabular data, missing values, nonlinear interactions, fast SHAP, existing portfolio skill.

Starting configuration, not a promised optimum:

```python
XGBRegressor(
    objective="reg:squarederror",
    n_estimators=600,
    learning_rate=0.03,
    max_depth=5,
    min_child_weight=8,
    subsample=0.8,
    colsample_bytree=0.75,
    reg_alpha=0.1,
    reg_lambda=2.0,
    random_state=42,
    n_jobs=-1,
)
```

Tune a small space inside grouped cross-validation. Do not run a huge sweep.

Optional scientifically useful target: predict the residual \(y_e - r_{e,-2}\) so the model is explicitly a correction to persistence.

### 12.3 Secondary warning view

\[
y_e^{cls} = \mathbb{1}[y_e \ge -6]
\]

Two valid implementations:

1. Derive warning probability from the regression ensemble / distribution.
2. Train `XGBClassifier`, then calibrate with `CalibratedClassifierCV` (sigmoid or isotonic) on disjoint validation predictions.

High-risk events are rare. Do not optimise accuracy. Use PR-AUC, recall at a fixed false-alert rate, class-weighted loss or `scale_pos_weight`, and choose the operating threshold on validation only. If oversampling is used, apply it only inside each training fold.

### 12.4 Uncertainty and abstention

Implement one, in this order:

1. 10–20 grouped-bootstrap tree models; report prediction quantiles.
2. Quantile regressors for low / median / high.
3. Conformal prediction on an untouched calibration split.

For the deadline, a small grouped ensemble is enough. Call it “model spread” unless interval coverage has been measured.

Return **REVIEW REQUIRED** when:

- the interval crosses the configured high-risk threshold
- critical features are missing
- the input is far outside the training distribution
- ensemble disagreement exceeds a validated limit

The original challenge analysis found that events missed by every top team had large orbital uncertainties and abrupt late jumps in risk. Abstention is more credible than false certainty on those cases.

### 12.5 Stretch, not primary

A GRU, TCN, or Transformer is optional. Short, irregular sequences, a strong persistence baseline, and the need for SHAP all argue against making a sequence model the exhibit. If attempted, compare it on the exact same held-out events.

Do not copy the 2019 winning heuristic cascade (threshold promotions plus test-set probing). That exploited a closed competition metric. PRISM must be a grouped, leakage-safe, explainable model.

---

## 13. Evaluation

### 13.1 Official ESA loss, reported as a research metric

\[
L(r,\hat{r}) = \frac{\mathrm{MSE}_{HR}(r,\hat{r})}{F_2}
\]

- High-risk class: \(r \ge -6\).
- \(\mathrm{MSE}_{HR}\) is computed only on true high-risk events.
- \(F_2\) uses \(\beta = 2\) to penalise false negatives more than false positives.

Reference numbers from the original challenge test set, after clipping low-risk predictions to \(-6.001\):

| System | \(L\) | \(\mathrm{MSE}_{HR}\) | \(F_2\) |
|---|---:|---:|---:|
| Constant \(-5\) | 2.5 | — | — |
| LRP baseline | 0.694 | 0.513 | 0.739 |
| Best 2019 team (sesc) | 0.556 | 0.407 | 0.733 |

PRISM will not reproduce the original hidden test ranking. Report \(L\) on **our** grouped hold-out. Still include it, because it is the metric ESA designed for this problem.

### 13.2 Regression metrics

- MAE, RMSE, median AE in log-risk units
- Error on high-risk events separately
- Share of predictions within 0.5 and 1.0 log units
- Improvement versus persistence

One log unit is a tenfold error in \(P_c\). Always state that interpretation.

### 13.3 Classification metrics

- PR-AUC as the main rare-event metric
- ROC-AUC as context
- Precision, recall, F1, confusion matrix at the chosen threshold
- Brier score, log loss, reliability plot
- Expected calibration error if implemented carefully

### 13.4 Robustness

- With and without `mission_id`
- By `c_object_type`
- By number of pre-cutoff CDMs
- By miss-distance and uncertainty bands
- When the latest message is older than exactly 48 hours
- Ablation: snapshot only vs snapshot+trends vs physics-informed extras

### 13.5 Error gallery

Publish, do not hide:

- five worst under-predictions
- five worst over-predictions
- missed high-risk events
- false escalations
- likely cause: missing data, unusual mission, rapidly changing covariance, or sparse history

---

## 14. Explainability

### 14.1 Global

Mean absolute SHAP over the hold-out set, grouped for judges:

- current estimated risk
- risk trend
- miss-distance geometry
- relative speed
- position uncertainty
- observation quality
- target / debris orbit properties
- space weather

### 14.2 Local

For one event show:

- base / average model output
- top three features pushing risk higher
- top three pushing it lower
- final predicted log-risk
- plain language from a reviewed feature-name dictionary

Do **not** use an LLM to invent the scientific explanation. Generate text deterministically from SHAP signs and the dictionary.

### 14.3 Optional what-if

A judge may edit miss distance or an uncertainty scale and see the new forecast. Label it “what-if model response.” Arbitrary edits can be physically inconsistent.

---

## 15. Functional requirements

Each requirement is mandatory unless marked Stretch.

### 15.1 Data pipeline

| ID | Requirement |
|---|---|
| D1 | Download ESA training data, checksum it, and write provenance. |
| D2 | Validate dtypes, ranges, duplicates, event ordering, and missingness. |
| D3 | Build one event-level table with a T-48 cutoff. |
| D4 | Automated tests prove no post-cutoff feature leakage. |
| D5 | Train / validation / calibration / test `event_id` sets are disjoint. |
| D6 | Derived miss-distance and speed checks have documented tolerances. |

### 15.2 Modelling

| ID | Requirement |
|---|---|
| M1 | Persistence, median/constant, and Ridge baselines exist. |
| M2 | Primary model is a grouped, seed-fixed gradient-boosted tree. |
| M3 | Chosen model beats persistence on the frozen hold-out set. |
| M4 | Warning probability is calibrated on out-of-fold predictions, never on test labels. |
| M5 | Uncertainty interval or ensemble spread is produced. |
| M6 | Abstention fires for the conditions in §12.4. |
| M7 | Serialized model reloads to matching predictions. |
| M8 | SHAP contributions sum to the model output within tolerance. |

### 15.3 Application

| ID | Requirement |
|---|---|
| A1 | FastAPI `POST /v1/risk/predict` accepts cutoff-safe CDM histories. |
| A2 | API rejects any message with `timeToTcaDays < 2`. |
| A3 | API returns forecast, \(P_c\), interval, warning probability, risk band, abstention, top factors, and disclaimer. |
| A4 | Next.js UI provides Event Queue, Case Workspace, and Model Laboratory. |
| A5 | Reveal-outcome is the only way to see post-T-48 truth. |
| A6 | Baseline-versus-model comparison is one click. |
| A7 | Five curated cases load from versioned JSON. |
| A8 | If FastAPI is stopped, the website shows an error instead of a silent JSON cache. |
| A9 | Disclaimer is visible on every prediction screen. |
| A10 | Colour is never the only risk encoding; text and icons are required. |

### 15.4 Exhibit operations

| ID | Requirement |
|---|---|
| E1 | Cold start on the presentation laptop is documented and rehearsed. |
| E2 | Wi-Fi-off test passes. |
| E3 | 90-second backup recording exists. |
| E4 | 3-minute and 7-minute scripts exist. |
| E5 | One-page research summary / model card is printable. |

---

## 16. Interface specification

### Screen 1 — Event queue

- Event ID and mission alias
- Time remaining to TCA
- Current reported log-risk
- Copilot forecast band
- Confidence / abstention
- Open case

### Screen 2 — Case workspace

- Status strip: TCA, current estimate, predicted final estimate, interval, configured warning probability
- Risk-history chart with a vertical T-48 cutoff and a hidden-future region
- Miss distance and uncertainty summary
- Observation count and recency
- Explanation panel
- Baseline versus model
- Reveal outcome

### Screen 3 — Model laboratory

- Held-out metrics including MAE vs LRP and ESA-style \(L\)
- Calibration / reliability curve
- Grouped feature importance
- Ablation table
- Failure-case gallery
- Provenance and limitations

### Visual language

- Near-black / navy mission-control background
- Cyan for observations and neutral telemetry
- Amber for review / uncertainty
- Red only for the highest configured warning band
- Monospaced numerals for telemetry; sans-serif for explanations

---

## 17. Architecture

```text
ESA raw CDMs
    → Python ingest + validate
    → Event builder at T-48 cutoff
    → Feature table
    → Grouped train / validation / calibration / test
    → T−48 XGBoost ensemble + calibration + SHAP
    → Versioned artifacts (model, schema, metrics, demo cases)
    → FastAPI inference (required)
    → Next.js exhibit (API-only; no JSON fallback)
```

### Repository layout

```text
PRISM/
  apps/
    web/                 # Next.js App Router UI
    api/                 # FastAPI
  ml/
    src/
      download.py
      validate.py
      build_events.py
      features.py
      split.py
      train_regressor.py
      train_classifier.py
      calibrate.py
      abstention.py
      explain.py
      evaluate.py
      export_demo_cases.py
    tests/
    artifacts/
      risk_regressor.json
      warning_calibrator.joblib
      feature_schema.json
      metrics.json
      demo_cases.json
  data/
    raw/                 # gitignored
    interim/
    processed/
    PROVENANCE.md
  docs/
  scripts/
    bootstrap.ps1
    download_data.ps1
    train.ps1
    run_demo.ps1
  prd.md
  README.md
  .env.example
```

No database is required for the competition build. Curated cases and cached explanations live in versioned JSON. A hosted Convex/Postgres backend is a post-competition portfolio option, not a venue dependency.

### API contract

`POST /v1/risk/predict`

Request:

```json
{
  "eventId": "demo-1042",
  "cutoffHours": 48,
  "messages": [
    {
      "timeToTcaDays": 2.18,
      "riskLog10": -5.2,
      "missDistanceM": 438,
      "relativeSpeedMps": 12640
    }
  ]
}
```

Response:

```json
{
  "predictedFinalRiskLog10": -5.4,
  "predictedFinalPc": 3.98e-6,
  "interval90Log10": [-6.1, -4.9],
  "configuredHighRiskProbability": 0.73,
  "highRiskThresholdLog10": -6,
  "riskBand": "review",
  "abstained": false,
  "topFactors": [
    {"feature": "risk_trend_last3", "direction": "higher", "contribution": 0.61},
    {"feature": "normalized_separation", "direction": "higher", "contribution": 0.34},
    {"feature": "miss_distance", "direction": "lower", "contribution": -0.21}
  ],
  "disclaimer": "Educational research prototype; not for operational decisions."
}
```

The API validates names, units, ranges, and cutoff.

---

## 18. Tech stack

Aligned with the existing portfolio: Next.js, TypeScript, React, Tailwind, shadcn/ui, Python, scikit-learn, XGBoost, LightGBM, SHAP, pandas, NumPy.

| Layer | Choice |
|---|---|
| UI | Next.js App Router, TypeScript, React, Tailwind, shadcn/ui |
| Charts | Recharts or ECharts |
| Client state | Zustand |
| Runtime validation | Zod |
| ML | Python 3.11+, pandas, NumPy, scikit-learn, XGBoost, SHAP |
| Optional comparison | LightGBM |
| Plots for the report | Matplotlib / Seaborn |
| Artifacts | joblib + XGBoost JSON/UBJ |
| API | FastAPI, Pydantic, Uvicorn |
| Python tests | pytest |
| UI tests | Vitest + React Testing Library |
| Judge-flow test | Playwright |
| Lint | Ruff, ESLint, Prettier |
| Deploy | Offline laptop first; Vercel + separate API only after the event |

Venue internet is not promised. Offline is mandatory.

---

## 19. Non-functional requirements

| ID | Requirement |
|---|---|
| N1 | Cold start to first curated case in under 60 seconds after processes are up. |
| N2 | Cached case load under 200 ms. |
| N3 | Live inference for one event under 2 seconds on the presentation laptop. |
| N4 | Works at 1920×1080 and on a projector. |
| N5 | Keyboard path exists for queue → open → reveal. |
| N6 | Colour-blind-safe risk encoding. |
| N7 | Laptop exhibit may run with Wi-Fi off; FastAPI must still be local. No silent JSON fallback. |
| N8 | All random seeds fixed and recorded in `metrics.json`. |

---

## 20. Model card

**Intended use.** Education, interpretability demonstration, and research into early conjunction-risk forecasting on public ESA challenge data.

**Out of scope.** Spacecraft operations, autonomous manoeuvres, claims about specific real objects, safety-critical decisions, or generalization to current Indian operational data without retraining and independent validation.

**Known limitations.**

- Historical, anonymized 2015–2019 ESA-supported missions, not the live Indian catalogue.
- Official Kelvins test set over-represents high-risk events; local splits must say so.
- Public features omit operational context agencies actually use.
- \(P_c\) depends on covariance quality and assumptions.
- SHAP explains the trained model, not physical causality.
- Educational thresholds are configurable and are not ISRO rules.
- A forecast can be confident and wrong outside the training distribution.
- Persistence is a strong baseline; gains, if any, will likely be modest.

**Human control.** Every recommendation is a request for analyst review, another observation, or contingency planning — never an instruction to fire thrusters.

---

## 21. Deliverables

### Required

- Working offline web demonstrator
- Reproducible ingest, cutoff, training, and evaluation code
- Frozen model artifact and feature schema
- Model card
- Evaluation report with baselines, grouped split, calibration, and failures
- Five curated demonstration events
- 90-second backup screen recording
- 3-minute and 7-minute presentation scripts
- Source and provenance list

### Stretch

- Public portfolio deployment
- Interactive what-if panel
- Formal conformal intervals (research surface measured; exhibit still shows bootstrap spread)
- Mission-holdout experiment
- Physical LED risk indicator driven by the API

---

## 22. Build plan (remaining calendar)

Today is **15 August 2026**. The exhibit target is **18 August 2026**. Freeze the demonstrator by the evening of **17 August**. Do not change the model after freeze.

| When | Outcome |
|---|---|
| 15 Aug | Data downloaded and checksummed. Event-level cutoff pipeline + leakage tests. Persistence, median, and Ridge baselines. First XGBoost snapshot model. |
| 16 Aug | Trend + physics features, grouped evaluation, model selection, calibration, ensemble spread, SHAP, five demo cases exported. |
| 17 Aug | FastAPI + Next.js queue/workspace/laboratory, first freeze of bootstrap exhibit. |
| 18 Aug | Restored T−48 ensemble freeze (MAE 3.059); API-only website; documentation. |

If an organiser later names 20 August as a paperwork deadline, 18–19 August are for documentation polish only.

---

## 23. Testing checklist

### Data

- [ ] One processed row per unique event
- [ ] No input message violates T-48
- [ ] Final target value never appears in engineered inputs
- [ ] Split IDs are disjoint
- [ ] Units are explicit

### Model

- [ ] Fixed seed reproduces metrics within tolerance
- [ ] Chosen model beats persistence on the untouched test set
- [ ] Reloaded artifact matches live predictions
- [ ] SHAP sums match output within tolerance
- [ ] Calibration unused test labels
- [ ] Missing / out-of-range inputs are safe

### Interface

- [ ] No post-T-48 information before Reveal outcome
- [ ] \(P_c = 10^{r}\) conversion is correct
- [ ] Risk is never colour-only
- [ ] Charts show units
- [ ] API-down path shows an error (no JSON fallback)
- [ ] Disclaimer on every prediction screen

---

## 24. Presentation

### 90-second path

1. **Problem.** Early risk estimates move as observations arrive; operators still need time to prepare.
2. **Data.** ESA CDM histories, cut off 48 hours before closest approach.
3. **Model.** Event-level XGBoost on T−48 summaries, calibrated warning probability, SHAP.
4. **Demo.** Event, forecast, reason, uncertainty, revealed outcome.
5. **Evidence.** Comparison with latest-risk baseline on held-out events.
6. **Limit.** Advisory educational prototype, never an autonomous flight system.

### Likely judge questions

**Why XGBoost instead of a neural network?**  
Structured, medium-sized, missing-valued tabular data. Trees iterate fast and explain with SHAP. A deeper model must beat this on the same hold-out to earn complexity.

**Are you computing collision probability from scratch?**  
No. We forecast the final risk already reported in the ESA event history, from data available by T-48 hours. The encounter-plane integral is teaching context only.

**How did you stop the model seeing the future?**  
Inputs use only rows with `time_to_tca >= 2`. The target is a later CDM. Splits are by whole event.

**What does SHAP prove?**  
Which inputs moved this trained model’s output. Not physical causation.

**Why calibration?**  
A 70% warning should mean about 70% of similar cases were high-risk. An uncalibrated score is just a ranking.

**Did you beat the ESA 2019 winners?**  
We are not on their hidden test set. We report leakage-safe local metrics against persistence, which only 12 of 96 original teams beat.

---

## 25. Success criteria

The project is competition-ready when all of the following are true:

1. It runs without internet on the presentation laptop.
2. The T-48 cutoff is real; tests prove no event leakage.
3. The chosen model beats persistence on an untouched event-level test set.
4. The warning view includes a calibration plot.
5. Every forecast has an explanation and an uncertainty or abstention state.
6. A judge can understand the problem and result in 90 seconds.
7. Claims stay educational and scientifically modest.
8. All five curated cases load.
9. Backup video and static demo mode work.

---

## 26. Sources

- ESA Collision Avoidance Challenge data: https://kelvins.esa.int/collision-avoidance-challenge/data/
- Challenge definition: https://kelvins.esa.int/collision-avoidance-challenge/challenge/
- Scoring: https://kelvins.esa.int/collision-avoidance-challenge/scoring/
- Uriot, Izzo, Simões, et al. *Spacecraft Collision Avoidance Challenge: design and results of a machine learning competition*. arXiv:2008.03069
- CCSDS CDM Recommended Standard: https://ccsds.org/publications/allpubs/entry/3064/
- ISRO NETRA: https://www.isro.gov.in/ISRO%20SSAControl%20Centre.html
- ISSAR 2025: https://www.isro.gov.in/Indian_Space_Situational_Awareness_Report_2025.html
- NASA CARA: https://www.nasa.gov/cara/
- ESA CREAM: https://www.esa.int/Space_Safety/Space_Debris/CREAM_avoiding_collisions_in_space_through_automation
- scikit-learn calibration: https://scikit-learn.org/stable/modules/calibration.html
- SHAP: https://shap.readthedocs.io/
- Builder portfolio / stack: https://arjuncodess.is-a.dev/

---

## 27. Locked product statement

**PRISM, Predictive Risk Intelligence for Space Monitoring, is an explainable early-warning research prototype. It learns from conjunction-message histories to forecast final collision risk 48 hours in advance, says when it is uncertain, and keeps the human operator visibly in control.**
