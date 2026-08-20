# PRISM presentation notes

Print this. Keep it next to the keyboard. Numbers stay the same even if your brain does not.

Selected policy: T−48 floor hurdle (classifier for later −30, residual XGBoost otherwise). No persist guard. Conformal interval.

## Two beats

1. **Here is the model.** Open a case. Today's report, the live forecast of the later report, conformal band or review required, SHAP in plain language.
2. **Here is what the study measured.** Horizon decay, floor anatomy, official-test *L*. The 18 August ensemble (MAE 5.080 → 3.059, ESA loss tie 0.167) is a snapshot row, not a live mode.

## If you blank

Say this and then open the review-required case:

"Two objects might get close. The reported collision chance keeps changing as new tracking data arrives. Waiting gives a better number, and it also eats planning time. PRISM stops the clock 48 hours early. The live model can call a later collapse to the dataset floor, or it can adjust today's report. On average MAE falls from 5.080 to 2.109, mostly by calling those floor jumps. It does not copy today's report just because that report is already at −6. The band is a conformal interval. This is a school research exhibit. A person still has to review it."

Then click Reveal. Breathe. Keep going.

## What this project actually is

PRISM is Predictive Risk Intelligence for Space Monitoring. Fancy name. Simple job.

It forecasts the later reported `log10(Pc)` from messages you would have had 48 hours before closest approach. That later number is a report from the conjunction process. It is not the true physical probability of a crash, and PRISM does not compute orbits from first principles.

The baseline is persistence. Persistence means: take the latest pre-cutoff risk and copy it forward. If a learned model cannot beat that, it is not adding much.

Data comes from the public ESA Collision Avoidance Challenge archive (2015 to 2019). Real CDMs. Anonymized events. 162,634 rows. 13,154 events. After the cutoff rule, 8,293 events are eligible. The frozen test split has 1,659 events.

T-48 exists because the ESA challenge test file only has messages with `time_to_tca` of at least 2 days. Hence we freeze there. We also report T-72, T-24, and T-12 so you can see what waiting does.

The high-risk line `log10(Pc) >= -6` is the ESA scoring class. It is not an ISRO operations rule. Only 66 eligible events meet it. Nine of those sit in the test split. Therefore that probability meter is a scarce-label estimate. Treat it that way when you talk.

## How the pipeline works

1. Keep only messages at or before the cutoff.
2. The later update is the label. It never goes into features.
3. Train, validation, calibration, and test use different event IDs. Validation is not reused to pick the test set.
4. Each event becomes one row of inspectable summaries: latest snapshot, plus slopes, changes, ranges, and observation-count history of those same snapshot fields.
5. Fit XGBoost on the final reported `log10(Pc)`. Also fit median, Ridge, and persistence.
6. Fit a floor classifier and a residual XGBoost on non-floor training events. Threshold 0.15 on validation. Do not copy today's −6 report.
7. Fit split conformal on calibration around that point. Abstain if the 90 percent band crosses −6 or a field is missing.
8. Keep the 18 August bootstrap ensemble in the metrics tables as a snapshot.

You can say "XGBoost on named T-48 history features" if someone asks why it is not a transformer. 8,293 events. Lots of missing cells. We wanted to read the reasons. A sequence model can wait until it wins on the same frozen split.

## The result you should lead with

Held-out test, 1,659 events. MAE is in `log10(Pc)` units.

- Persistence MAE: 5.080
- Live floor hurdle MAE: 2.109 (median AE 0; F2 = 0)
- August snapshot MAE: 3.059; ESA-style loss 0.167 for persistence and that snapshot
- F2: 0.361 for the snapshot; 0 for the live floor hurdle

ESA-style loss is high-risk MSE divided by F2. F2 is F-beta with beta 2, so missing a high-risk event hurts more than a false alarm.

The exact tie is not a bug. The guard copies today's report once it is already at or above -6. That is exactly the region ESA-style loss cares about. Therefore PRISM matches persistence there on purpose. The MAE win is mostly the boring middle of the risk range.

One sentence that carries the whole talk:

"The model has useful signal beyond persistence, especially when the forecast is early, but that signal does not automatically become better high-risk decisions or honest uncertainty."

Say that. Then show the horizon table.

## Horizon table (put this on a slide)

Waiting helps persistence more than it helps PRISM. Learned forecasting is most useful when information is sparse.

| Horizon | XGBoost MAE | Persistence MAE |
| --- | ---: | ---: |
| T-72 | 3.214 | 7.748 |
| T-48 | 2.808 | 5.080 |
| T-24 | 2.110 | 2.634 |
| T-12 | 1.384 | 1.444 |

At T-72 the gap is huge. At T-12 it is basically gone. That is the finding people remember.

## Other numbers worth having in a pocket

- History vs snapshot (single XGBoost): 2.904 snapshot, 2.851 after history, 2.808 after covariance trends. History is temporal transforms of snapshot fields, so that bump is evolution, not random extra columns.
- Selected ensemble MAE at T-48: 3.059. Unguarded XGBoost: 2.808, F2 = 0.
- Abstention: 77.7 percent coverage, 370 sent to review, accepted MAE 1.902.
- False reassurance: an accepted forecast below -6 while the final report is at or above -6. Count is **one**.
- Nominal 90 percent bootstrap band covers 47.7 percent of outcomes. 50 percent band covers 25.8 percent. Hence we call it model spread.
- Mission holdout overall MAE: 2.688 vs persistence 4.843. High-risk MAE 19.2 on one held-out high-risk event.
- `mission_id` changes unguarded MAE from 2.808 to 2.840. We left it out of the exhibit.
- Failures: over-prediction 366, under-prediction 209, moderate 97, sparse history 63, collapse to -30 39, close approach 26, false high-risk 20, late high-risk jumps 2. Accurate within 0.5: 837.

## What to click

Use the abstention case for the short path. Use the lab if they want receipts.

Start FastAPI and Next before anyone sits down. `NEXT_PUBLIC_API_URL` must be set. If the API is down, the page errors. That is intended.

### 90 seconds

Open the abstention case.

"These six cases are real ESA events: two low, one that needs a person, three already high today. The page is frozen before T-48. You see today's reported risk, the live floor-hurdle forecast, the high-risk estimate, and a conformal interval. SHAP tells us why the residual model moved. It does not tell us what physically caused the encounter. The future is hidden on purpose."

Click Reveal.

"Later messages and the final reported value show up now. Reveal is the only way to load that truth."

If there is time: "In the lab, live MAE is 2.109. The August snapshot was 3.059 with an ESA-style loss tie at 0.167. The live model does not keep that tie."

### 3 minutes

Add the data path (162,634 to 8,293 to 1,659), the leakage rule, persistence as the baseline, and the horizon story. Then the lab. Then false reassurance = 1. Then the mission holdout: 19.2. Then sit down.

### 7 minutes

Walk the same path slower. Open the lab tables in this order:

1. Horizon. "Useful early. Persistence catches up late."
2. History ablation. "Evolution of the same fields. Covariance trends help a little after that."
3. Abstention. "77.7 percent coverage. Define false reassurance. One of those."
4. Spread coverage. "47.7 percent is badly off. We did not relabel it as confidence."
5. Failures and SHAP contrast.

Close with the one-sentence finding. Credit ESA, ISRO NETRA, NASA CARA, and CCSDS as context. They did not endorse this project. You built it at City Montessori School, Kanpur Road Campus.

## How the live website works

The six cases are curated real events, not made-up rows. The site requires `NEXT_PUBLIC_API_URL` and talks to FastAPI.

- The queue and the lab load cases and metrics from the API.
- Opening a case runs `POST /v1/risk/predict` on the cutoff-safe messages. That is a live T-48 model call.
- Reveal still hides the later messages until you ask. That part is a research control, not a fake forecast.

If the API is down, the site fails in the open. There is no frozen JSON fallback. For a Wi-Fi-off rehearsal, keep FastAPI running on the laptop.

## Answers when they interrupt you

**Did you beat the 2019 ESA winners?** We cannot say. Official test labels are still hidden. We report our own frozen split.

**Why is MAE so big?** Some events jump between the dataset floor of -30 and a finite risk. A few of those wreck the mean. Median absolute error on the selected policy is 0.474. That is why both numbers exist.

**Is 47.7 percent okay for a 90 percent interval?** No. That is the point. We labeled it model spread and used it to abstain.

**Why keep persistence?** It protects the high-risk tail. Dropping the guard would make average error look nicer and would weaken the part ESA actually scores.

**Why XGBoost?** Named features. Missing values. 8,293 rows. We wanted explanations we can read.

**Does this fly a satellite?** It is a research exhibit. A qualified person has to review anything that looks like a decision.

**Would it work on a mission you have never seen?** We tried holding out four missions. The rare high-risk tail was weak. High-risk MAE 19.2 on one event. That is an honest miss.

## Rehearsal still on you

Wi-Fi off is fine if FastAPI and Next are both local. Projector at 1920x1080. Time a cold start. Time a case load. Time one live predict. Record a 90-second backup video on the presentation laptop. The repo cannot do that part for you.
