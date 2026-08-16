# PRISM presentation notes

Print this. Keep it next to the keyboard. Numbers stay the same even if your brain does not.

## If you blank

Say this and then open the abstention case:

"Two objects might get close. The reported collision chance keeps changing as new tracking data arrives. Waiting gives a better number, and it also eats planning time. PRISM stops the clock 48 hours early and asks: can the history of those reports beat just copying today's value? On average, yes. On the ESA high-risk score, it ties, because if today's report is already at -6 or worse, PRISM copies it on purpose. The ranges on screen are model spread. They are not a 90 percent promise. This is a school research exhibit. A person still has to review it."

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
5. Fit XGBoost. Also fit median, Ridge, and persistence.
6. Take the median of ten bootstrap XGBoost models. If the current report is already at or above -6, copy that report. That is the persistence guard.
7. A separate calibration split maps the forecast to "chance this later report is high-risk."
8. Abstain when the 90 percent bootstrap band crosses -6, a critical field is missing, or the models disagree by more than 1.25 log units.

You can say "XGBoost on named history features" if someone asks why it is not a transformer. 8,293 events. Lots of missing cells. We wanted to read the reasons. A sequence model can wait until it wins on the same frozen split.

## The result you should lead with

Held-out test, 1,659 events. MAE is in `log10(Pc)` units.

- Persistence MAE: 5.080
- PRISM MAE: 3.052 (about 40 percent lower)
- ESA-style loss: 0.167 for both
- F2: 0.361 for both

ESA-style loss is high-risk MSE divided by F2. F2 is F-beta with beta 2, so missing a high-risk event hurts more than a false alarm.

The exact tie is not a bug. The guard copies today's report once it is already at or above -6. That is exactly the region ESA-style loss cares about. Therefore PRISM matches persistence there on purpose. The MAE win is mostly the boring middle of the risk range.

One sentence that carries the whole talk:

"The model has useful signal beyond persistence, especially when the forecast is early, but that signal does not automatically become better high-risk decisions or honest uncertainty."

Say that. Then show the horizon table.

## Horizon table (put this on a slide)

Waiting helps persistence more than it helps PRISM. Learned forecasting is most useful when information is sparse.

| Horizon | XGBoost MAE | Persistence MAE |
| --- | ---: | ---: |
| T-72 | 3.241 | 7.748 |
| T-48 | 2.838 | 5.080 |
| T-24 | 2.097 | 2.634 |
| T-12 | 1.390 | 1.444 |

At T-72 the gap is huge. At T-12 it is basically gone. That is the finding people remember.

## Other numbers worth having in a pocket

- History vs snapshot (single XGBoost): 2.960 snapshot, 2.842 after history, 2.838 after covariance trends. History is temporal transforms of snapshot fields, so that bump is evolution, not random extra columns.
- Abstention: 78.2 percent coverage, 21.8 percent sent to review, accepted MAE 1.920. All nine test high-risk events are flagged or sent to review.
- False reassurance: an accepted forecast below -6 while the final report is at or above -6. Count is zero.
- Nominal 90 percent bootstrap band covers 48.6 percent of outcomes. 50 percent band covers 26.4 percent. Hence we call it model spread.
- Mission holdout high-risk MAE: 18.1, on one held-out high-risk event. Random event splits do not prove the model works on a new spacecraft family.
- `mission_id` changes MAE from 2.838 to 2.835. We left it out of the exhibit.
- Failures: over-prediction 362, under-prediction 210, sparse history 62, collapse to the -30 floor 40, late high-risk jumps 2.
- When the single XGBoost is off by two or more log units, tracking-completeness features have higher mean |SHAP| than on accurate cases. That is association in the explanation, not a physics proof.

## What to click

Use the abstention case for the short path. Use the lab if they want receipts.

### 90 seconds

Open the abstention case.

"These six cases are real ESA events: two low, one that needs a person, three high. The page is frozen before T-48. You see today's reported risk, the live forecast of the later report, the high-risk estimate, and the spread across ten bootstrap models. The band crosses -6, so PRISM says review required. SHAP tells us why this fitted model moved. It does not tell us what physically caused the encounter. The future is hidden on purpose."

Click Reveal.

"Later messages and the final reported value show up now. Reveal is the only way to load that truth."

If there is time: "In the lab, MAE drops from 5.080 to 3.052. ESA-style loss ties at 0.167 because of the guard. The 90 percent band only covers 48.6 percent of outcomes."

### 3 minutes

Add the data path (162,634 to 8,293 to 1,659), the leakage rule, persistence as the baseline, and the horizon story. Then the lab. Then the mission holdout: 18.1. Then sit down.

### 7 minutes

Walk the same path slower. Open the lab tables in this order:

1. Horizon. "Useful early. Persistence catches up late."
2. History ablation. "Evolution of the same fields. Covariance trends barely help after that."
3. Abstention. "78.2 percent coverage. Define false reassurance. Zero of those."
4. Spread coverage. "48.6 percent is badly off. We did not relabel it as confidence."
5. Failures and SHAP contrast.

Close with the one-sentence finding. Credit ESA, ISRO NETRA, NASA CARA, and CCSDS as context. They did not endorse this project. You built it at City Montessori School, Kanpur Road Campus.

## How the live website works

The six cases are curated real events, not made-up rows. After deploy, the site should set `NEXT_PUBLIC_API_URL` and talk to FastAPI.

- The queue and the lab load cases and metrics from the API.
- Opening a case runs `POST /v1/risk/predict` on the cutoff-safe messages. That is a live model call.
- Reveal still hides the later messages until you ask. That part is a research control, not a fake forecast.

If the API is down, the production site should fail in the open. Local Next without the env file can still read the frozen JSON so you can rehearse offline.

## Answers when they interrupt you

**Did you beat the 2019 ESA winners?** We cannot say. Official test labels are still hidden. We report our own frozen split.

**Why is MAE so big?** Some events jump between the dataset floor of -30 and a finite risk. A few of those wreck the mean. Median absolute error is 0.474. That is why both numbers exist.

**Is 48.6 percent okay for a 90 percent interval?** No. That is the point. We labeled it model spread and used it to abstain.

**Why keep persistence?** It protects the high-risk tail. Dropping the guard would make average error look nicer and would weaken the part ESA actually scores.

**Why XGBoost?** Named features. Missing values. 8,293 rows. We wanted explanations we can read.

**Does this fly a satellite?** It is a research exhibit. A qualified person has to review anything that looks like a decision.

**Would it work on a mission you have never seen?** We tried holding out four missions. The rare high-risk tail was weak. High-risk MAE 18.1 on one event. That is an honest miss.

## Rehearsal still on you

Wi-Fi off. Projector at 1920x1080. Time a cold start. Time a case load. Time one live predict. Record a 90-second backup video on the presentation laptop. The repo cannot do that part for you.
